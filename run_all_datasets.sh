#!/usr/bin/env bash
# =============================================================================
# run_all_datasets.sh — Train and evaluate on MELD, DailyDialog, IEMOCAP
#
# Each dataset is trained with the best known config (no aux losses, staged).
# Per-emotion F1 / accuracy / weighted-F1 saved to multi_dataset_results.csv
#
# Usage:
#   bash run_all_datasets.sh [--device cuda] [--batch_size 8] [--skip_iemocap]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEVICE="cuda"
BATCH_SIZE=8
SKIP_IEMOCAP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)       DEVICE="$2";      shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";  shift 2 ;;
    --skip_iemocap) SKIP_IEMOCAP=1;   shift   ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

PYTHON=""
for c in python3 python; do
  if command -v "$c" &>/dev/null; then PYTHON="$c"; break; fi
done
[[ -z "$PYTHON" ]] && { echo "[ERROR] Python 3 not found"; exit 1; }

RESULTS_CSV="./multi_dataset_results.csv"
CKPT_BASE="./checkpoints_multidataset"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; BOLD='\033[1m'; RED='\033[0;31m'; NC='\033[0m'
log()    { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
warn()   { echo -e "\033[1;33m[WARN]\033[0m $*"; }
header() {
  echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}\n"
}

# ── Run one dataset ───────────────────────────────────────────────────────────
# Args: dataset_key  display_name  [extra python args ...]
run_dataset() {
  local key="$1"
  local name="$2"
  shift 2
  local out_dir="$CKPT_BASE/$key"
  local log_file="$out_dir/train.log"
  mkdir -p "$out_dir"

  header "$name"

  local exit_code=0
  $PYTHON main.py --mode train \
    --batch_size    "$BATCH_SIZE"  \
    --device        "$DEVICE"      \
    --output_dir    "$out_dir"     \
    --staged_training              \
    --posterior_loss_weight  0.0   \
    --surprise_reg_weight    0.0   \
    --contrastive_loss_weight 0.0  \
    --future_pred_weight_1   0.0   \
    --future_pred_weight_2   0.0   \
    "$@" 2>&1 | tee "$log_file" || exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo -e "${RED}[FAILED]${NC} $name training failed (exit $exit_code)"
    return 1
  fi
  ok "$name training done"
}

# ── Check / prep dataset ──────────────────────────────────────────────────────
ensure_csv() {
  local dataset="$1"
  local out_dir="$2"
  if [[ -f "$out_dir/train.csv" && -f "$out_dir/val.csv" && -f "$out_dir/test.csv" ]]; then
    ok "Data already present: $out_dir"
    return 0
  fi
  log "Downloading and preprocessing $dataset..."
  $PYTHON prep_data.py --dataset "$dataset" --out_dir "$out_dir"
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. MELD  (direct HuggingFace — no preprocessing needed)
# ─────────────────────────────────────────────────────────────────────────────
run_dataset "meld" "MELD (eusip/silicone meld_e)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. DailyDialog
# ─────────────────────────────────────────────────────────────────────────────
DAILYDIALOG_DIR="./data/dailydialog"
ensure_csv "dailydialog" "$DAILYDIALOG_DIR"

run_dataset "dailydialog" "DailyDialog" \
  --local_data      "$DAILYDIALOG_DIR" \
  --utterance_col   "Utterance"        \
  --speaker_col     "Speaker"          \
  --emotion_col     "Emotion"          \
  --dialogue_id_col "Dialogue_ID"

# ─────────────────────────────────────────────────────────────────────────────
# 3. IEMOCAP  (optional — skip with --skip_iemocap)
# ─────────────────────────────────────────────────────────────────────────────
IEMOCAP_DIR="./data/iemocap"
if [[ $SKIP_IEMOCAP -eq 1 ]]; then
  warn "Skipping IEMOCAP (--skip_iemocap set)"
elif [[ ! -f "$IEMOCAP_DIR/train.csv" ]]; then
  log "Attempting IEMOCAP download..."
  $PYTHON prep_data.py --dataset iemocap --out_dir "$IEMOCAP_DIR" || {
    warn "IEMOCAP download failed — skipping. Run 'huggingface-cli login' and retry."
    SKIP_IEMOCAP=1
  }
fi

if [[ $SKIP_IEMOCAP -eq 0 && -f "$IEMOCAP_DIR/train.csv" ]]; then
  run_dataset "iemocap" "IEMOCAP" \
    --local_data      "$IEMOCAP_DIR" \
    --utterance_col   "Utterance"    \
    --speaker_col     "Speaker"      \
    --emotion_col     "Emotion"      \
    --dialogue_id_col "Dialogue_ID"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Compile results CSV
# ─────────────────────────────────────────────────────────────────────────────
header "Compiling Results → $RESULTS_CSV"

$PYTHON - "$CKPT_BASE" "$RESULTS_CSV" <<'PYEOF'
import os, json, csv, sys

ckpt_base   = sys.argv[1]
results_csv = sys.argv[2]

DATASET_NAMES = {"meld": "MELD", "dailydialog": "DailyDialog", "iemocap": "IEMOCAP"}

summary_rows  = []   # one row per dataset (overall metrics)
emotion_rows  = []   # one row per dataset × emotion

for key, display in DATASET_NAMES.items():
    metrics_path = os.path.join(ckpt_base, key, "test_metrics.json")
    if not os.path.isfile(metrics_path):
        print(f"  [SKIP] No test_metrics.json for {display}")
        continue

    with open(metrics_path) as f:
        m = json.load(f)

    summary_rows.append({
        "dataset":        display,
        "accuracy":       round(m.get("accuracy",     0), 4),
        "weighted_f1":    round(m.get("weighted_f1",  0), 4),
        "posterior_f1":   round(m.get("posterior_f1", 0) or 0, 4),
        "mean_surprise":  round(m.get("mean_surprise",0), 4),
        **{f"bucket_f1_{k}": round(v, 4)
           for k, v in m.get("bucket_f1", {}).items()},
    })

    for emo, stats in m.get("per_emotion", {}).items():
        emotion_rows.append({
            "dataset":   display,
            "emotion":   emo,
            "precision": round(stats.get("precision", 0), 4),
            "recall":    round(stats.get("recall",    0), 4),
            "f1":        round(stats.get("f1-score",  0), 4),
            "support":   int(stats.get("support",     0)),
        })

# ── Summary table ──────────────────────────────────────────────────────────
print("\n  Overall Metrics:")
if summary_rows:
    cols = list(summary_rows[0].keys())
    w = max(len(c) for c in cols)
    print("  " + "  ".join(f"{c:<{max(w,10)}}" for c in cols))
    print("  " + "  ".join("-" * max(w,10) for c in cols))
    for r in summary_rows:
        print("  " + "  ".join(f"{str(r[c]):<{max(w,10)}}" for c in cols))

# ── Per-emotion table ──────────────────────────────────────────────────────
print("\n  Per-Emotion Metrics:")
if emotion_rows:
    print(f"  {'Dataset':<14} {'Emotion':<12} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Support':>9}")
    print(f"  {'-'*14} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
    for r in emotion_rows:
        print(f"  {r['dataset']:<14} {r['emotion']:<12} "
              f"{r['f1']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['support']:>9}")

# ── Save CSVs ──────────────────────────────────────────────────────────────
if summary_rows:
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)
    print(f"\n  Summary CSV → {results_csv}")

emotion_csv = results_csv.replace(".csv", "_per_emotion.csv")
if emotion_rows:
    with open(emotion_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset","emotion","precision","recall","f1","support"])
        writer.writeheader(); writer.writerows(emotion_rows)
    print(f"  Per-emotion CSV → {emotion_csv}")
PYEOF

echo ""
ok "All done. Results in $RESULTS_CSV"
