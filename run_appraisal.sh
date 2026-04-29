#!/usr/bin/env bash
# =============================================================================
# run_appraisal.sh — Train and evaluate on the appraisal theory dataset
#
# Source: https://github.com/shuiguolanzi397/Emotion-Prediction-Dataset
# Contains MELD + EmoryNLP + DailyDialog filtered for appraisal-rich dialogues
# (min 4 distinct emotions, min 2 emotion transitions per dialogue).
#
# Runs 4 configurations:
#   1. appraisal_meld        — MELD subset
#   2. appraisal_emorynlp    — EmoryNLP subset
#   3. appraisal_dailydialog — DailyDialog subset
#   4. appraisal_all         — All 3 combined
#
# Usage:
#   bash run_appraisal.sh [--device cuda] [--batch_size 8] [--sources meld emorynlp dailydialog all]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEVICE="cuda"
BATCH_SIZE=8
SOURCES=("meld" "emorynlp" "dailydialog" "all")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)      DEVICE="$2";      shift 2 ;;
    --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
    --sources)
      shift
      SOURCES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SOURCES+=("$1"); shift
      done ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

PYTHON=""
for c in python3 python; do
  if command -v "$c" &>/dev/null; then PYTHON="$c"; break; fi
done
[[ -z "$PYTHON" ]] && { echo "[ERROR] Python 3 not found"; exit 1; }

RESULTS_CSV="./appraisal_results.csv"
CKPT_BASE="./checkpoints_appraisal"
DATA_BASE="./data/appraisal"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; BOLD='\033[1m'; RED='\033[0;31m'; NC='\033[0m'
log()    { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
warn()   { echo -e "\033[1;33m[WARN]\033[0m $*"; }
header() {
  echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}\n"
}

# ── Prep one appraisal source ─────────────────────────────────────────────────
ensure_appraisal_csv() {
  local source="$1"
  local out_dir="$2"
  if [[ -f "$out_dir/train.csv" && -f "$out_dir/val.csv" && -f "$out_dir/test.csv" ]]; then
    ok "Data already present: $out_dir"
    return 0
  fi
  log "Preprocessing appraisal source='$source' ..."
  $PYTHON prep_data.py --dataset appraisal --source "$source" --out_dir "$out_dir"
}

# ── Train one configuration ───────────────────────────────────────────────────
run_appraisal_dataset() {
  local key="$1"
  local name="$2"
  local data_dir="$3"
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
    --local_data      "$data_dir"  \
    --utterance_col   "Utterance"  \
    --speaker_col     "Speaker"    \
    --emotion_col     "Emotion"    \
    --dialogue_id_col "Dialogue_ID" \
    2>&1 | tee "$log_file" || exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo -e "${RED}[FAILED]${NC} $name training failed (exit $exit_code)"
    return 1
  fi
  ok "$name training done"
}

# ── Source display names ──────────────────────────────────────────────────────
declare -A SOURCE_DISPLAY=(
  ["meld"]="Appraisal-MELD"
  ["emorynlp"]="Appraisal-EmoryNLP"
  ["dailydialog"]="Appraisal-DailyDialog"
  ["all"]="Appraisal-All (combined)"
)

# ── Run each source ───────────────────────────────────────────────────────────
for source in "${SOURCES[@]}"; do
  display="${SOURCE_DISPLAY[$source]:-Appraisal-$source}"
  data_dir="$DATA_BASE/$source"

  ensure_appraisal_csv "$source" "$data_dir"
  run_appraisal_dataset "appraisal_$source" "$display" "$data_dir"
done

# ── Compile results CSV ───────────────────────────────────────────────────────
header "Compiling Results → $RESULTS_CSV"

$PYTHON - "$CKPT_BASE" "$RESULTS_CSV" <<'PYEOF'
import os, json, csv, sys

ckpt_base   = sys.argv[1]
results_csv = sys.argv[2]

SOURCE_NAMES = {
    "appraisal_meld":        "Appraisal-MELD",
    "appraisal_emorynlp":    "Appraisal-EmoryNLP",
    "appraisal_dailydialog": "Appraisal-DailyDialog",
    "appraisal_all":         "Appraisal-All",
}

summary_rows = []
emotion_rows = []
all_bucket_keys = []

for key, display in SOURCE_NAMES.items():
    metrics_path = os.path.join(ckpt_base, key, "test_metrics.json")
    if not os.path.isfile(metrics_path):
        print(f"  [SKIP] No test_metrics.json for {display}")
        continue

    with open(metrics_path) as f:
        m = json.load(f)

    bucket_f1 = m.get("bucket_f1", {})
    for k in bucket_f1:
        if f"bucket_f1_{k}" not in all_bucket_keys:
            all_bucket_keys.append(f"bucket_f1_{k}")

    summary_rows.append({
        "dataset":          display,
        "accuracy":         round(m.get("accuracy",        0), 4),
        "weighted_f1":      round(m.get("weighted_f1",     0), 4),
        "recognition_f1":   round(m.get("recognition_f1",  0) or 0, 4),
        "posterior_f1":     round(m.get("posterior_f1",    0) or 0, 4),
        "mean_surprise":    round(m.get("mean_surprise",   0), 4),
        **{f"bucket_f1_{k}": round(v, 4) for k, v in bucket_f1.items()},
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

# Fill missing bucket keys with 0 so all rows have the same columns
base_cols = ["dataset","accuracy","weighted_f1","recognition_f1","posterior_f1","mean_surprise"]
all_cols   = base_cols + sorted(all_bucket_keys)
for r in summary_rows:
    for col in all_cols:
        r.setdefault(col, 0)

# ── Summary table ──────────────────────────────────────────────────────────
print("\n  Overall Metrics:")
if summary_rows:
    cols = all_cols
    w = max(len(c) for c in cols)
    print("  " + "  ".join(f"{c:<{max(w,12)}}" for c in cols))
    print("  " + "  ".join("-" * max(w,12) for c in cols))
    for r in summary_rows:
        print("  " + "  ".join(f"{str(r[c]):<{max(w,12)}}" for c in cols))

# ── Per-emotion table ──────────────────────────────────────────────────────
print("\n  Per-Emotion Metrics:")
if emotion_rows:
    print(f"  {'Dataset':<25} {'Emotion':<14} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Support':>9}")
    print(f"  {'-'*25} {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
    for r in emotion_rows:
        print(f"  {r['dataset']:<25} {r['emotion']:<14} "
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
