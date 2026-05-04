#!/usr/bin/env bash
# =============================================================================
# run_all_datasets.sh - Train and evaluate on MELD, DailyDialog, IEMOCAP,
#                       EmoryNLP (MultiDialog already done separately)
#
# Uses the best known config (all_new):
#   no staged training, recognition_loss_weight=0,
#   cross-speaker emotion context, future prediction head (w=0.1),
#   joint transition head (w=0.1)
#
# Usage:
#   bash run_all_datasets.sh [--device cuda] [--batch_size 8] [--llama_model MODEL]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
DEVICE="cuda"
BATCH_SIZE=8
LLAMA_MODEL="meta-llama/Llama-3.1-8B"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)      DEVICE="$2";      shift 2 ;;
    --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
    --llama_model) LLAMA_MODEL="$2"; shift 2 ;;
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
  echo -e "\n${BOLD}${CYAN}==========================================${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}==========================================${NC}\n"
}

# Run one dataset
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
  $PYTHON test_ablation_runner.py \
    --name        "$key"          \
    --output_dir  "$out_dir"      \
    --batch_size  "$BATCH_SIZE"   \
    --device      "$DEVICE"       \
    --llama_model "$LLAMA_MODEL"  \
    --no_staged                   \
    --recognition_loss_weight 0.0 \
    --use_cross_speaker_emo       \
    --use_future_pred             \
    --future_pred_weight 0.1      \
    --use_joint_transition        \
    --joint_transition_weight 0.1 \
    "$@" 2>&1 | tee "$log_file" || exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo -e "${RED}[FAILED]${NC} $name training failed (exit $exit_code)"
    return 1
  fi
  ok "$name training done"
}

# Check / prep dataset
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

# -----------------------------------------------------------------------------
# 1. EmoryNLP
# -----------------------------------------------------------------------------
EMORYNLP_DIR="./data/emorynlp"
ensure_csv "emorynlp" "$EMORYNLP_DIR"

run_dataset "emorynlp" "EmoryNLP" \
  --local_data      "$EMORYNLP_DIR" \
  --utterance_col   "Utterance"     \
  --speaker_col     "Speaker"       \
  --emotion_col     "Emotion"       \
  --dialogue_id_col "Dialogue_ID"

# -----------------------------------------------------------------------------
# 2. IEMOCAP
# -----------------------------------------------------------------------------
IEMOCAP_DIR="./data/iemocap"
ensure_csv "iemocap" "$IEMOCAP_DIR"

run_dataset "iemocap" "IEMOCAP" \
  --local_data      "$IEMOCAP_DIR" \
  --utterance_col   "Utterance"    \
  --speaker_col     "Speaker"      \
  --emotion_col     "Emotion"      \
  --dialogue_id_col "Dialogue_ID"

# -----------------------------------------------------------------------------
# 3. DailyDialog
# -----------------------------------------------------------------------------
DAILYDIALOG_DIR="./data/dailydialog"
ensure_csv "dailydialog" "$DAILYDIALOG_DIR"

run_dataset "dailydialog" "DailyDialog" \
  --local_data      "$DAILYDIALOG_DIR" \
  --utterance_col   "Utterance"        \
  --speaker_col     "Speaker"          \
  --emotion_col     "Emotion"          \
  --dialogue_id_col "Dialogue_ID"

# -----------------------------------------------------------------------------
# 4. MELD  (loads from HuggingFace)
# -----------------------------------------------------------------------------
run_dataset "meld" "MELD"

# -----------------------------------------------------------------------------
# Compile results CSV
# -----------------------------------------------------------------------------
header "Compiling Results"

$PYTHON - "$CKPT_BASE" "$RESULTS_CSV" <<'PYEOF'
import os, json, csv, sys

ckpt_base   = sys.argv[1]
results_csv = sys.argv[2]

DATASET_NAMES = {
    "multidialog": "MultiDialog",
    "emorynlp":    "EmoryNLP",
    "iemocap":     "IEMOCAP",
    "dailydialog": "DailyDialog",
    "meld":        "MELD",
}

summary_rows = []
emotion_rows = []

for key, display in DATASET_NAMES.items():
    metrics_path = os.path.join(ckpt_base, key, "test_metrics.json")
    if not os.path.isfile(metrics_path):
        print("  [SKIP] No test_metrics.json for " + display)
        continue

    with open(metrics_path) as f:
        m = json.load(f)

    row = {
        "dataset":        display,
        "accuracy":       round(m.get("accuracy",       0), 4),
        "weighted_f1":    round(m.get("weighted_f1",    0), 4),
        "recognition_f1": round(m.get("recognition_f1", 0) or 0, 4),
        "mean_surprise":  round(m.get("mean_surprise",  0), 4),
    }
    for k, v in m.get("bucket_f1", {}).items():
        row["bucket_f1_" + k] = round(v, 4)
    summary_rows.append(row)

    for emo, stats in m.get("per_emotion", {}).items():
        emotion_rows.append({
            "dataset":   display,
            "emotion":   emo,
            "precision": round(stats.get("precision", 0), 4),
            "recall":    round(stats.get("recall",    0), 4),
            "f1":        round(stats.get("f1-score",  0), 4),
            "support":   int(stats.get("support",     0)),
        })

print("\n  Overall Metrics:")
if summary_rows:
    cols = list(summary_rows[0].keys())
    w = max(len(c) for c in cols)
    w = max(w, 10)
    print("  " + "  ".join(c.ljust(w) for c in cols))
    print("  " + "  ".join("-" * w for c in cols))
    for r in summary_rows:
        print("  " + "  ".join(str(r[c]).ljust(w) for c in cols))

print("\n  Per-Emotion Metrics:")
if emotion_rows:
    print("  {:<14} {:<20} {:>7} {:>7} {:>7} {:>9}".format(
        "Dataset", "Emotion", "F1", "Prec", "Rec", "Support"))
    print("  " + "-" * 70)
    for r in emotion_rows:
        print("  {:<14} {:<20} {:>7.4f} {:>7.4f} {:>7.4f} {:>9}".format(
            r["dataset"], r["emotion"],
            r["f1"], r["precision"], r["recall"], r["support"]))

if summary_rows:
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("\n  Summary CSV -> " + results_csv)

emotion_csv = results_csv.replace(".csv", "_per_emotion.csv")
if emotion_rows:
    with open(emotion_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset","emotion","precision","recall","f1","support"])
        writer.writeheader()
        writer.writerows(emotion_rows)
    print("  Per-emotion CSV -> " + emotion_csv)
PYEOF

echo ""
ok "All done. Results in $RESULTS_CSV"
