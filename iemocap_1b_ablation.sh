#!/usr/bin/env bash
# =============================================================================
# iemocap_1b_ablation.sh - Ablation study on IEMOCAP with LLaMA 1B.
#
# Base config (full): all active modules from run_all_datasets
#   no staged, recognition_loss_weight=0,
#   CrossSpeakerEmotionContext, FuturePredHead (w=0.1), TransitionHead (w=0.1),
#   EmotionLabelContext GRU, emotion label embedding, LoRA
#
# Each config removes exactly one active module from the full config.
#
# Configs:
#   full              - complete all_new config (reference)
#   no_cross_speaker  - standard speaker context (no other-speaker emotion)
#   no_future_pred    - remove future prediction head
#   no_joint_trans    - remove joint transition head
#   no_label_gru      - disable EmotionLabelContext GRU (dim=1)
#   no_emotion_embed  - disable emotion label embedding (dim=1)
#   no_label_modules  - disable both label GRU and embedding
#   no_lora           - frozen LLaMA throughout
#
# Results -> iemocap_1b_ablation_results.csv
# Logs    -> iemocap_1b_ablation_checkpoints/<name>/train.log
#
# Usage:
#   bash iemocap_1b_ablation.sh [--device cuda] [--batch_size 4]
# =============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="cuda"
BATCH_SIZE=4
LLAMA_MODEL="meta-llama/Llama-3.2-1B"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)      DEVICE="$2";      shift 2 ;;
    --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
    --llama_model) LLAMA_MODEL="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

OUTPUT_BASE="./iemocap_1b_ablation_checkpoints"
RESULTS_CSV="./iemocap_1b_ablation_results.csv"
IEMOCAP_DIR="./data/iemocap"

PYTHON=""
for c in python3 python; do
  if command -v "$c" &>/dev/null; then PYTHON="$c"; break; fi
done
[[ -z "$PYTHON" ]] && { echo "[ERROR] Python 3 not found"; exit 1; }

CYAN='\033[0;36m'; GREEN='\033[0;32m'; BOLD='\033[1m'; RED='\033[0;31m'; NC='\033[0m'
log()    { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
header() {
  echo -e "\n${BOLD}${CYAN}==========================================${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}==========================================${NC}\n"
}

if [[ -f "$IEMOCAP_DIR/train.csv" && -f "$IEMOCAP_DIR/val.csv" && -f "$IEMOCAP_DIR/test.csv" ]]; then
  ok "IEMOCAP data present"
else
  log "Preprocessing IEMOCAP..."
  $PYTHON prep_data.py --dataset iemocap --out_dir "$IEMOCAP_DIR"
fi

# Fixed args for every run
BASE_ARGS=(
  --local_data      "$IEMOCAP_DIR"
  --utterance_col   "Utterance"
  --speaker_col     "Speaker"
  --emotion_col     "Emotion"
  --dialogue_id_col "Dialogue_ID"
  --llama_model     "$LLAMA_MODEL"
  --device          "$DEVICE"
  --batch_size      "$BATCH_SIZE"
  --no_staged
  --recognition_loss_weight 0.0
)

# Parse [TEST] prior F1 and recognition F1 from log
parse_metrics() {
  $PYTHON - "$1" <<'PYEOF'
import re, sys
with open(sys.argv[1]) as f:
    txt = f.read()
parts = txt.split('[TEST]')
if len(parts) < 2:
    print('N/A N/A N/A'); sys.exit(0)
block = parts[-1]
prior = re.search(r'Prior F1\s*:\s*([\d.]+)',       block)
recog = re.search(r'Recognition F1\s*:\s*([\d.]+)', block)
gap   = re.search(r'gap vs prior ([+-][\d.]+)',       block)
print(
    prior.group(1) if prior else 'N/A',
    recog.group(1) if recog else 'N/A',
    gap.group(1)   if gap   else 'N/A',
)
PYEOF
}

CURRENT_IDX=0
NUM_CONFIGS=8

run_ablation() {
  local name="$1"
  local desc="$2"
  shift 2

  CURRENT_IDX=$((CURRENT_IDX + 1))
  local out_dir="$OUTPUT_BASE/$name"
  local log_file="$out_dir/train.log"

  header "[$CURRENT_IDX/$NUM_CONFIGS] $name"
  log "Description: $desc"
  mkdir -p "$out_dir"

  local exit_code=0
  $PYTHON test_ablation_runner.py \
    --name       "$name"    \
    --output_dir "$out_dir" \
    "${BASE_ARGS[@]}"       \
    "$@" 2>&1 | tee "$log_file" || exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo -e "${RED}[FAILED]${NC} $name (exit $exit_code)"
    printf '"%s","%s",FAILED,FAILED,FAILED\n' "$name" "$desc" >> "$RESULTS_CSV"
    return 0
  fi

  local prior recog gap
  read -r prior recog gap <<< "$(parse_metrics "$log_file")"
  printf '"%s","%s",%s,%s,%s\n' "$name" "$desc" "$prior" "$recog" "$gap" >> "$RESULTS_CSV"
  ok "$name  ->  Prior F1: $prior  |  Gap: $gap"
}

# =============================================================================
header "IEMOCAP 1B Ablation - All Active Modules"
log "Model      : $LLAMA_MODEL"
log "Device     : $DEVICE"
log "Output     : $OUTPUT_BASE"

mkdir -p "$OUTPUT_BASE"
echo "name,description,prior_f1,recognition_f1,gap" > "$RESULTS_CSV"

run_ablation "full" \
  "All active modules: cross-speaker, future-pred, joint-trans, label-GRU, emo-embed, LoRA" \
  --use_cross_speaker_emo \
  --use_future_pred       --future_pred_weight 0.1 \
  --use_joint_transition  --joint_transition_weight 0.1

run_ablation "no_cross_speaker" \
  "Remove CrossSpeakerEmotionContext (standard speaker context GRU)" \
  --use_future_pred       --future_pred_weight 0.1 \
  --use_joint_transition  --joint_transition_weight 0.1

run_ablation "no_future_pred" \
  "Remove future prediction head" \
  --use_cross_speaker_emo \
  --use_joint_transition  --joint_transition_weight 0.1

run_ablation "no_joint_trans" \
  "Remove joint transition BCE head" \
  --use_cross_speaker_emo \
  --use_future_pred       --future_pred_weight 0.1

run_ablation "no_label_gru" \
  "Disable EmotionLabelContext GRU (label_context_dim=1)" \
  --use_cross_speaker_emo \
  --use_future_pred       --future_pred_weight 0.1 \
  --use_joint_transition  --joint_transition_weight 0.1 \
  --label_context_dim 1

run_ablation "no_emotion_embed" \
  "Disable emotion label embedding (emotion_label_embed_dim=1)" \
  --use_cross_speaker_emo \
  --use_future_pred       --future_pred_weight 0.1 \
  --use_joint_transition  --joint_transition_weight 0.1 \
  --emotion_label_embed_dim 1

run_ablation "no_label_modules" \
  "Disable both label GRU and emotion embedding" \
  --use_cross_speaker_emo \
  --use_future_pred       --future_pred_weight 0.1 \
  --use_joint_transition  --joint_transition_weight 0.1 \
  --label_context_dim 1   --emotion_label_embed_dim 1

run_ablation "no_lora" \
  "Frozen LLaMA throughout (no LoRA adaptation)" \
  --use_cross_speaker_emo \
  --use_future_pred       --future_pred_weight 0.1 \
  --use_joint_transition  --joint_transition_weight 0.1 \
  --no_lora

# =============================================================================
header "Results Summary"

$PYTHON - "$RESULTS_CSV" <<'PYEOF'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("  No results."); sys.exit(0)

nw  = max(len(r['name']) for r in rows)
ref = next((r for r in rows if r['name'] == 'full'), None)

def floatable(v):
    try: float(v); return True
    except: return False

print("  {:<{w}}  {:>10}  {:>10}  {:>8}  {:>9}".format(
    "Name", "Prior F1", "Recog F1", "Gap", "vs full", w=nw))
print("  {:<{w}}  {:>10}  {:>10}  {:>8}  {:>9}".format(
    "-"*nw, "-"*10, "-"*10, "-"*8, "-"*9, w=nw))
for r in rows:
    vs = ''
    if ref and r['name'] != 'full' and floatable(r['prior_f1']) and floatable(ref['prior_f1']):
        vs = "{:+.4f}".format(float(r['prior_f1']) - float(ref['prior_f1']))
    print("  {:<{w}}  {:>10}  {:>10}  {:>8}  {:>9}".format(
        r['name'], r['prior_f1'], r['recognition_f1'], r['gap'], vs, w=nw))
PYEOF

echo ""
ok "Done. Results -> $RESULTS_CSV"
