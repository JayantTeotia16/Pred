#!/usr/bin/env bash
# =============================================================================
# run.sh — Universal Dispositional Emotion Prediction pipeline
#
# Works with ANY conversational emotion dataset:
#   - HuggingFace datasets (auto-downloaded)
#   - Local CSV files
#   - Named or anonymous speakers
#   - Any number of emotion classes
#
# Usage:
#   bash run.sh [MODE] [OPTIONS]
#
# Modes:
#   sanity    Shape/loss check, no data or LLaMA needed (default)
#   train     Full training run
#   eval      Evaluate a checkpoint on the test set
#   analyse   Landscape + history F1 + temporal drift plots
#   all       train → eval → analyse
#
# ── Dataset options (examples) ─────────────────────────────────────────────
#
#   Default (MELD text-only via HuggingFace):
#     bash run.sh train
#
#   Any other HuggingFace dataset:
#     bash run.sh train \
#       --hf_dataset eusip/silicone --hf_config dyda_e \
#       --utterance_col Utterance --emotion_col Emotion \
#       --dialogue_id_col Dialogue_ID --speaker_col "" \
#       --train_split train --val_split validation --test_split test
#
#   Local CSV files:
#     bash run.sh train \
#       --local_data ./my_data \       # dir with train.csv / val.csv / test.csv
#       --utterance_col text \
#       --emotion_col label \
#       --dialogue_id_col conv_id \
#       --speaker_col speaker_name     # or "" if no speaker column
#
# ── All options ────────────────────────────────────────────────────────────
#   --llama_model     HuggingFace model name   (default: meta-llama/Llama-3.2-1B)
#   --hf_dataset      HF dataset name          (default: eusip/silicone)
#   --hf_config       HF dataset config        (default: meld_e)
#   --local_data      Local CSV directory      (overrides HF if set)
#   --utterance_col   Column with text         (default: Utterance)
#   --speaker_col     Column with speaker      (default: Speaker; "" = none)
#   --emotion_col     Column with emotion int  (default: Emotion)
#   --dialogue_id_col Column with dialogue id  (default: Dialogue_ID)
#   --train_split     HF train split name      (default: train)
#   --val_split       HF val split name        (default: validation)
#   --test_split      HF test split name       (default: test)
#   --checkpoint      Path to .pt file         (required for eval/analyse)
#   --batch_size      N                        (default: 4)
#   --epochs          N                        (default: 25)
#   --device          cuda or cpu              (default: cuda)
#   --no_scene        Disable scene dynamics
#   --output_dir      Checkpoint save dir      (default: ./checkpoints)
#   --analysis_dir    Analysis output dir      (default: ./analysis_outputs)
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()    { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()    { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
header() {
  echo -e "\n${BOLD}${BLUE}══════════════════════════════════════${NC}"
  echo -e "${BOLD}${BLUE}  $*${NC}"
  echo -e "${BOLD}${BLUE}══════════════════════════════════════${NC}\n"
}

# ── Defaults ─────────────────────────────────────────────────────────────────
MODE="${1:-sanity}"
shift || true

LLAMA_MODEL="meta-llama/Llama-3.2-1B"
HF_DATASET="eusip/silicone"
HF_CONFIG="meld_e"
LOCAL_DATA=""
UTTERANCE_COL="Utterance"
SPEAKER_COL="Speaker"
EMOTION_COL="Emotion"
DIALOGUE_ID_COL="Dialogue_ID"
TRAIN_SPLIT="train"
VAL_SPLIT="validation"
TEST_SPLIT="test"
CHECKPOINT=""
BATCH_SIZE=4
EPOCHS=25
DEVICE="cuda"
NO_SCENE=""
OUTPUT_DIR="./checkpoints"
ANALYSIS_DIR="./analysis_outputs"

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama_model)     LLAMA_MODEL="$2";     shift 2 ;;
    --hf_dataset)      HF_DATASET="$2";      shift 2 ;;
    --hf_config)       HF_CONFIG="$2";       shift 2 ;;
    --local_data)      LOCAL_DATA="$2";      shift 2 ;;
    --utterance_col)   UTTERANCE_COL="$2";   shift 2 ;;
    --speaker_col)     SPEAKER_COL="$2";     shift 2 ;;
    --emotion_col)     EMOTION_COL="$2";     shift 2 ;;
    --dialogue_id_col) DIALOGUE_ID_COL="$2"; shift 2 ;;
    --train_split)     TRAIN_SPLIT="$2";     shift 2 ;;
    --val_split)       VAL_SPLIT="$2";       shift 2 ;;
    --test_split)      TEST_SPLIT="$2";      shift 2 ;;
    --checkpoint)      CHECKPOINT="$2";      shift 2 ;;
    --batch_size)      BATCH_SIZE="$2";      shift 2 ;;
    --epochs)          EPOCHS="$2";          shift 2 ;;
    --device)          DEVICE="$2";          shift 2 ;;
    --no_scene)        NO_SCENE="--no_scene"; shift ;;
    --output_dir)      OUTPUT_DIR="$2";      shift 2 ;;
    --analysis_dir)    ANALYSIS_DIR="$2";    shift 2 ;;
    *) err "Unknown option: $1" ;;
  esac
done

# ── CD to script directory ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Python detection ──────────────────────────────────────────────────────────
PYTHON=""
for candidate in python3 python; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c "import sys; print(sys.version_info >= (3,9))")
    if [[ "$ver" == "True" ]]; then PYTHON="$candidate"; break; fi
  fi
done
[[ -z "$PYTHON" ]] && err "Python 3.9+ not found."
ok "Python: $($PYTHON --version)"

# ── GPU check ─────────────────────────────────────────────────────────────────
if [[ "$DEVICE" == "cuda" ]]; then
  has_gpu=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
  if [[ "$has_gpu" != "True" ]]; then
    warn "No CUDA GPU found — falling back to CPU (training will be slow)."
    DEVICE="cpu"
  fi
fi
log "Device: $DEVICE"

# ── Build python args string ──────────────────────────────────────────────────
build_py_args() {
  local args="--llama_model $LLAMA_MODEL"
  args+=" --utterance_col $UTTERANCE_COL"
  args+=" --emotion_col $EMOTION_COL"
  args+=" --dialogue_id_col $DIALOGUE_ID_COL"
  args+=" --train_split $TRAIN_SPLIT"
  args+=" --val_split $VAL_SPLIT"
  args+=" --test_split $TEST_SPLIT"
  args+=" --batch_size $BATCH_SIZE"
  args+=" --epochs $EPOCHS"
  args+=" --device $DEVICE"
  args+=" --output_dir $OUTPUT_DIR"
  args+=" --analysis_dir $ANALYSIS_DIR"
  [[ -n "$SPEAKER_COL" ]]  && args+=" --speaker_col $SPEAKER_COL"
  [[ -n "$NO_SCENE" ]]     && args+=" $NO_SCENE"
  [[ -n "$CHECKPOINT" ]]   && args+=" --checkpoint $CHECKPOINT"
  if [[ -n "$LOCAL_DATA" ]]; then
    args+=" --local_data $LOCAL_DATA"
  else
    args+=" --hf_dataset $HF_DATASET --hf_config $HF_CONFIG"
  fi
  echo "$args"
}

# ═════════════════════════════════════════════════════════════════════════════
install_deps() {
  header "Installing dependencies"
  $PYTHON -m pip install --quiet --upgrade pip
  $PYTHON -m pip install --quiet -r requirements.txt
  ok "Dependencies installed."
}

# ═════════════════════════════════════════════════════════════════════════════
check_llama() {
  [[ "$MODE" == "sanity" ]] && return
  header "Checking LLaMA access ($LLAMA_MODEL)"
  $PYTHON - <<PYEOF
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained("$LLAMA_MODEL")
    print("  LLaMA config accessible.")
except Exception as e:
    print(f"  WARNING: {e}")
    print("  Run: huggingface-cli login")
    import sys; sys.exit(1)
PYEOF
  ok "LLaMA accessible."
}

# ═════════════════════════════════════════════════════════════════════════════
check_dataset() {
  [[ "$MODE" == "sanity" ]] && return
  header "Dataset check"
  if [[ -n "$LOCAL_DATA" ]]; then
    for f in train.csv val.csv test.csv; do
      [[ -f "$LOCAL_DATA/$f" ]] && ok "Found $LOCAL_DATA/$f" || err "Missing $LOCAL_DATA/$f"
    done
  else
    log "HuggingFace dataset: $HF_DATASET / $HF_CONFIG"
    log "(Will be auto-downloaded on first run)"
    $PYTHON -c "
from datasets import load_dataset
try:
    ds = load_dataset('$HF_DATASET', '$HF_CONFIG', split='$TRAIN_SPLIT', trust_remote_code=True)
    print(f'  Train split: {len(ds)} examples')
    print(f'  Columns: {list(ds.features.keys())}')
except Exception as e:
    print(f'  ERROR: {e}')
    import sys; sys.exit(1)
"
    ok "Dataset accessible."
  fi
}

# ═════════════════════════════════════════════════════════════════════════════
run_sanity() {
  header "Sanity Check"
  $PYTHON main.py --mode sanity
  ok "Sanity check passed."
}

run_train() {
  header "Training"
  mkdir -p "$OUTPUT_DIR"
  log "LLaMA model   : $LLAMA_MODEL"
  log "Dataset       : ${LOCAL_DATA:-$HF_DATASET/$HF_CONFIG}"
  log "Batch / Epochs: $BATCH_SIZE / $EPOCHS"
  log "Device        : $DEVICE"
  [[ -n "$NO_SCENE" ]] && warn "Scene dynamics DISABLED"

  PY_ARGS="$(build_py_args)"
  $PYTHON main.py --mode train $PY_ARGS

  [[ -f "$OUTPUT_DIR/best_model.pt" ]] && CHECKPOINT="$OUTPUT_DIR/best_model.pt"
  ok "Training complete. Best checkpoint: ${CHECKPOINT:-not found}"
}

run_eval() {
  header "Evaluation"
  [[ -n "$CHECKPOINT" && ! -f "$CHECKPOINT" ]] && err "Checkpoint not found: $CHECKPOINT"
  PY_ARGS="$(build_py_args)"
  $PYTHON main.py --mode eval $PY_ARGS
  ok "Evaluation complete."
}

run_analyse() {
  header "Analysis"
  [[ -n "$CHECKPOINT" && ! -f "$CHECKPOINT" ]] && err "Checkpoint not found: $CHECKPOINT"
  mkdir -p "$ANALYSIS_DIR"
  PY_ARGS="$(build_py_args)"
  $PYTHON main.py --mode analyse $PY_ARGS
  ok "Analysis outputs → $ANALYSIS_DIR"
  echo ""
  for f in "$ANALYSIS_DIR"/*.png "$ANALYSIS_DIR"/*.json; do
    [[ -f "$f" ]] && echo "    $f"
  done
}

# ═════════════════════════════════════════════════════════════════════════════
echo -e "\n${BOLD}Dispositional Emotion Prediction — Universal Model${NC}"
echo -e "Mode: ${CYAN}${MODE}${NC}\n"

install_deps

case "$MODE" in
  sanity)  run_sanity ;;
  train)   check_dataset; check_llama; run_train ;;
  eval)    check_dataset; check_llama; run_eval ;;
  analyse) check_dataset; check_llama; run_analyse ;;
  all)     check_dataset; check_llama; run_train; run_eval; run_analyse ;;
  *)       err "Unknown mode: $MODE. Choose: sanity|train|eval|analyse|all" ;;
esac

echo ""; ok "Done — mode=$MODE"
