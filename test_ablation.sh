#!/usr/bin/env bash
# =============================================================================
# test_ablation.sh — Ablation study for new architectural ideas on IEMOCAP.
#
# Tests four changes against a no_staged base:
#   base             — full model, no staged training  (reference)
#   no_recog         — Change 1: recognition_loss_weight=0 + no staged
#   cross_speaker    — Change 2: other-speaker emotion in speaker context GRU
#   future_pred      — Change 3: auxiliary s(t)→emotion[t+1] head
#   joint_transition — Change 4: joint BCE transition head on s(t)
#   all_new          — Changes 1+2+3+4 combined
#
# Results → test_ablation_results.csv
# Logs    → test_ablation_checkpoints/<name>/train.log
#
# Usage:
#   bash test_ablation.sh [--device cuda] [--batch_size 4] [--llama_model MODEL]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEVICE="cuda"
BATCH_SIZE=4
LLAMA_MODEL="meta-llama/Llama-3.1-8B"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)      DEVICE="$2";      shift 2 ;;
    --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
    --llama_model) LLAMA_MODEL="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

OUTPUT_BASE="./test_ablation_checkpoints"
RESULTS_CSV="./test_ablation_results.csv"
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
  echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}\n"
}

# ── Ensure IEMOCAP data ───────────────────────────────────────────────────────
if [[ -f "$IEMOCAP_DIR/train.csv" && -f "$IEMOCAP_DIR/val.csv" && -f "$IEMOCAP_DIR/test.csv" ]]; then
  ok "IEMOCAP data present: $IEMOCAP_DIR"
else
  log "Preprocessing IEMOCAP ..."
  $PYTHON prep_data.py --dataset iemocap --out_dir "$IEMOCAP_DIR"
fi

# ── Fixed args passed to every run ───────────────────────────────────────────
DATA_ARGS=(
  --local_data      "$IEMOCAP_DIR"
  --utterance_col   "Utterance"
  --speaker_col     "Speaker"
  --emotion_col     "Emotion"
  --dialogue_id_col "Dialogue_ID"
  --llama_model     "$LLAMA_MODEL"
  --device          "$DEVICE"
  --batch_size      "$BATCH_SIZE"
)

# ── Parse [TEST] metrics from a training log ──────────────────────────────────
parse_test_metrics() {
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

# ── Run one ablation ──────────────────────────────────────────────────────────
CURRENT_IDX=0
run_ablation() {
  local name="$1"
  local desc="$2"
  shift 2

  CURRENT_IDX=$((CURRENT_IDX + 1))
  local out_dir="$OUTPUT_BASE/$name"
  local log_file="$out_dir/train.log"

  header "[$CURRENT_IDX/${#ABLATIONS[@]}] $name"
  log "Description : $desc"
  mkdir -p "$out_dir"

  local exit_code=0
  $PYTHON test_ablation_runner.py \
    --name        "$name"     \
    --output_dir  "$out_dir"  \
    "${DATA_ARGS[@]}"         \
    "$@" 2>&1 | tee "$log_file" || exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo -e "${RED}[FAILED]${NC} $name exited with code $exit_code"
    printf '"%s","%s",FAILED,FAILED,FAILED\n' "$name" "$desc" >> "$RESULTS_CSV"
    return 0
  fi

  local prior_f1 recog_f1 gap
  read -r prior_f1 recog_f1 gap <<< "$(parse_test_metrics "$log_file")"
  printf '"%s","%s",%s,%s,%s\n' "$name" "$desc" "$prior_f1" "$recog_f1" "$gap" >> "$RESULTS_CSV"
  ok "$name  →  Prior F1: $prior_f1  |  Recog F1: $recog_f1  |  Gap: $gap"
}

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION TABLE
# ══════════════════════════════════════════════════════════════════════════════
ABLATIONS=(
  "base"
  "no_recog"
  "cross_speaker"
  "future_pred"
  "joint_transition"
  "all_new"
)

header "New Ideas Ablation — IEMOCAP (8B)"
log "Device     : $DEVICE"
log "Batch size : $BATCH_SIZE"
log "LLaMA      : $LLAMA_MODEL"
log "Output     : $OUTPUT_BASE"
log "Results    : $RESULTS_CSV"

mkdir -p "$OUTPUT_BASE"
echo "name,description,test_prior_f1,test_recognition_f1,recog_prior_gap" > "$RESULTS_CSV"

run_ablation "base" \
  "Full model, no staged training — reference point" \
  --no_staged

run_ablation "no_recog" \
  "Change 1: recognition_loss_weight=0 + no staged" \
  --no_staged \
  --recognition_loss_weight 0.0

run_ablation "cross_speaker" \
  "Change 2: other-speaker emotion injected into speaker context GRU" \
  --no_staged \
  --recognition_loss_weight 0.0 \
  --use_cross_speaker_emo

run_ablation "future_pred" \
  "Change 3: auxiliary s(t)->emotion[t+1] prediction head (w=0.1)" \
  --no_staged \
  --recognition_loss_weight 0.0 \
  --use_future_pred \
  --future_pred_weight 0.1

run_ablation "joint_transition" \
  "Change 4: joint BCE transition head on s(t) (w=0.1)" \
  --no_staged \
  --recognition_loss_weight 0.0 \
  --use_joint_transition \
  --joint_transition_weight 0.1

run_ablation "all_new" \
  "All four changes combined" \
  --no_staged \
  --recognition_loss_weight 0.0 \
  --use_cross_speaker_emo \
  --use_future_pred       --future_pred_weight 0.1 \
  --use_joint_transition  --joint_transition_weight 0.1

# ── Summary ───────────────────────────────────────────────────────────────────
header "Results Summary"

$PYTHON - "$RESULTS_CSV" <<'PYEOF'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("  No results yet."); sys.exit(0)

nw  = max(len(r['name']) for r in rows)
ref = next((r for r in rows if r['name'] == 'base'), None)

def floatable(v):
    try: float(v); return True
    except: return False

print(f"  {'Name':<{nw}}  {'Prior F1':>10}  {'Recog F1':>10}  {'Gap':>8}  {'vs base':>9}")
print(f"  {'-'*nw}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*9}")
for r in rows:
    vs = ''
    if ref and r['name'] != 'base' and floatable(r['test_prior_f1']) and floatable(ref['test_prior_f1']):
        vs = f"{float(r['test_prior_f1']) - float(ref['test_prior_f1']):+.4f}"
    print(f"  {r['name']:<{nw}}"
          f"  {r['test_prior_f1']:>10}"
          f"  {r['test_recognition_f1']:>10}"
          f"  {r['recog_prior_gap']:>8}"
          f"  {vs:>9}")
PYEOF

echo ""
ok "Done. Results saved to $RESULTS_CSV"
