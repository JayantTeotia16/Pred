#!/usr/bin/env bash
# =============================================================================
# ablation.sh — Component ablation study on IEMOCAP
#
# IEMOCAP is the primary ablation dataset: strong dispositional signal
# (prior beats recognition by +0.22), 6 emotion classes, balanced speakers.
#
# Configurations:
#   full_model       — staged training, all components
#   baseline         — LLaMA-LoRA classifier, no dispositional modules
#   no_staged        — same as full_model but non-staged (flat LR)
#   no_lora          — frozen LLaMA throughout (no LoRA adaptation)
#   no_label_gru     — ablate EmotionLabelContext GRU (dim → 1)
#   no_emotion_embed — ablate past-emotion label embedding (dim → 1)
#   no_label_modules — ablate all emotion-label conditioning (both → 1)
#
# Results → ablation_results.csv
# Logs    → ablation_checkpoints/<name>/train.log
#
# Usage:
#   bash ablation.sh [--device cuda|cpu] [--batch_size N] [--llama_model MODEL]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse flags ───────────────────────────────────────────────────────────────
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

OUTPUT_BASE="./ablation_checkpoints"
RESULTS_CSV="./ablation_results.csv"
IEMOCAP_DIR="./data/iemocap"

PYTHON=""
for c in python3 python; do
  if command -v "$c" &>/dev/null; then PYTHON="$c"; break; fi
done
[[ -z "$PYTHON" ]] && { echo "[ERROR] Python 3 not found"; exit 1; }

# ── Colours ───────────────────────────────────────────────────────────────────
CYAN='\033[0;36m'; GREEN='\033[0;32m'; BOLD='\033[1m'; RED='\033[0;31m'; NC='\033[0m'
log()    { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
header() {
  echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}\n"
}

# ── Ensure IEMOCAP CSVs exist ─────────────────────────────────────────────────
if [[ -f "$IEMOCAP_DIR/train.csv" && -f "$IEMOCAP_DIR/val.csv" && -f "$IEMOCAP_DIR/test.csv" ]]; then
  ok "IEMOCAP data already present: $IEMOCAP_DIR"
else
  log "Preprocessing IEMOCAP ..."
  $PYTHON prep_data.py --dataset iemocap --out_dir "$IEMOCAP_DIR"
fi

# ── Fixed data args passed to every runner invocation ────────────────────────
DATA_ARGS=(
  --local_data      "$IEMOCAP_DIR"
  --utterance_col   "Utterance"
  --speaker_col     "Speaker"
  --emotion_col     "Emotion"
  --dialogue_id_col "Dialogue_ID"
  --llama_model     "$LLAMA_MODEL"
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
prior = re.search(r'Prior F1\s*:\s*([\d.]+)',      block)
recog = re.search(r'Recognition F1\s*:\s*([\d.]+)', block)
gap   = re.search(r'gap vs prior ([+-][\d.]+)',      block)
print(
    prior.group(1) if prior else 'N/A',
    recog.group(1) if recog else 'N/A',
    gap.group(1)   if gap   else 'N/A',
)
PYEOF
}

# ── Run one ablation ──────────────────────────────────────────────────────────
run_ablation() {
  local name="$1"
  local desc="$2"
  shift 2

  local out_dir="$OUTPUT_BASE/$name"
  local log_file="$out_dir/train.log"

  header "[$((CURRENT_IDX))/${#ABLATIONS[@]}] $name"
  log "Description : $desc"
  mkdir -p "$out_dir"

  local exit_code=0
  $PYTHON ablation_runner.py \
    --name       "$name"       \
    --output_dir "$out_dir"    \
    --device     "$DEVICE"     \
    --batch_size "$BATCH_SIZE" \
    "${DATA_ARGS[@]}"          \
    "$@" 2>&1 | tee "$log_file" || exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo -e "${RED}[FAILED]${NC} $name exited with code $exit_code — logging FAILED and continuing"
    printf '"%s","%s",FAILED,FAILED,FAILED\n' "$name" "$desc" >> "$RESULTS_CSV"
    return 0
  fi

  local prior_f1 recog_f1 gap
  read -r prior_f1 recog_f1 gap <<< "$(parse_test_metrics "$log_file")"

  printf '"%s","%s",%s,%s,%s\n' "$name" "$desc" "$prior_f1" "$recog_f1" "$gap" \
    >> "$RESULTS_CSV"

  ok "$name  →  Prior F1: $prior_f1  |  Recog F1: $recog_f1  |  Gap: $gap"
}

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION TABLE
# Format: "name|description|runner_args"
# Delimiter | must not appear in name or description.
# ══════════════════════════════════════════════════════════════════════════════
ABLATIONS=(
  "full_model\
|All components active: staged training, LoRA, label GRU, emotion embed, cross-speaker attn\
|--staged_training"

  "baseline\
|LLaMA-LoRA classifier only — no dispositional state, no speaker context\
|--baseline --epochs 31"

  "no_staged\
|Non-staged training — flat LR for 31 epochs, same total budget as staged\
|--epochs 31"

  "no_lora\
|Frozen LLaMA encoder throughout — no LoRA adaptation\
|--staged_training --no_lora"

  "no_label_gru\
|Ablate EmotionLabelContext GRU — label_context_dim set to 1\
|--staged_training --label_context_dim 1"

  "no_emotion_embed\
|Ablate past-emotion label embedding — emotion_label_embed_dim set to 1\
|--staged_training --emotion_label_embed_dim 1"

  "no_label_modules\
|Ablate all emotion-label conditioning — both label GRU and embed disabled\
|--staged_training --label_context_dim 1 --emotion_label_embed_dim 1"
)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

header "Dispositional Emotion Prediction — Ablation Study (IEMOCAP)"
log "Device     : $DEVICE"
log "Batch size : $BATCH_SIZE"
log "LLaMA      : $LLAMA_MODEL"
log "Data       : $IEMOCAP_DIR"
log "Configs    : ${#ABLATIONS[@]}"
log "Output     : $OUTPUT_BASE"
log "Results    : $RESULTS_CSV"

mkdir -p "$OUTPUT_BASE"
echo "name,description,test_prior_f1,test_recognition_f1,recog_prior_gap" \
  > "$RESULTS_CSV"

CURRENT_IDX=0
for entry in "${ABLATIONS[@]}"; do
  CURRENT_IDX=$((CURRENT_IDX + 1))

  IFS='|' read -r name desc args_str <<< "$entry"
  read -ra runner_args <<< "$args_str"

  run_ablation "$name" "$desc" "${runner_args[@]}"
done

# ── Summary table ─────────────────────────────────────────────────────────────
header "Results Summary  (${#ABLATIONS[@]} configurations)"
$PYTHON - "$RESULTS_CSV" <<'PYEOF'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
nw   = max(len(r['name']) for r in rows)

print(f"  {'Name':<{nw}}  {'Prior F1':>10}  {'Recog F1':>10}  {'Gap':>8}  {'vs full':>9}")
print(f"  {'-'*nw}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*9}")

ref = next((r for r in rows if r['name'] == 'full_model'), None)

def floatable(v):
    try: float(v); return True
    except (ValueError, TypeError): return False

for r in rows:
    prior  = r['test_prior_f1']
    recog  = r['test_recognition_f1']
    gap    = r['recog_prior_gap']
    vs_full = ''
    if (ref and r['name'] != 'full_model'
            and floatable(prior) and floatable(ref['test_prior_f1'])):
        diff    = float(prior) - float(ref['test_prior_f1'])
        vs_full = f"{diff:+.4f}"
    print(
        f"  {r['name']:<{nw}}"
        f"  {prior:>10}"
        f"  {recog:>10}"
        f"  {gap:>8}"
        f"  {vs_full:>9}"
    )
PYEOF

echo ""
ok "Done. Results saved to $RESULTS_CSV"
