#!/usr/bin/env bash
# =============================================================================
# ablation.sh — Component ablation study for Dispositional Emotion Prediction
#
# Defines all 12 ablation configurations in a table and loops through them.
# No existing files are modified — overrides are applied in-memory by
# ablation_runner.py.
#
# Results → ablation_results.csv
# Logs    → ablation_checkpoints/<name>/train.log
#
# Usage:
#   bash ablation.sh [--device cuda|cpu] [--batch_size N]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse flags ───────────────────────────────────────────────────────────────
DEVICE="cuda"
BATCH_SIZE=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)     DEVICE="$2";     shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

OUTPUT_BASE="./ablation_checkpoints"
RESULTS_CSV="./ablation_results.csv"

PYTHON=""
for c in python3 python; do
  if command -v "$c" &>/dev/null; then PYTHON="$c"; break; fi
done
[[ -z "$PYTHON" ]] && { echo "[ERROR] Python 3 not found"; exit 1; }

# ── Colours ───────────────────────────────────────────────────────────────────
CYAN='\033[0;36m'; GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'
log()    { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
header() {
  echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}\n"
}

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
prior = re.search(r'Prior F1\s*:\s*([\d.]+)',     block)
post  = re.search(r'Posterior F1\s*:\s*([\d.]+)', block)
gap   = re.search(r'gap \+([\d.]+)',              block)
print(
    prior.group(1) if prior else 'N/A',
    post.group(1)  if post  else 'N/A',
    gap.group(1)   if gap   else 'N/A',
)
PYEOF
}

# ── Run one ablation ──────────────────────────────────────────────────────────
# Usage: run_ablation <name> <description> [runner_args...]
run_ablation() {
  local name="$1"
  local desc="$2"
  shift 2

  local out_dir="$OUTPUT_BASE/$name"
  local log_file="$out_dir/train.log"

  header "[$((CURRENT_IDX))/${#ABLATIONS[@]}] $name"
  log "Description : $desc"
  mkdir -p "$out_dir"

  $PYTHON ablation_runner.py \
    --name       "$name"       \
    --output_dir "$out_dir"    \
    --device     "$DEVICE"     \
    --batch_size "$BATCH_SIZE" \
    "$@" 2>&1 | tee "$log_file"

  local prior_f1 post_f1 gap
  read -r prior_f1 post_f1 gap <<< "$(parse_test_metrics "$log_file")"

  printf '"%s","%s",%s,%s,%s\n' "$name" "$desc" "$prior_f1" "$post_f1" "$gap" \
    >> "$RESULTS_CSV"

  ok "$name  →  Prior F1: $prior_f1  |  Post F1: $post_f1  |  Gap: $gap"
}

# ══════════════════════════════════════════════════════════════════════════════
# ABLATION TABLE
# Format: "name|description|runner_args"
# Delimiter | must not appear in name or description.
# Runner args are word-split, so no quoted spaces within the args field.
# ══════════════════════════════════════════════════════════════════════════════
ABLATIONS=(
  # ── Module ablations ───────────────────────────────────────────────────────
  "full_model\
|All components: staged + scene + LoRA + label GRU + emotion embed + cross-speaker attn + all losses\
|--staged_training"

  "baseline\
|LLaMA-LoRA classifier only — no dispositional state, no speaker context, no scene\
|--baseline --epochs 16"

  "no_scene\
|Ablate SceneDynamicsField — no shared affective field across speakers\
|--staged_training --no_scene"

  "no_lora\
|Ablate LoRA — frozen LLaMA encoder throughout all phases\
|--staged_training --no_lora"

  "no_label_gru\
|Ablate EmotionLabelContext GRU — label_context_dim=1 (effectively disabled)\
|--staged_training --label_context_dim 1"

  "no_emotion_embed\
|Ablate past-emotion label embedding — emotion_label_embed_dim=1 (effectively disabled)\
|--staged_training --emotion_label_embed_dim 1"

  "no_label_modules\
|Ablate all emotion-label conditioning — both label GRU and past-emotion embed disabled\
|--staged_training --label_context_dim 1 --emotion_label_embed_dim 1"

  # ── Loss ablations ─────────────────────────────────────────────────────────
  "no_future_pred\
|Ablate future-prediction auxiliary losses — w_fut1=0 w_fut2=0\
|--staged_training --future_pred_weight_1 0.0 --future_pred_weight_2 0.0"

  "no_posterior\
|Ablate posterior head loss — w_post=0 (prior training signal only)\
|--staged_training --posterior_loss_weight 0.0"

  "no_surprise\
|Ablate surprise calibration regularizer — w_surp=0\
|--staged_training --surprise_reg_weight 0.0"

  "no_contrastive\
|Ablate contrastive speaker loss — w_cont=0\
|--staged_training --contrastive_loss_weight 0.0"

  "no_aux_losses\
|All auxiliary losses off — prior CE only (no future, posterior, surprise, contrastive)\
|--staged_training --future_pred_weight_1 0.0 --future_pred_weight_2 0.0 --posterior_loss_weight 0.0 --surprise_reg_weight 0.0 --contrastive_loss_weight 0.0"
)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

header "Dispositional Emotion Prediction — Ablation Study"
log "Device     : $DEVICE"
log "Batch size : $BATCH_SIZE"
log "Configs    : ${#ABLATIONS[@]}"
log "Output     : $OUTPUT_BASE"
log "Results    : $RESULTS_CSV"

mkdir -p "$OUTPUT_BASE"
echo "name,description,test_prior_f1,test_posterior_f1,prior_posterior_gap" \
  > "$RESULTS_CSV"

CURRENT_IDX=0
for entry in "${ABLATIONS[@]}"; do
  CURRENT_IDX=$((CURRENT_IDX + 1))

  # Split on | into name, description, args string
  IFS='|' read -r name desc args_str <<< "$entry"

  # Word-split args_str into an array
  read -ra runner_args <<< "$args_str"

  run_ablation "$name" "$desc" "${runner_args[@]}"
done

# ── Summary table ─────────────────────────────────────────────────────────────
header "Results Summary  (${#ABLATIONS[@]} configurations)"
$PYTHON - "$RESULTS_CSV" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
nw = max(len(r['name']) for r in rows)
print(f"  {'Name':<{nw}}  {'Prior F1':>10}  {'Post F1':>10}  {'Gap':>8}")
print(f"  {'-'*nw}  {'-'*10}  {'-'*10}  {'-'*8}")
ref = next((r for r in rows if r['name'] == 'full_model'), None)
for r in rows:
    prior = r['test_prior_f1']
    delta = ''
    if ref and prior != 'N/A' and ref['test_prior_f1'] != 'N/A' and r['name'] != 'full_model':
        diff = float(prior) - float(ref['test_prior_f1'])
        delta = f"  ({diff:+.4f})"
    print(
        f"  {r['name']:<{nw}}"
        f"  {prior:>10}"
        f"  {r['test_posterior_f1']:>10}"
        f"  {r['prior_posterior_gap']:>8}"
        f"{delta}"
    )
PYEOF

echo ""
ok "Done. Results saved to $RESULTS_CSV"
