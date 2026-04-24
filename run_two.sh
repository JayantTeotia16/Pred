#!/usr/bin/env bash
# Quick sanity run: baseline (error-prone) then full_model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="cuda"
BATCH_SIZE=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)     DEVICE="$2";     shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

PYTHON=""
for c in python3 python; do
  if command -v "$c" &>/dev/null; then PYTHON="$c"; break; fi
done
[[ -z "$PYTHON" ]] && { echo "[ERROR] Python 3 not found"; exit 1; }

OUTPUT_BASE="./ablation_checkpoints"
RESULTS_CSV="./run_two_results.csv"
mkdir -p "$OUTPUT_BASE"
echo "name,description,test_prior_f1,test_posterior_f1,prior_posterior_gap" > "$RESULTS_CSV"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()  { echo -e "${GREEN}[OK]${NC} $*"; }

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

run_one() {
  local name="$1"; local desc="$2"; shift 2
  local out_dir="$OUTPUT_BASE/$name"
  local log_file="$out_dir/train.log"
  echo -e "\n${BOLD}${CYAN}══ $name ══${NC}"
  log "Desc: $desc"
  mkdir -p "$out_dir"

  local exit_code=0
  $PYTHON ablation_runner.py \
    --name "$name" --output_dir "$out_dir" \
    --device "$DEVICE" --batch_size "$BATCH_SIZE" \
    "$@" 2>&1 | tee "$log_file" || exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo -e "${RED}[FAILED]${NC} $name exited $exit_code"
    printf '"%s","%s",FAILED,FAILED,FAILED\n' "$name" "$desc" >> "$RESULTS_CSV"
    return 0
  fi

  local prior_f1 post_f1 gap
  read -r prior_f1 post_f1 gap <<< "$(parse_test_metrics "$log_file")"
  printf '"%s","%s",%s,%s,%s\n' "$name" "$desc" "$prior_f1" "$post_f1" "$gap" >> "$RESULTS_CSV"
  ok "$name → Prior F1: $prior_f1 | Post F1: $post_f1 | Gap: $gap"
}

# 1. Baseline first (the one that had the error)
run_one "baseline" \
  "LLaMA-LoRA classifier only — no dispositional state, no speaker context, no scene" \
  --baseline --epochs 16

# 2. Full model
run_one "full_model" \
  "All components: staged + scene + LoRA + label GRU + emotion embed + cross-speaker attn + all losses" \
  --staged_training

echo ""
ok "Done. Results in $RESULTS_CSV"
