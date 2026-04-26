#!/usr/bin/env bash
# Best config (no aux losses) + focal_gamma=2 to handle MELD class imbalance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="cuda"
BATCH_SIZE=8
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

NAME="focal_gamma2"
DESC="no_aux_losses + focal_gamma=2"
OUT_DIR="./ablation_checkpoints/$NAME"
LOG="$OUT_DIR/train.log"
mkdir -p "$OUT_DIR"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'
echo -e "\n${BOLD}${CYAN}══ $NAME ══${NC}"
echo -e "${CYAN}Desc:${NC} $DESC"

exit_code=0
$PYTHON ablation_runner.py \
  --name                    "$NAME"      \
  --output_dir              "$OUT_DIR"   \
  --device                  "$DEVICE"    \
  --batch_size              "$BATCH_SIZE" \
  --staged_training                      \
  --posterior_loss_weight   0.0          \
  --surprise_reg_weight     0.0          \
  --contrastive_loss_weight 0.0          \
  --future_pred_weight_1    0.0          \
  --future_pred_weight_2    0.0          \
  --focal_gamma             2.0          \
  2>&1 | tee "$LOG" || exit_code=$?

if [[ $exit_code -ne 0 ]]; then
  echo -e "\033[0;31m[FAILED]\033[0m exited with code $exit_code"
  exit $exit_code
fi

$PYTHON - "$LOG" <<'PYEOF'
import re, sys
with open(sys.argv[1]) as f:
    txt = f.read()
parts = txt.split('[TEST]')
if len(parts) < 2:
    print("No [TEST] block found"); sys.exit(0)
block = parts[-1]
prior = re.search(r'Prior F1\s*:\s*([\d.]+)', block)
post  = re.search(r'Posterior F1\s*:\s*([\d.]+)', block)
gap   = re.search(r'gap \+([\d.]+)', block)
print(f"\n{'='*50}")
print(f"  Prior F1    : {prior.group(1) if prior else 'N/A'}")
print(f"  Posterior F1: {post.group(1)  if post  else 'N/A'}")
print(f"  Gap         : {gap.group(1)   if gap   else 'N/A'}")
print(f"{'='*50}")
print(f"  Target to beat: no_aux_losses = 0.3765")
PYEOF
