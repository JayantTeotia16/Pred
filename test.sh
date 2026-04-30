#!/usr/bin/env bash
# =============================================================================
# test.sh — Test new model ideas on IEMOCAP
#
# Currently tests: Counterfactual Speaker Modeling (model_v2.py)
#
# Results → checkpoints_v2/test_metrics.json
#           checkpoints_v2/gate_analysis.json
#
# Usage:
#   bash test.sh [--device cuda] [--batch_size 4] [--llama_model MODEL]
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

IEMOCAP_DIR="./data/iemocap"
OUTPUT_DIR="./checkpoints_v2"

PYTHON=""
for c in python3 python; do
  if command -v "$c" &>/dev/null; then PYTHON="$c"; break; fi
done
[[ -z "$PYTHON" ]] && { echo "[ERROR] Python 3 not found"; exit 1; }

CYAN='\033[0;36m'; GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'
log()    { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
header() {
  echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}"
  echo -e "${BOLD}${CYAN}  $*${NC}"
  echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}\n"
}

# ── Ensure IEMOCAP data exists ────────────────────────────────────────────────
if [[ -f "$IEMOCAP_DIR/train.csv" && -f "$IEMOCAP_DIR/val.csv" && -f "$IEMOCAP_DIR/test.csv" ]]; then
  ok "IEMOCAP data present: $IEMOCAP_DIR"
else
  log "Preprocessing IEMOCAP ..."
  $PYTHON prep_data.py --dataset iemocap --out_dir "$IEMOCAP_DIR"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
header "Counterfactual Speaker Modeling — IEMOCAP"
log "Device     : $DEVICE"
log "Batch size : $BATCH_SIZE"
log "LLaMA      : $LLAMA_MODEL"
log "Output     : $OUTPUT_DIR"

$PYTHON test_model.py \
  --device      "$DEVICE"      \
  --batch_size  "$BATCH_SIZE"  \
  --llama_model "$LLAMA_MODEL" \
  --local_data  "$IEMOCAP_DIR" \
  --output_dir  "$OUTPUT_DIR"  \
  2>&1 | tee "$OUTPUT_DIR/train.log"

# ── Print summary ─────────────────────────────────────────────────────────────
header "Summary"

$PYTHON - "$OUTPUT_DIR" <<'PYEOF'
import json, sys, os

out = sys.argv[1]

metrics_path = os.path.join(out, "test_metrics.json")
gate_path    = os.path.join(out, "gate_analysis.json")

if os.path.isfile(metrics_path):
    m = json.load(open(metrics_path))
    print(f"  Prior F1      : {m.get('weighted_f1',   'N/A')}")
    print(f"  Recognition F1: {m.get('recognition_f1','N/A')}")
    print(f"  Accuracy      : {m.get('accuracy',      'N/A')}")
    print(f"  Mean Surprise : {m.get('mean_surprise',  'N/A')}")

if os.path.isfile(gate_path):
    g = json.load(open(gate_path))
    print(f"\n  Reactivity Head Analysis:")
    print(f"  AUROC                      : {g.get('auroc', 'N/A')}")
    print(f"  F1 (transition class)      : {g.get('f1_transition_class', 'N/A')}")
    print(f"  Mean reactivity            : {g.get('mean_reactivity', 'N/A')}")
    print(f"  Reactivity @ transitions   : {g.get('reactivity_at_transition', 'N/A')}")
    print(f"  Reactivity @ stable        : {g.get('reactivity_at_stable', 'N/A')}")
    lift = g.get('reactivity_lift_at_transition')
    if lift is not None:
        direction = 'reactive ↑ at transitions' if lift > 0 else 'no clear pattern'
        print(f"  Reactivity lift            : {lift:.4f}  ({direction})")
    print(f"\n  Per-speaker reactivity:")
    for spk, val in g.get("per_speaker_reactivity", {}).items():
        bar = "█" * int(val * 20)
        print(f"    Speaker {spk}: {val:.4f}  {bar}")
PYEOF

ok "Done. Results in $OUTPUT_DIR/"
