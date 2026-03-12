#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_experiments.sh  —  Run all 8 Can experiments sequentially.
#
# Usage:
#   bash scripts/run_experiments.sh              # run all
#   bash scripts/run_experiments.sh e02 e03      # run specific experiments
#   SKIP="e04 e07" bash scripts/run_experiments.sh   # skip heavy fine-tune exps
#
# Prerequisites (run once before this script):
#   python scripts/download_dataset.py --task can --type ph
#   python scripts/download_dataset.py --task can --type mh
#
# Logs go to:  experiments/<name>/logs/
# Checkpoints: experiments/<name>/checkpoints/
# ─────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")/.."   # run from project root

# ── Experiment registry ───────────────────────────────────────────────
declare -A EXPERIMENT_NAMES
EXPERIMENT_NAMES=(
    [e00]="baseline_can_ph       | SmallCNN | can_ph         | obs=1"
    [e01]="phm_small_cnn         | SmallCNN | can_ph+mh      | obs=1"
    [e02]="ph_resnet18           | ResNet18 | can_ph         | obs=1  frozen"
    [e03]="phm_resnet18          | ResNet18 | can_ph+mh      | obs=1  frozen"
    [e04]="phm_resnet18_ft       | ResNet18 | can_ph+mh      | obs=1  finetune"
    [e05]="ph_resnet18_obs2      | ResNet18 | can_ph         | obs=2  frozen"
    [e06]="phm_resnet18_obs2     | ResNet18 | can_ph+mh      | obs=2  frozen"
    [e07]="phm_resnet18_obs2_ft  | ResNet18 | can_ph+mh      | obs=2  finetune"
)

ORDERED=(e00 e01 e02 e03 e04 e05 e06 e07)

# ── Parse args: if specific IDs given, only run those ─────────────────
if [ $# -gt 0 ]; then
    ORDERED=("$@")
fi

# ── Helper ────────────────────────────────────────────────────────────
log() { echo -e "\n\033[1;36m[run_experiments] $*\033[0m"; }
err() { echo -e "\033[1;31m[ERROR] $*\033[0m" >&2; }

# ── Main loop ─────────────────────────────────────────────────────────
TOTAL=${#ORDERED[@]}
PASSED=0
FAILED=()
START_ALL=$(date +%s)

for i in "${!ORDERED[@]}"; do
    EXP="${ORDERED[$i]}"
    CONFIG="experiments/${EXP}_*/config.yaml"
    # Expand glob
    CONFIG_FILE=$(ls $CONFIG 2>/dev/null | head -1)

    if [ -z "$CONFIG_FILE" ]; then
        err "Config not found for $EXP (looked for $CONFIG) — skipping"
        FAILED+=("$EXP")
        continue
    fi

    # Skip if in SKIP env var
    if [[ " ${SKIP:-} " == *" $EXP "* ]]; then
        log "SKIPPING $EXP (in SKIP list)"
        continue
    fi

    DESC="${EXPERIMENT_NAMES[$EXP]:-$EXP}"
    log "[$((i+1))/$TOTAL] Starting $EXP — $DESC"
    echo "  Config: $CONFIG_FILE"
    START=$(date +%s)

    python scripts/train_visual.py \
        --config "$CONFIG_FILE" \
        --run_name "$EXP" \
        && STATUS="✓ DONE" || { STATUS="✗ FAILED"; FAILED+=("$EXP"); }

    END=$(date +%s)
    ELAPSED=$(( (END - START) / 60 ))
    log "$EXP $STATUS  (${ELAPSED}m)"

    [ "$STATUS" = "✓ DONE" ] && PASSED=$((PASSED+1))
done

# ── Summary ───────────────────────────────────────────────────────────
TOTAL_MIN=$(( ($(date +%s) - START_ALL) / 60 ))
echo ""
echo "═══════════════════════════════════════════════"
echo "  Experiments complete: $PASSED / $TOTAL passed"
echo "  Total time: ${TOTAL_MIN}m"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED[*]}"
fi
echo "═══════════════════════════════════════════════"
echo ""
echo "Next: evaluate checkpoints with:"
echo "  python scripts/evaluate_robosuite_visual.py \\"
echo "    --checkpoint experiments/<name>/checkpoints/best.pt \\"
echo "    --config experiments/<name>/config.yaml \\"
echo "    --num_episodes 50 --num_ddim_steps 20"
