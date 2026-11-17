#!/bin/bash
# Runs the one-shot pipeline on all 10 originally-failing eval assets.
EVAL_DIR=/home/rachelto/mvpainter_handoff/eval_assets_e2e
UIDS_FILE="${1:-/home/rachelto/mvpainter_handoff/_failed_uids.txt}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while IFS= read -r uid; do
    echo "=========================================="
    echo "UID: $uid"
    echo "Start: $(date)"
    "$THIS_DIR/run_full_pipeline.sh" \
        "$EVAL_DIR/$uid/_input_mesh.glb" \
        "$EVAL_DIR/$uid/view_gt_shaded.png" \
        "$THIS_DIR/runs/$uid" \
        0
    echo "Finish: $(date)"
done < "$UIDS_FILE"

echo "=========================================="
echo "Batch complete."
