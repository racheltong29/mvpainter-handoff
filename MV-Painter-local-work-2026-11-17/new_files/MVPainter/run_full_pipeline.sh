#!/bin/bash
# Runs the full MVPainter pipeline (multiview paint -> PBR decompose -> bake/combine)
# on a single mesh + reference image, end to end, in one command.
#
# Usage: ./run_full_pipeline.sh <input.glb> <reference_image.png> [output_dir] [geo_rotation]
# If output_dir is omitted, defaults to MVPainter/runs/<glb basename>.
set -e

GLB="$1"
IMG="$2"
OUT="$3"
GEO_ROTATION="${4:-0}"

if [ -z "$GLB" ] || [ -z "$IMG" ]; then
    echo "Usage: $0 <input.glb> <reference_image.png> [output_dir] [geo_rotation]"
    exit 1
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLB="$(realpath "$GLB")"
IMG="$(realpath "$IMG")"

if [ -z "$OUT" ]; then
    OUT="$THIS_DIR/runs/$(basename "$GLB" .glb)"
fi
OUT="$(mkdir -p "$OUT" && realpath "$OUT")"

# Use the env's python by absolute path rather than bare `python` after
# `conda activate` - this shell's PATH has a stale project .venv ahead of
# conda's entries, so bare `python`/`conda activate` alone silently resolves
# to the wrong interpreter (no torch/numpy). Same reasoning as infer_paint.py's
# own _PYTHON = sys.executable pattern.
PYTHON="/home/rachelto/anaconda3/envs/mvpainter/bin/python"
export LD_LIBRARY_PATH="/home/rachelto/anaconda3/envs/mvpainter/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GLB_DIR="$OUT/_input_glb"
IMG_DIR="$OUT/_input_img"
mkdir -p "$GLB_DIR" "$IMG_DIR"
# infer_multiview.py pairs files by basename (glb_dir/X.glb <-> img_dir/X.png),
# so the copies must share a basename even if the source files don't
# (e.g. eval_assets_e2e's _input_mesh.glb + view_gt_shaded.png).
cp "$GLB" "$GLB_DIR/asset.glb"
cp "$IMG" "$IMG_DIR/asset.png"

cd "$THIS_DIR"

echo "=== [1/3] infer_multiview.py ==="
"$PYTHON" infer_multiview.py \
    --input_glb_dir "$GLB_DIR" \
    --input_img_dir "$IMG_DIR" \
    --output_dir "$OUT/mv_res" \
    --geo_rotation "$GEO_ROTATION" --diffusion_steps 75 --seed 12

echo "=== [2/3] infer_pbr.py ==="
"$PYTHON" infer_pbr.py \
    --mv_res_dir "$OUT/mv_res/mvpainter"

echo "=== [3/3] infer_paint.py ==="
"$PYTHON" infer_paint.py \
    --mv_res_dir "$OUT/mv_res/mvpainter" \
    --output_dir "$OUT/painted" \
    --geo_rotation "$GEO_ROTATION" --use_pbr

echo "Done. Final textured GLB(s) in: $OUT/painted/combine_pbr/"
