#!/usr/bin/env bash
set -euo pipefail

POINTS="${POINTS:-$HOME/garfield-semantic/outputs/ortho_projection_s005/points.npy}"
VIEWS="${VIEWS:-$HOME/garfield-semantic/outputs/labeling_views}"
MASKS="${MASKS:-$HOME/garfield-semantic/outputs/labeling_masks_finetuned}"
OUT="${OUT:-$HOME/garfield/outputs/PFTdrone_direct_sam3_quick}"

SCRIPT="$HOME/garfield/feature_clustering_experiments/direct_sam3_pftdrone_quick/direct_sam3_pftdrone_quick.py"

echo "POINTS=$POINTS"
echo "VIEWS=$VIEWS"
echo "MASKS=$MASKS"
echo "OUT=$OUT"
echo

/usr/bin/time -v \
  -o "$OUT.time.txt" \
  python "$SCRIPT" \
    --points "$POINTS" \
    --views "$VIEWS" \
    --masks "$MASKS" \
    --output "$OUT"

echo
echo "Final semantic cloud:"
echo "  $OUT/semantic_pointcloud.ply"
echo "Per-class clouds:"
echo "  $OUT/per_class/"
echo "Timing:"
echo "  $OUT.time.txt"
