#!/usr/bin/env bash
set -euo pipefail
VIEWS=""; OUT=""; CONFIG=""; TAG="semantic"; SCRIPT="pipeline/scripts/infer_sam3_multiclass_generic.py"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --views) VIEWS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$VIEWS" && -n "$OUT" && -n "$CONFIG" ]] || { echo "Usage: $0 --views DIR --out DIR --config YAML [--tag NAME]" >&2; exit 2; }
[[ -f "$VIEWS/view_params.json" ]] || { echo "Missing $VIEWS/view_params.json" >&2; exit 2; }
mapfile -t PROMPTS < <(python - "$CONFIG" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1]))
for p in cfg['sam3']['prompts']: print(p)
PY
)
if python pipeline/scripts/check_multiclass_masks.py --views "$VIEWS" --masks "$OUT" --prompts "${PROMPTS[@]}"; then
  echo "All local SAM3 masks are valid. Skipping HPC."
  exit 0
fi
mapfile -t CFG < <(python - "$CONFIG" <<'PY'
import sys,yaml
cfg=yaml.safe_load(open(sys.argv[1])); h=cfg['hpc']; s=cfg['sam3']
for k in ('user','host','work_dir','account','partition'): print(h[k])
print(s['hpc_conda_env']); print(h.get('cpus_per_task',8)); print(h.get('gpus',1)); print(h.get('time_limit','04:00:00'))
PY
)
R_USER="${CFG[0]}"; R_HOST="${CFG[1]}"; WORK="${CFG[2]}"; ACCOUNT="${CFG[3]}"; PARTITION="${CFG[4]}"; ENV="${CFG[5]}"; CPUS="${CFG[6]}"; GPUS="${CFG[7]}"; TIME="${CFG[8]}"
REMOTE="$R_USER@$R_HOST"; REMOTE_BASE="$WORK/pipeline_v2_${TAG}"; REMOTE_VIEWS="$REMOTE_BASE/views"; REMOTE_MASKS="$REMOTE_BASE/masks"
ssh "$REMOTE" "rm -rf '$REMOTE_VIEWS' '$REMOTE_MASKS'; mkdir -p '$REMOTE_VIEWS' '$REMOTE_MASKS'"
echo "Uploading JPG views to $REMOTE:$REMOTE_VIEWS"
scp -q "$VIEWS"/*.jpg "$REMOTE:$REMOTE_VIEWS/"
scp -q "$SCRIPT" "$REMOTE:$REMOTE_BASE/infer_sam3_multiclass_generic.py"
scp -q "$CONFIG" "$REMOTE:$REMOTE_BASE/pipeline_config.yaml"
printf -v PROMPT_ARGS ' %q' "${PROMPTS[@]}"
ssh "$REMOTE" "set +u; source activate '$ENV'; set -u; cd '$WORK/sam3_repo'; srun -t '$TIME' -N1 -n1 --cpus-per-task='$CPUS' --gres=gpu:'$GPUS' -p '$PARTITION' -A '$ACCOUNT' python -u '$REMOTE_BASE/infer_sam3_multiclass_generic.py' --config-yaml '$REMOTE_BASE/pipeline_config.yaml' --views-dir '$REMOTE_VIEWS' --output-dir '$REMOTE_MASKS' --prompts $PROMPT_ARGS"
rm -rf "$OUT"; mkdir -p "$OUT"
scp -q "$REMOTE:$REMOTE_MASKS/*_masks.npz" "$OUT/"
scp -q "$REMOTE:$REMOTE_MASKS/mask_report.csv" "$OUT/" || true
python pipeline/scripts/check_multiclass_masks.py --views "$VIEWS" --masks "$OUT" --prompts "${PROMPTS[@]}"
echo "SAM3 masks ready: $OUT"
