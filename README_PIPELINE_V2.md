# GARField Pipeline v2 — semantic integration candidate

This package is the first production integration of the successful experimental semantic workflow.
It **does not overwrite** the current `pipeline/Snakefile` or `pipeline/config.yaml`.

## Replaced stages

Old path:

```text
Fine HDBSCAN
→ SAM2 cluster-pair evidence
→ conservative cluster merging
→ old cluster-to-mask matching
```

Candidate v2 path:

```text
COLMAP
→ SAM3 grouping masks
→ GARField
→ GARField-Gauss
→ GARField 256-D feature projection
→ spherical K-means K=400
→ optional prototype average-linkage merge (cosine 0.90)
→ AUTO broad semantic views + multiclass SAM3
→ broad K400 cluster semantic baseline
→ AUTO or CUSTOM refinement views + multiclass SAM3
→ pointwise multiclass matching
→ generalized component-aware refinement
→ architecture > sky cleanup
→ final semantic point cloud + per-class PLYs
```

The component-aware stage covers `window`, `door`, `opening`, `column`, and `beam`.

## Why AUTO + CUSTOM views both remain

The automatic COLMAP-guided view bank is preserved for dataset independence and broad semantic coverage.
The refinement bank is configurable:

```yaml
semantic_views:
  refinement:
    mode: custom   # auto | custom
```

For the LSU Tower, `custom` uses the viewer path already validated in the experiments:

```text
pipeline/camera_paths/towerlsu.json
```

For another dataset, set `mode: auto`. The refinement stage then reuses the automatic view bank and no manual camera path is needed.

## K400 note

The currently successful semantic chain used K400-derived groups with the prototype cosine-0.90 merge enabled.
That is preserved by default:

```yaml
clustering:
  prototype_merge:
    enabled: true
    cosine_threshold: 0.90
```

To test raw K400 without the prototype merge, change only:

```yaml
prototype_merge:
  enabled: false
```

## Install safely

From `~/garfield`, unzip this package. It adds only new candidate files:

```bash
unzip -o garfield_pipeline_v2_integration.zip
```

The important new files are:

```text
pipeline/Snakefile.component_aware
pipeline/config.component_aware.yaml
pipeline/camera_paths/towerlsu.json
pipeline/scripts/
```

## Dry run FIRST

```bash
cd ~/garfield
conda activate nerfstudio3

snakemake \
  -s pipeline/Snakefile.component_aware \
  -n -p --cores 1
```

Do not launch the full DAG until the dry-run looks correct, because COLMAP, GARField training, rendering, and HPC SAM3 can be expensive.

The candidate final target is:

```text
outputs/towerlsu/production_v2/final_semantics/semantic_pointcloud.ply
```

## Local machine overrides

If `pipeline/config.local.yaml` exists, the candidate Snakefile deep-merges it over the candidate config. This lets machine-specific paths/settings stay outside the public repository.

## Deliberately not integrated yet

These are the next phase after the semantic DAG is validated:

1. video → images with FFmpeg;
2. your existing `brep_to_idf.py` as the final rule;
3. final GitHub cleanup and migration;
4. removal/replacement of legacy Ultralytics execution paths.
