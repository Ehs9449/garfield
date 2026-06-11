# GARField-Semantic: Drone-to-BIM Semantic Segmentation Pipeline

Automatic semantic segmentation of building components from drone imagery using GARField (Group Anything with Radiance Fields), 3D Gaussian Splatting, and domain-adapted SAM 3.

> Forked from [chungmin99/garfield](https://github.com/chungmin99/garfield)

## Pipeline Overview

```
                                    ┌─────────────────────┐
                                    │   Drone Images      │
                                    └─────────┬───────────┘
                                              │
                                    ┌─────────▼───────────┐
                            Stage 1 │   COLMAP (SfM)      │
                                    │   Camera poses +     │
                                    │   sparse point cloud │
                                    └─────────┬───────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                    ┌─────────▼───────────┐        ┌──────────▼──────────┐
            Stage 2a│   GARField (NeRF)   │ Stage 2b│  GARField-Gauss    │
                    │   256-dim grouping  │        │  Gaussian Splatting │
                    │   features          │        │  RGB rendering      │
                    └─────────┬───────────┘        └──────────┬──────────┘
                              │                               │
                    ┌─────────▼───────────┐        ┌──────────▼──────────┐
            Stage 3 │  Ortho Projection   │ Stage 5 │  Render 21 Views   │
                    │  37 views → 256-dim │        │  for SAM 3 labeling│
                    │  features on points │        └──────────┬──────────┘
                    └─────────┬───────────┘                   │
                              │                    ┌──────────▼──────────┐
                    ┌─────────▼───────────┐ Stage 6 │  SAM 3 PCS (HPC)  │
            Stage 4 │  HDBSCAN Clustering │        │  Fine-tuned on HDB│
                    │  61 clusters        │        │  Text-prompted     │
                    └─────────┬───────────┘        └──────────┬──────────┘
                              │                               │
                              └───────────────┬───────────────┘
                                              │
                                    ┌─────────▼───────────┐
                            Stage 7 │  Cluster Matching   │
                                    │  IoU + majority vote│
                                    └─────────┬───────────┘
                                              │
                                    ┌─────────▼───────────┐
                                    │ Semantic Point Cloud│
                                    │ 267,979 labeled pts │
                                    └─────────────────────┘
```

| Stage | Description | Model | Environment |
|-------|------------|-------|-------------|
| 1. COLMAP | Structure from Motion | COLMAP | ColmapLnx |
| 2a. GARField | Train grouping features (NeRF) | GARField | nerfstudio3 |
| 2b. GARField-Gauss | Train Gaussian Splatting for rendering | GARField-Gauss | nerfstudio3 |
| 3. Projection | Orthographic feature projection to point cloud | Uses 2a | nerfstudio3 |
| 4. Clustering | HDBSCAN on GARField features | — | nerfstudio3 |
| 5. Render | Generate labeling views | Uses 2b | nerfstudio3 |
| 6. SAM 3 PCS | Text-prompted segmentation | Fine-tuned SAM 3 | sam3 (HPC) |
| 7. Matching | Cluster-to-mask IoU matching + majority vote | — | nerfstudio3 |

**Two parallel branches:** GARField (2a) provides grouping features for clustering. GARField-Gauss (2b) provides clean rendered views for SAM 3 labeling. Both branches merge at Stage 7 where cluster labels are assigned via majority voting.

## Quick Start

### Prerequisites

- COLMAP (conda env: `ColmapLnx`)
- Nerfstudio with GARField (conda env: `nerfstudio3`)
- SAM 3 with fine-tuned checkpoint (conda env: `sam3`, on HPC)
- Snakemake: `pip install snakemake pyyaml`

### Run

1. Place your drone images in `data/YOUR_PROJECT/images/`
2. Edit `pipeline/config.yaml` with your dataset paths and parameters
3. Run the full pipeline:

```bash
snakemake -s pipeline/Snakefile --cores 4 all
```

Or run individual stages:

```bash
snakemake -s pipeline/Snakefile --cores 4 colmap              # Stage 1: COLMAP only
snakemake -s pipeline/Snakefile --cores 4 garfield_train       # Stage 2a: GARField training
snakemake -s pipeline/Snakefile --cores 4 garfield_gauss_train # Stage 2b: Gaussian Splatting
snakemake -s pipeline/Snakefile --cores 4 clustering           # Up to clustering (stages 1-4)
snakemake -s pipeline/Snakefile --cores 4 sam3_inference        # SAM 3 on HPC (stage 6)
snakemake -s pipeline/Snakefile --cores 4 semantic_labeling     # Final matching (stage 7)
snakemake -s pipeline/Snakefile -n all                         # Dry run (show plan)
```

## Repository Structure

```
garfield-semantic/
├── README.md                              # This file
├── pyproject.toml                         # Package config (from original GARField)
│
├── garfield/                              # GARField source code
│   ├── garfield_config.py                 # GARField configuration
│   ├── garfield_model.py                  # GARField model
│   ├── garfield_field.py                  # GARField neural field
│   ├── garfield_gaussian_pipeline.py      # MODIFIED: feature rendering support
│   ├── garfield_pipeline.py               # GARField pipeline
│   ├── garfield_datamanager.py            # Data loading
│   ├── img_group_model.py                 # MODIFIED: SAM2 mask generation
│   ├── garfield_ortho_projection.py       # NEW: orthographic feature projection
│   ├── render_views_for_labeling.py       # NEW: render views for SAM 3
│   ├── run_sam3_pcs.py                    # NEW: SAM 3 PCS inference
│   ├── match_clusters_to_masks.py         # NEW: cluster matching (off-the-shelf)
│   ├── match_clusters_to_masks_finetuned.py  # NEW: cluster matching (fine-tuned)
│   ├── label_clusters_sam3.py             # NEW: end-to-end SAM 3 labeling
│   └── label_clusters_nerf.py             # NEW: NeRF-based labeling
│
├── pipeline/                              # Pipeline orchestration
│   ├── Snakefile                          # Snakemake pipeline definition
│   ├── config.yaml                        # All paths and parameters
│   └── scripts/
│       └── run_pipeline.sh                # Bash script alternative
│
├── sam3_finetune/                          # SAM 3 fine-tuning on building facades
│   ├── convert_hdb_to_coco.py             # HDB dataset conversion (VIA → COCO)
│   ├── hdb_building_finetune.yaml         # SAM 3 training config
│   └── README.md                          # Fine-tuning instructions
│
├── data/                                  # Input data (not tracked by git)
│   └── YOUR_PROJECT/
│       └── images/                        # Drone images
│
└── outputs/                               # Pipeline outputs (not tracked by git)
    ├── YOUR_PROJECT/
    │   ├── garfield/.../config.yml        # Stage 2a output
    │   └── garfield-gauss/.../config.yml  # Stage 2b output
    ├── ortho_projection/                  # Stage 3 output
    ├── labeling_views/                    # Stage 5 output
    ├── labeling_masks_finetuned/          # Stage 6 output
    └── semantic_labels_finetuned/         # Stage 7 output
        ├── semantic_pointcloud.ply        # Final labeled point cloud
        ├── semantic_labels.json           # Per-cluster labels
        └── per_label/                     # Individual PLY per class
```

## SAM 3 Fine-Tuning

We fine-tuned SAM 3 on the HDB building facade dataset (2,679 images, 11 classes) to improve building element detection. See `sam3_finetune/README.md` for details.

**Results:**
- AP@50: 34.6% on HDB validation set
- Semantic labeling improved from 62% roof / 37% unknown (off-the-shelf) to 33% wall / 23% roof / 3% unknown (fine-tuned)

## Citation

If you use this work, please cite:

```bibtex
@misc{garfield_semantic_2026,
  title={Unsupervised Building Component Discovery via Orthographic Feature Projection from Neural Radiance Fields},
  author={Ehsan Aghazadeh},
  year={2026}
}
```

## Acknowledgments

- [GARField](https://github.com/chungmin99/garfield) - Group Anything with Radiance Fields
- [SAM 3](https://github.com/facebookresearch/sam3) - Segment Anything Model 3
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) - NeRF framework
- HDB Building Facade Dataset - University of Hong Kong
