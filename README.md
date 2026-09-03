# AI-Enabled Geometry Generation, Segmentation, and Interoperability for Digital Twin

Automatic semantic segmentation of building components from drone imagery using GARField (Group Anything with Radiance Fields), 3D Gaussian Splatting, and domain-adapted SAM 3.

> Forked from [chungmin99/garfield](https://github.com/chungmin99/garfield)

## Overview and Purpose

This repository provides the AI-Enabled Geometry Generation and Interoperability component of the Socio-Technical Cyberinfrastructure for Digital Twins (STC-DT) ecosystem. Its purpose is to transform drone imagery into semantically enriched 3D representations of buildings that can support downstream digital twin and BIM workflows.

The current pipeline combines photogrammetry, neural scene representation, feature-based grouping, semantic segmentation, and point-cloud labeling. It is designed to identify and organize building components such as walls, roofs, windows, and doors, and to generate structured geometric outputs that can be used for digital twin development, scan-to-BIM processing, and interoperability with other STC-DT components.

## Key Capabilities

- **Drone-to-3D reconstruction:** Processes multi-view drone imagery using COLMAP to estimate camera poses and generate the geometric foundation for 3D reconstruction.
- **Neural 3D scene representation:** Uses GARField and 3D Gaussian Splatting to reconstruct the scene and extract grouping features across multiple viewpoints.
- **Building-component segmentation:** Uses a domain-adapted SAM 3 model to identify building components such as walls, roofs, windows, and doors.
- **3D feature clustering:** Projects learned GARField features into 3D and groups geometrically and semantically related points using GPU-accelerated HDBSCAN with parameter search based on silhouette score.
- **Semantic 3D labeling:** Associates 2D segmentation masks with 3D clusters to generate semantically labeled point clouds.
- **Geometry processing:** Supports downstream geometric processing of reconstructed building components for digital twin and BIM applications.
- **Reproducible workflow:** Integrates the major processing stages through a configurable Snakemake pipeline.

## Role in the STC-DT Ecosystem

This repository serves as the geometry-focused component of the STC-DT ecosystem. It converts visual observations of the built environment into semantically enriched 3D representations that can be exchanged with other digital twin components.

Within STC-DT, the generated geometry and semantic information can provide a spatial foundation for integration with knowledge graphs, BIM information, infrastructure data, sensor information, and other digital twin resources. The component therefore connects AI-based reality capture and scene understanding with higher-level semantic and digital twin interoperability workflows.

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
            Stage 4 │ Optuna + HDBSCAN    │        │  Fine-tuned on HDB│
                    │ Hyperparameter      │        │  Text-prompted     │
                    │ optimization        │        │                    │
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
| 4. Clustering | Optuna-optimized HDBSCAN | Optuna + cuML HDBSCAN | nerfstudio3 |
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

### Configuration

Create your local configuration from the provided template:

```bash
cp pipeline/config.example.yaml pipeline/config.yaml
```

Then edit `pipeline/config.yaml` for your machine, dataset, and HPC environment.

The main project-specific settings are:

| Variable | Description |
|---|---|
| `dataset.name` | Project name used to organize the pipeline outputs |
| `dataset.path` | Path to the input drone-image dataset |
| `colmap.conda_env` | Conda environment containing COLMAP |
| `garfield.conda_env` | Conda environment containing Nerfstudio and GARField |
| `sam3.hpc_conda_env` | Path to the SAM 3 Conda environment on the HPC system |
| `sam3.checkpoint` | Path to the fine-tuned SAM 3 checkpoint |
| `sam3.bpe_path` | Path to the SAM 3 BPE vocabulary file |
| `hpc.user` | HPC username |
| `hpc.host` | HPC hostname |
| `hpc.work_dir` | Working directory on the HPC system |
| `hpc.account` | HPC allocation/account |
| `hpc.partition` | GPU partition used for SAM 3 inference |

Other parameters, such as projection scale, clustering settings, labeling viewpoints, SAM 3 confidence threshold and prompts, and matching threshold, have example values in `pipeline/config.example.yaml` and can be adjusted for a specific experiment.

The `sam3_finetune` settings are only required when fine-tuning SAM 3. If an existing fine-tuned checkpoint is used, configure `sam3.checkpoint` instead.

`pipeline/config.yaml` is intentionally excluded from Git because it contains machine-specific configuration. Do not store passwords, API keys, access tokens, or other secrets in this file.

### Run

1. Place your drone images in the `images/` subdirectory of the dataset path specified by `dataset.path` in `pipeline/config.yaml`
2. Edit `pipeline/config.yaml` with your dataset paths and parameters
3. Run the full pipeline:

```bash
snakemake -s pipeline/Snakefile --cores 4 all
```

Verify pipeline status:

```bash
snakemake -s pipeline/Snakefile all --cores 4 -n -p
```

Expected output:

```text
Nothing to be done
(all requested files are present and up to date)
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
├── LICENSE
├── pyproject.toml                         # Package config (from original GARField)
│
├── garfield/                              # GARField source code
│   ├── garfield_config.py                 # GARField configuration
│   ├── garfield_model.py                  # GARField model
│   ├── garfield_field.py                  # GARField neural field
│   ├── garfield_pipeline.py               # GARField pipeline
│   ├── garfield_gaussian_pipeline.py      # MODIFIED: feature rendering support
│   ├── garfield_datamanager.py            # Data loading
│   ├── garfield_pixel_sampler.py          # Pixel sampling
│   ├── garfield_interaction.py            # Interaction utilities
│   ├── img_group_model.py                 # MODIFIED: SAM2 mask generation
│   │
│   ├── garfield_ortho_projection.py       # Stage 3: orthographic feature projection
│   ├── cluster_sweep.py                   # Stage 4: Optuna-optimized HDBSCAN
│   ├── render_views_for_labeling.py       # Stage 5: render views for SAM 3
│   ├── run_sam3_finetuned_pcs.py          # Stage 6: fine-tuned SAM 3 inference
│   └── match_clusters_to_masks_finetuned.py  # Stage 7: cluster matching
│
├── pipeline/                              # Pipeline orchestration
│   ├── Snakefile                          # Snakemake pipeline definition
│   ├── config.yaml                        # All paths and parameters
│   ├── visualize_pipeline.py              # Per-stage diagnostics + charts
│   └── generate_dashboard.py              # Interactive HTML dashboard
│
├── sam3_finetune/                          # SAM 3 fine-tuning on building facades
│   └── README.md                          # Fine-tuning instructions
│
├── archive/                               # Archived prototype scripts (not used)
│   └── old_scripts/
│       ├── garfield/                      # Old labeling/projection prototypes
│       └── pipeline/                      # Snakefile backups
│
├── data/                                  # Input data (not tracked by git)
│   └── YOUR_PROJECT/
│       └── images/                        # Drone images
│
└── outputs/                               # Pipeline outputs (not tracked by git)
    ├── YOUR_PROJECT/
    │   ├── garfield/.../config.yml        # Stage 2a output
    │   └── garfield-gauss/.../config.yml  # Stage 2b output
    ├── ortho_projection_s005/             # Stage 3 output (features, points)
    ├── ortho_projection_cropped/          # Stage 4 output (cluster_labels.npy)
    ├── labeling_views/                    # Stage 5 output
    ├── labeling_masks_finetuned/          # Stage 6 output
    └── semantic_labels_finetuned/         # Stage 7 output
        ├── semantic_pointcloud.ply        # Final labeled point cloud
        ├── semantic_labels.json           # Per-cluster labels
        └── per_label/                     # Individual PLY per class
```

## Archived Scripts

Older prototype scripts and backups are preserved under

```text
archive/old_scripts/
```

These scripts are not used by the current Snakemake workflow but are retained for reproducibility and historical reference.

## SAM 3 Fine-Tuning

We fine-tuned SAM 3 on the HDB building facade dataset (2,679 images, 11 classes) to improve building element detection. See `sam3_finetune/README.md` for details.

**Results:**
- AP@50: 34.6% on HDB validation set
- Semantic labeling improved from 62% roof / 37% unknown (off-the-shelf) to 33% wall / 23% roof / 3% unknown (fine-tuned)

## Feature Clustering

GARField grouping features are clustered using GPU-accelerated HDBSCAN. Instead of exhaustive grid search, the current implementation uses Optuna for hyperparameter optimization.

Parameters optimized:

- min_cluster_size
- min_samples
- cluster_selection_epsilon

Outputs generated:

- cluster_labels.npy
- clustered_pointcloud.ply
- optimization_results.json

Example:

```bash
python garfield/cluster_sweep.py \
        --input-dir outputs/ortho_projection_s005 \
        --output-dir outputs/ortho_projection_cropped \
        --n-trials 50 \
        --sample-size 10000
```

The best parameter set is selected using silhouette score.

## Citation

If you use this work, please cite:

```bibtex
@misc{garfield_semantic_2026,
  title={Unsupervised Building Component Discovery via Orthographic Feature Projection from Neural Radiance Fields},
  author={Ehsan Agha Ebrahimi},
  year={2026}
}
```

## Acknowledgments

- [GARField](https://github.com/chungmin99/garfield) - Group Anything with Radiance Fields
- [SAM 3](https://github.com/facebookresearch/sam3) - Segment Anything Model 3
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) - NeRF framework
- HDB Building Facade Dataset - University of Hong Kong
