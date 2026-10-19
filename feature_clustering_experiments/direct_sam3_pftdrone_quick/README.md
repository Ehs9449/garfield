# PFTdrone Direct SAM3 Quick Test

This package is for the existing PFTdrone assets from the old
`~/garfield-semantic` run.

Default inputs:

```text
~/garfield-semantic/outputs/ortho_projection_s005/points.npy
~/garfield-semantic/outputs/labeling_views/view_params.json
~/garfield-semantic/outputs/labeling_masks_finetuned/*_masks.npz
```

It does NOT load:

```text
avg_features.npy
cluster_labels.npy
K400/HDBSCAN labels
```

Install from `~/garfield`:

```bash
mkdir -p feature_clustering_experiments/direct_sam3_pftdrone_quick
unzip -o pftdrone_direct_sam3_quick.zip \
  -d feature_clustering_experiments/direct_sam3_pftdrone_quick
```

Then:

```bash
conda activate nerfstudio3
cd ~/garfield

bash \
feature_clustering_experiments/direct_sam3_pftdrone_quick/run_pftdrone_quick.sh
```

Output:

```text
~/garfield/outputs/PFTdrone_direct_sam3_quick/
├── semantic_pointcloud.ply
├── semantic_labels_direct.npy
├── per_class/
├── class_counts.csv
├── per_view_summary.csv
└── summary.json
```

The runner also writes:

```text
~/garfield/outputs/PFTdrone_direct_sam3_quick.time.txt
```

The semantic point cloud and `per_class/` files can then be inspected before
running the B-rep / IDF stage.

Important fairness note: this is a QUICK semantic ablation. It reuses XYZ
points and rendered views from an earlier GARField-era run. It therefore tests
the direct SAM3 semantic assignment itself, not a completely GARField-free
end-to-end runtime.
