"""
render_perspective_grouping.py
Render REALISTIC perspective views of a trained GARField model, showing the RGB
image beside the grouping (instance) colors, to visualize grouping quality.

It renders from your actual dataset cameras (real drone viewpoints), so the views
look like real photos and line up with the frames you will annotate for Option B.

For each chosen camera it saves:
  <tag>_cam<i>_scale<s>_rgb.png       - rendered color
  <tag>_cam<i>_scale<s>_grouping.png  - grouping features colored by PCA
  <tag>_cam<i>_scale<s>_panel.png     - [Photo | Rendered RGB | Grouping] side by side

Run it once for the pretrained model and once for the fine-tuned model, with the
SAME --cam-indices and --scale, then compare the panels.

Usage:
  python render_perspective_grouping.py --config <config.yml> --outdir renders_persp \
         --tag pretrained --cam-indices 0,15,40 --scale 1.0
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from nerfstudio.utils.eval_utils import eval_setup


def instance_to_colors(feat: torch.Tensor, method: str = "pca") -> np.ndarray:
    """feat: [H, W, D] -> RGB [H, W, 3] in [0,1]."""
    H, W, D = feat.shape
    flat = feat.reshape(-1, D).float().cpu().numpy()
    if method == "pca":
        x = PCA(n_components=3).fit_transform(flat)
    elif method == "first3":
        x = flat[:, :3]
    else:
        raise ValueError(method)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x.reshape(H, W, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="trained model config.yml")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="model", help="'pretrained' or 'finetuned'")
    ap.add_argument("--cam-indices", type=str, default="0",
                    help="comma-separated dataset camera indices, e.g. 0,15,40")
    ap.add_argument("--scale", type=float, default=1.0, help="GARField grouping scale")
    ap.add_argument("--color-method", type=str, default="pca", choices=["pca", "first3"])
    ap.add_argument("--split", type=str, default="train", choices=["train", "eval"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, pipeline, _, _ = eval_setup(args.config, test_mode="test")
    pipeline.model.eval()

    # Set the grouping scale (same knob the viewer/ortho scripts use).
    if hasattr(pipeline.model, "scale_slider"):
        pipeline.model.scale_slider.value = args.scale
        print("grouping scale set to", args.scale)
    else:
        print("WARNING: model has no scale_slider; grouping scale not set")

    dm = pipeline.datamanager
    dataset = dm.train_dataset if args.split == "train" else getattr(dm, "eval_dataset", dm.train_dataset)
    cameras = dataset.cameras
    n = int(cameras.size) if hasattr(cameras, "size") else len(cameras)
    print(f"dataset has {n} cameras")

    args.outdir.mkdir(parents=True, exist_ok=True)
    idxs = [int(s) for s in args.cam_indices.split(",") if s.strip() != ""]

    for i in idxs:
        if i >= n:
            print(f"skip cam {i} (only {n} cameras)"); continue
        cam = cameras[i : i + 1].to(device)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(cam)
        if "instance" not in outputs:
            print("ERROR: 'instance' not in outputs. keys:", list(outputs.keys()))
            print("      (make sure the grouping field is trained and scale_slider exists)")
            continue

        rgb = torch.clamp(outputs["rgb"], 0, 1).cpu().numpy()
        grp = instance_to_colors(outputs["instance"], args.color_method)

        # Real photo for this camera, if available.
        gt = None
        try:
            gt = np.clip(dataset[i]["image"].cpu().numpy(), 0, 1)
        except Exception:
            pass

        base = args.outdir / f"{args.tag}_cam{i}_scale{args.scale}"
        plt.imsave(str(base) + "_rgb.png", rgb)
        plt.imsave(str(base) + "_grouping.png", grp)

        panels = [("Rendered RGB", rgb), ("Grouping (PCA)", grp)]
        if gt is not None:
            panels = [("Photo", gt)] + panels
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
        if len(panels) == 1:
            axes = [axes]
        for ax, (t, im) in zip(axes, panels):
            ax.imshow(im)
            ax.set_title(f"{t}  (cam {i}, scale {args.scale})", fontsize=11)
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(str(base) + "_panel.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved", str(base) + "_panel.png")


if __name__ == "__main__":
    main()
