from pathlib import Path
import argparse

import cv2
import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup

from garfield.img_group_model import ImgGroupModel, ImgGroupModelConfig


def build_color_list(n=256):
    """Generate visually distinct RGB colors."""
    colors = [(0, 0, 0)]

    for i in range(1, n):
        hue = (i * 137.508) % 360
        sat = 0.75 + (i % 3) * 0.08
        val = 0.85 + (i % 2) * 0.10

        h = hue / 60.0
        c = val * sat
        x = c * (1 - abs(h % 2 - 1))
        m = val - c

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors.append(
            (
                int((r + m) * 255),
                int((g + m) * 255),
                int((b + m) * 255),
            )
        )

    return colors


COLORS = build_color_list()


def create_mask_visualizations(image_rgb, masks, alpha=0.45):
    """
    Create:
      - colored masks on black background
      - colored masks overlaid on the original image
    """

    h, w = image_rgb.shape[:2]

    masks_only = np.zeros((h, w, 3), dtype=np.uint8)
    overlay = image_rgb.copy().astype(np.float64)

    for i, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        color = COLORS[(i + 1) % len(COLORS)]

        for channel in range(3):
            masks_only[:, :, channel][mask_bool] = color[channel]

            overlay[:, :, channel][mask_bool] = (
                (1 - alpha) * image_rgb[:, :, channel][mask_bool]
                + alpha * color[channel]
            )

    return overlay.astype(np.uint8), masks_only


def checkpoint_raw_sam2_masks(
    dataset_path: Path,
    output_dir: Path,
    image_name=None,
    all_images=False,
):
    """
    Checkpoint 01:
    Visualize raw automatic masks produced by the fine-tuned SAM2.1
    model before GARField post-processing/grouping.
    """

    images_dir = dataset_path / "images"

    image_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    if image_name is not None:
        image_paths = [images_dir / image_name]
    elif all_images:
        image_paths = image_files
    else:
        image_paths = [image_files[0]]

    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(image_path)

    print("Loading fine-tuned SAM2.1 model...")

    config = ImgGroupModelConfig(
        model_type="sam2",
        sam_model_type="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam_model_ckpt="/home/eaghae1/sam2/checkpoints/sam2.1_hiera_large.pt",
        sam_finetuned_ckpt="/home/eaghae1/sam2/checkpoints/checkpoint.pt",
        device="cuda",
    )

    model = ImgGroupModel(
        config=config,
        device="cuda",
    )

    print(f"Images to process: {len(image_paths)}")

    for image_index, image_path in enumerate(image_paths, start=1):

        print()
        print(
            f"[{image_index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        masks = model(image_rgb)

        checkpoint_dir = (
            output_dir
            / "checkpoint_01_raw_sam2_masks"
            / image_path.stem
        )

        individual_dir = (
            checkpoint_dir / "individual_masks"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        individual_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(checkpoint_dir / "original.jpg"),
            image_bgr,
        )

        for i, mask in enumerate(masks):

            mask_u8 = (
                mask.astype(np.uint8) * 255
            )

            cv2.imwrite(
                str(
                    individual_dir
                    / f"mask_{i:04d}.png"
                ),
                mask_u8,
            )

        overlay_rgb, masks_only_rgb = (
            create_mask_visualizations(
                image_rgb,
                masks,
            )
        )

        overlay_bgr = cv2.cvtColor(
            overlay_rgb,
            cv2.COLOR_RGB2BGR,
        )

        masks_only_bgr = cv2.cvtColor(
            masks_only_rgb,
            cv2.COLOR_RGB2BGR,
        )

        cv2.imwrite(
            str(checkpoint_dir / "overlay.jpg"),
            overlay_bgr,
        )

        cv2.imwrite(
            str(checkpoint_dir / "masks_only.png"),
            masks_only_bgr,
        )

        print(
            f"  masks={len(masks)} -> "
            f"{checkpoint_dir}"
        )

    print()
    print("Checkpoint 01 complete.")



def get_garfield_instance_features(
    config_path: Path,
    camera_index: int = 0,
    scale: float = 0.05,
):
    """
    Load the trained GARField model and render the raw 256-D instance
    feature map for one training camera.
    """

    print(f"Loading GARField model from: {config_path}")

    _, pipeline, _, _ = eval_setup(
        config_path,
        test_mode="test",
    )

    if hasattr(pipeline.model, "scale_slider"):
        pipeline.model.scale_slider.value = scale

    dataset = pipeline.datamanager.train_dataset

    camera = dataset.cameras[camera_index : camera_index + 1].to(
        pipeline.device
    )

    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera)

    if "instance" not in outputs:
        raise KeyError(
            f"'instance' not found in GARField outputs: {list(outputs.keys())}"
        )

    features = outputs["instance"].detach().cpu().numpy()

    image_path = Path(
        dataset.image_filenames[camera_index]
    )

    print(f"Camera index: {camera_index}")
    print(f"Image: {image_path.name}")
    print(f"Feature shape: {features.shape}")

    return image_path, features



def save_feature_vector_grid(
    image_path: Path,
    features: np.ndarray,
    output_dir: Path,
    grid_step: int = 128,
):
    """
    Overlay sampled feature-vector IDs on the RGB image and save the
    corresponding full 256-D vectors to CSV.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h_img, w_img = image_bgr.shape[:2]
    h_feat, w_feat, dim = features.shape

    rows = []
    point_id = 1

    for y in range(grid_step // 2, h_feat, grid_step):
        for x in range(grid_step // 2, w_feat, grid_step):

            img_x = int(round(x * w_img / w_feat))
            img_y = int(round(y * h_img / h_feat))

            vector = features[y, x]

            label = f"P{point_id:02d}"

            cv2.circle(image_bgr, (img_x, img_y), 5, (255, 255, 255), -1)
            cv2.putText(
                image_bgr,
                label,
                (img_x + 7, img_y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            rows.append(
                [label, x, y, img_x, img_y] + vector.tolist()
            )

            point_id += 1

    overlay_path = output_dir / "feature_vector_grid.jpg"
    cv2.imwrite(str(overlay_path), image_bgr)

    csv_path = output_dir / "feature_vectors.csv"

    header = (
        ["id", "feature_x", "feature_y", "image_x", "image_y"]
        + [f"f{i:03d}" for i in range(dim)]
    )

    np.savetxt(
        csv_path,
        np.array([r[1:] for r in rows], dtype=np.float32),
        delimiter=",",
        header=",".join(header[1:]),
        comments="",
    )

    print(f"Saved vector-ID overlay: {overlay_path}")
    print(f"Saved full 256-D vectors: {csv_path}")



def save_feature_vector_grid(
    image_path: Path,
    features: np.ndarray,
    output_dir: Path,
    grid_step: int = 128,
):
    """
    Overlay sampled point IDs on the RGB image and save the corresponding
    full 256-D GARField vectors.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h_img, w_img = image_bgr.shape[:2]
    h_feat, w_feat, dim = features.shape

    records = []
    point_id = 1

    for y in range(grid_step // 2, h_feat, grid_step):
        for x in range(grid_step // 2, w_feat, grid_step):

            img_x = int(round(x * w_img / w_feat))
            img_y = int(round(y * h_img / h_feat))

            vector = features[y, x].astype(np.float32)
            label = f"P{point_id:02d}"

            cv2.circle(
                image_bgr,
                (img_x, img_y),
                5,
                (255, 255, 255),
                -1,
            )

            cv2.putText(
                image_bgr,
                label,
                (img_x + 7, img_y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            records.append((label, x, y, img_x, img_y, vector))
            point_id += 1

    overlay_path = output_dir / "feature_vector_grid.jpg"
    cv2.imwrite(str(overlay_path), image_bgr)

    csv_path = output_dir / "feature_vectors.csv"

    with open(csv_path, "w") as f:
        header = (
            ["id", "feature_x", "feature_y", "image_x", "image_y"]
            + [f"f{i:03d}" for i in range(dim)]
        )
        f.write(",".join(header) + "\n")

        for label, x, y, img_x, img_y, vector in records:
            values = [
                label,
                str(x),
                str(y),
                str(img_x),
                str(img_y),
            ] + [f"{v:.8f}" for v in vector]

            f.write(",".join(values) + "\n")

    print(f"Saved overlay: {overlay_path}")
    print(f"Saved vectors: {csv_path}")

def main():

    parser = argparse.ArgumentParser(
        description="Visualize intermediate GARField pipeline checkpoints."
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
    )

    parser.add_argument(
        "--image-name",
        default=None,
        help="Process one specific image.",
    )

    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Process every image in the dataset.",
    )

    parser.add_argument(
        "--output-root",
        default="outputs",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    output_dir = (
        Path(args.output_root)
        / args.dataset_name
        / "diagnostics"
    )

    checkpoint_raw_sam2_masks(
        dataset_path=dataset_path,
        output_dir=output_dir,
        image_name=args.image_name,
        all_images=args.all_images,
    )


if __name__ == "__main__":
    main()
