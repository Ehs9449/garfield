import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-score", type=float, default=0.3)
    args = parser.parse_args()

    views_dir = Path(args.views_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed diagnostic colors for semantic classes.
    colors = {
        "window": (0, 255, 255),
        "wall": (255, 165, 0),
        "roof": (255, 0, 255),
        "door": (255, 0, 0),
        "column": (0, 255, 0),
        "beam": (255, 255, 0),
        "ceiling": (180, 180, 255),
        "floor": (150, 100, 50),
        "curtain wall": (0, 150, 255),
        "opening": (255, 100, 100),
        "vegetation": (0, 150, 0),
        "ground": (150, 150, 150),
        "sky": (100, 180, 255),
        "staircase": (180, 100, 255),
        "railing": (255, 200, 100),
    }

    image_files = sorted(views_dir.glob("*.jpg"))

    for image_path in image_files:
        npz_path = masks_dir / f"{image_path.stem}_masks.npz"

        if not npz_path.exists():
            print(f"Missing masks: {image_path.name}")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        overlay = image.astype(np.float32).copy()

        data = np.load(npz_path, allow_pickle=True)
        detected = []

        labels = sorted(
            {
                key[:-6]
                for key in data.files
                if key.endswith("_masks")
            }
        )

        for label in labels:
            masks = data[f"{label}_masks"]
            scores = data[f"{label}_scores"]

            color = np.array(colors.get(label, (255, 255, 255)),
                             dtype=np.float32)

            for mask, score in zip(masks, scores):
                if float(score) < args.min_score:
                    continue

                if mask.shape != image.shape[:2]:
                    print(
                        f"Skipping wrong mask shape {mask.shape} "
                        f"for {image_path.name} / {label}"
                    )
                    continue

                mask = mask.astype(bool)

                if not mask.any():
                    continue

                overlay[mask] = (
                    0.55 * overlay[mask] +
                    0.45 * color
                )

                detected.append((label, float(score)))

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Put original and SAM3 overlay side-by-side.
        h, w = image.shape[:2]
        canvas = Image.new("RGB", (w * 2, h + 45), "white")
        canvas.paste(Image.fromarray(image), (0, 45))
        canvas.paste(Image.fromarray(overlay), (w, 45))

        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), "Original", fill="black")
        draw.text((w + 5, 5), "SAM3 semantic masks", fill="black")
        draw.text((5, 22), image_path.name, fill="black")

        output_path = output_dir / f"{image_path.stem}_sam3_overlay.jpg"
        canvas.save(output_path, quality=95)

        counts = {}
        for label, score in detected:
            counts[label] = counts.get(label, 0) + 1

        summary = ", ".join(
            f"{label}:{count}"
            for label, count in sorted(counts.items())
        )

        print(f"{image_path.name}: {summary}")

    print(f"\nSaved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
