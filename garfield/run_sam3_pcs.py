#!/usr/bin/env python3
"""
SCRIPT 2: Run SAM 3 PCS on rendered images
Environment: sam3

Reads rendered images, runs SAM 3 PCS with building text prompts,
saves labeled masks as numpy arrays.

Setup:
    conda create -n sam3 python=3.12
    conda activate sam3
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    cd /path/to/sam3 && pip install -e .

Usage:
    conda activate sam3
    python run_sam3_pcs.py \
        --input-dir outputs/labeling_views \
        --output-dir outputs/labeling_masks
"""

import numpy as np
from pathlib import Path
import argparse
import json
import os

# Building element prompts
BUILDING_PROMPTS = [
    "window",
    "wall",
    "roof",
    "door",
    "entrance",
    "column",
    "parapet",
    "balcony",
    "staircase",
    "HVAC equipment",
    "ground",
    "vegetation",
    "mechanical equipment",
    "facade panel",
    "canopy",
]


def load_sam3():
    """Load SAM 3 model."""
    try:
        # Try ultralytics first
        from ultralytics.models.sam import SAM3SemanticPredictor
        overrides = dict(
            conf=0.2,
            task="segment",
            mode="predict",
            model="sam3.pt",
            save=False,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        print("✓ Loaded SAM 3 via ultralytics")
        return predictor, "ultralytics"
    except ImportError:
        pass

    try:
        # Try Facebook's official sam3 repo
        from sam3.build_sam3 import build_sam3_image_model
        from sam3.semantic_predictor import SAM3SemanticPredictor as FB_SAM3Predictor

        model = build_sam3_image_model(checkpoint_path="sam3_hiera_large.pt")
        model = model.cuda()
        model.eval()
        predictor = FB_SAM3Predictor(model)
        print("✓ Loaded SAM 3 via facebook repo")
        return predictor, "facebook"
    except ImportError:
        pass

    print("ERROR: Could not load SAM 3.")
    print("Install via: pip install -e /path/to/sam3")
    print("Or install ultralytics >= 8.3.237 with sam3.pt downloaded")
    return None, None


def run_pcs_ultralytics(predictor, image_path, prompts):
    """Run PCS using ultralytics SAM3SemanticPredictor."""
    predictor.set_image(str(image_path))
    results = predictor(text=prompts)

    labeled_masks = []

    for result in results:
        if result.masks is None:
            continue

        masks = result.masks.data.cpu().numpy()

        for i in range(len(masks)):
            if result.boxes is not None and len(result.boxes) > i:
                cls_idx = int(result.boxes.cls[i])
                label = result.names.get(cls_idx, f"class_{cls_idx}")
                conf = float(result.boxes.conf[i])
            else:
                label = prompts[min(i, len(prompts) - 1)]
                conf = 1.0

            labeled_masks.append({
                'label': label,
                'mask': masks[i].astype(bool),
                'confidence': conf,
            })

    return labeled_masks


def run_pcs_facebook(predictor, image_path, prompts):
    """Run PCS using Facebook's official SAM 3 repo."""
    from PIL import Image
    import torch

    image = np.array(Image.open(image_path).convert("RGB"))

    predictor.set_image(image)

    labeled_masks = []

    for prompt in prompts:
        try:
            with torch.no_grad():
                result = predictor.predict(text=prompt)

            if result is not None:
                masks = result.get('masks', None)
                if masks is not None:
                    for mask in masks:
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        labeled_masks.append({
                            'label': prompt,
                            'mask': mask.astype(bool),
                            'confidence': 1.0,
                        })
        except Exception as e:
            print(f"    Warning: prompt '{prompt}' failed: {e}")

    return labeled_masks


def main():
    parser = argparse.ArgumentParser(description="Run SAM 3 PCS on rendered images")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/labeling_views"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/labeling_masks"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM 3
    predictor, backend = load_sam3()
    if predictor is None:
        return

    # Load view params
    with open(args.input_dir / "view_params.json") as f:
        view_params = json.load(f)

    print(f"Processing {len(view_params)} images...")
    print(f"Prompts: {BUILDING_PROMPTS}")
    print()

    all_results = []

    for i, vp in enumerate(view_params):
        img_path = args.input_dir / vp['image_file']

        if not img_path.exists():
            print(f"  [{i+1}] {vp['image_file']} - NOT FOUND, skipping")
            continue

        print(f"  [{i+1}/{len(view_params)}] {vp['image_file']}...", end=" ", flush=True)

        # Run SAM 3 PCS
        try:
            if backend == "ultralytics":
                labeled_masks = run_pcs_ultralytics(predictor, img_path, BUILDING_PROMPTS)
            else:
                labeled_masks = run_pcs_facebook(predictor, img_path, BUILDING_PROMPTS)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        print(f"found {len(labeled_masks)} masks")

        if len(labeled_masks) == 0:
            continue

        # Print detected labels
        from collections import Counter
        label_counts = Counter(m['label'] for m in labeled_masks)
        print(f"    Detected: {dict(label_counts)}")

        # Save masks as numpy arrays
        view_name = vp['view_name']
        masks_data = {
            'view_name': view_name,
            'image_file': vp['image_file'],
            'n_masks': len(labeled_masks),
            'labels': [m['label'] for m in labeled_masks],
            'confidences': [m['confidence'] for m in labeled_masks],
        }

        # Save each mask as separate NPY file
        mask_dir = args.output_dir / view_name
        mask_dir.mkdir(exist_ok=True)

        for j, m in enumerate(labeled_masks):
            mask_path = mask_dir / f"mask_{j:03d}_{m['label'].replace(' ', '_')}.npy"
            np.save(mask_path, m['mask'])

        # Save metadata
        with open(mask_dir / "masks_info.json", 'w') as f:
            json.dump(masks_data, f, indent=2)

        all_results.append(masks_data)

        # Save visualization
        try:
            from PIL import Image
            import cv2

            img = cv2.imread(str(img_path))
            vis = img.copy()

            for m in labeled_masks:
                color = np.random.randint(50, 255, 3).tolist()
                mask_overlay = np.zeros_like(vis)
                mask_overlay[m['mask']] = color
                vis = cv2.addWeighted(vis, 1.0, mask_overlay, 0.4, 0)

                # Find mask centroid for label text
                ys, xs = np.where(m['mask'])
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(vis, m['label'], (cx-30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            vis_path = args.output_dir / f"{view_name}_pcs.jpg"
            cv2.imwrite(str(vis_path), vis)
        except Exception:
            pass

    # Save summary
    with open(args.output_dir / "all_masks_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Saved masks to {args.output_dir}")
    print(f"  Each view has a subfolder with mask_XXX_label.npy files")
    print(f"  Visualizations saved as *_pcs.jpg")
    print(f"\nNext: conda activate nerfstudio3 && python match_clusters_to_masks.py")


if __name__ == "__main__":
    main()
