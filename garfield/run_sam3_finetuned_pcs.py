import os
import argparse
import yaml
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Paths
parser = argparse.ArgumentParser()
parser.add_argument("--config-yaml", default="/work/eaghae1/pipeline_config.yaml",
                    help="Pipeline config YAML copied from local machine")
args = parser.parse_args()

with open(args.config_yaml, "r") as f:
    cfg = yaml.safe_load(f)

bpe_path = cfg["sam3"]["bpe_path"]
checkpoint_path = cfg["sam3"]["checkpoint"]
confidence_threshold = float(cfg["sam3"]["confidence_threshold"])
prompts = cfg["sam3"]["prompts"]

print("Building model...")
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
model = build_sam3_image_model(bpe_path=bpe_path)

# Load fine-tuned weights
print("Loading fine-tuned checkpoint...")
ckpt = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

# Get view images
view_files = sorted([f for f in Path(views_dir).iterdir() if f.suffix == ".jpg"])
print(f"Found {len(view_files)} views")

# Process each view
for img_path in view_files:
    print(f"\nProcessing: {img_path.name}")
    img = np.array(Image.open(img_path).convert("RGB"))
    
    view_results = {}
    for prompt in prompts:
        state = processor.set_image(img)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        
        masks = state.get("masks", None)
        scores = state.get("scores", None)
        boxes = state.get("boxes", None)
        
        if masks is not None and len(masks) > 0:
            n_masks = masks.shape[0]
            masks_np = masks.squeeze(1).cpu().numpy().astype(bool)
            scores_np = scores.cpu().float().numpy()
            boxes_np = boxes.cpu().float().numpy()
            
            view_results[prompt] = {
                "masks": masks_np,
                "scores": scores_np,
                "boxes": boxes_np,
                "count": n_masks,
            }
            print(f"  {prompt}: {n_masks} detections, best score: {scores_np.max():.3f}")
        else:
            print(f"  {prompt}: 0 detections")
    
    # Save results
    save_path = Path(output_dir) / f"{img_path.stem}_masks.npz"
    save_dict = {}
    for label, data in view_results.items():
        save_dict[f"{label}_masks"] = data["masks"]
        save_dict[f"{label}_scores"] = data["scores"]
        save_dict[f"{label}_boxes"] = data["boxes"]
    np.savez_compressed(save_path, **save_dict)
    print(f"  Saved: {save_path}")

print("\nDone! Results saved to:", output_dir)
