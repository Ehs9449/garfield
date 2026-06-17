"""
Convert HDB dataset from VIA format to COCO JSON format for SAM 3 fine-tuning.
Just paste in PyCharm and run. No command line arguments needed.
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image

# ============================================================
# CHANGE THESE TWO PATHS TO MATCH YOUR SETUP
# ============================================================
INPUT_DIR = r"D:\Project\Dataset\HDB"
OUTPUT_DIR = r"D:\Project\Dataset\HDB_COCO"
# ============================================================

CLASS_MAP = {
    "Beam": 1,
    "Ceiling": 2,
    "Column": 3,
    "CurtainWall": 4,
    "Door": 5,
    "Floor": 6,
    "Lift": 7,
    "Opening": 8,
    "Roof": 9,
    "Wall": 10,
    "Window": 11,
}

CATEGORIES = [
    {"id": v, "name": k, "supercategory": "building"}
    for k, v in sorted(CLASS_MAP.items(), key=lambda x: x[1])
]


def find_via_jsons(hdb_root, split):
    results = []
    hdb_root = Path(hdb_root)
    for building_dir in sorted(hdb_root.iterdir()):
        if not building_dir.is_dir():
            continue
        if building_dir.name.startswith("_") or building_dir.name.startswith("."):
            continue
        for location in ["Exterior", "Interior"]:
            split_dir = building_dir / location / split
            if not split_dir.exists():
                continue
            json_files = [
                f for f in split_dir.iterdir()
                if f.suffix == ".json"
                and f.name != "_class_map.json"
                and not f.name.startswith("_")
            ]
            for jf in json_files:
                results.append({
                    "json_path": jf,
                    "image_dir": split_dir,
                    "building": building_dir.name,
                    "location": location,
                })
    return results


def parse_via_json(json_path, image_dir):
    with open(json_path, "r") as f:
        data = json.load(f)
    entries = []
    for key, value in data.items():
        filename = value.get("filename", "")
        regions = value.get("regions", [])
        img_path = image_dir / filename
        if not img_path.exists():
            found = False
            for f in image_dir.iterdir():
                if f.name.lower() == filename.lower():
                    img_path = f
                    filename = f.name
                    found = True
                    break
            if not found:
                print(f"  WARNING: Image not found: {img_path}")
                continue
        entries.append({
            "filename": filename,
            "image_path": img_path,
            "regions": regions,
        })
    return entries


def polygon_to_bbox(points_x, points_y):
    x_min = min(points_x)
    y_min = min(points_y)
    x_max = max(points_x)
    y_max = max(points_y)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def polygon_area(points_x, points_y):
    n = len(points_x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points_x[i] * points_y[j]
        area -= points_x[j] * points_y[i]
    return abs(area) / 2.0


def convert_split(hdb_root, output_dir, split):
    print(f"\n{'=' * 60}")
    print(f"Converting split: {split}")
    print(f"{'=' * 60}")

    via_sources = find_via_jsons(hdb_root, split)
    print(f"Found {len(via_sources)} JSON files")

    split_out = Path(output_dir) / split
    split_out.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    image_id = 0
    ann_id = 0
    skipped_classes = set()
    copied_images = 0

    for source in via_sources:
        json_path = source["json_path"]
        image_dir = source["image_dir"]
        building = source["building"]
        location = source["location"]

        print(f"\n  Processing: {building}/{location}/{split}")
        print(f"    JSON: {json_path.name}")

        entries = parse_via_json(json_path, image_dir)
        print(f"    Images: {len(entries)}")

        for entry in entries:
            image_id += 1

            # Create unique filename to avoid collisions across buildings
            new_filename = f"{building}_{location}_{entry['filename']}"
            new_filename = new_filename.replace(" ", "_")

            # Get image dimensions
            try:
                with Image.open(entry["image_path"]) as img:
                    width, height = img.size
            except Exception as e:
                print(f"    WARNING: Cannot read image {entry['image_path']}: {e}")
                continue

            # Copy image to output
            dst_path = split_out / new_filename
            if not dst_path.exists():
                shutil.copy2(entry["image_path"], dst_path)
            copied_images += 1

            coco["images"].append({
                "id": image_id,
                "file_name": new_filename,
                "height": height,
                "width": width,
            })

            for region in entry["regions"]:
                shape = region.get("shape_attributes", {})
                attrs = region.get("region_attributes", {})

                class_name = attrs.get("ClassName", "")
                if class_name not in CLASS_MAP:
                    skipped_classes.add(class_name)
                    continue

                category_id = CLASS_MAP[class_name]

                if shape.get("name") != "polygon":
                    continue

                points_x = shape.get("all_points_x", [])
                points_y = shape.get("all_points_y", [])

                if len(points_x) < 3:
                    continue

                ann_id += 1

                segmentation = []
                for x, y in zip(points_x, points_y):
                    segmentation.extend([float(x), float(y)])

                bbox = polygon_to_bbox(points_x, points_y)
                area = polygon_area(points_x, points_y)

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "bbox": bbox,
                    "area": float(area),
                    "iscrowd": 0,
                })

    # Save COCO JSON
    coco_path = split_out / "_annotations.coco.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)

    # Summary
    print(f"\n  Summary for {split}:")
    print(f"    Images: {len(coco['images'])}")
    print(f"    Annotations: {len(coco['annotations'])}")
    print(f"    Copied images: {copied_images}")
    if skipped_classes:
        print(f"    Skipped classes: {skipped_classes}")

    class_counts = {}
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        cat_name = [c["name"] for c in CATEGORIES if c["id"] == cat_id][0]
        class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

    print(f"    Per-class counts:")
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"      {name:20s}: {count}")

    print(f"    Saved: {coco_path}")


# ============================================================
# MAIN - Just run this file, no arguments needed
# ============================================================
print(f"Input:  {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Classes: {list(CLASS_MAP.keys())}")

for split in ["train", "val", "test"]:
    convert_split(INPUT_DIR, OUTPUT_DIR, split)

print(f"\n{'=' * 60}")
print(f"DONE! Dataset ready at: {OUTPUT_DIR}")
print(f"{'=' * 60}")
