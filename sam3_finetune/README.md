# SAM 3 Fine-Tuning on HDB Building Facade Dataset

## Overview

Fine-tuning SAM 3's Promptable Category Segmentation (PCS) on the HDB building facade dataset to improve text-prompted detection of architectural elements (windows, walls, doors, roofs, etc.).

## Dataset

- **Source:** HDB (Hong Kong University) Building Facade Dataset
- **Buildings:** 30+ university buildings, Exterior and Interior
- **Classes (11):** Wall, Window, Ceiling, Door, Floor, Column, Beam, Opening, CurtainWall, Roof, Lift
- **Original format:** VIA (VGG Image Annotator) polygon annotations
- **Converted format:** COCO JSON with RLE-encoded segmentation masks

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 2,679  | 36,608      |
| Val   | 350    | 5,114       |
| Test  | 349    | 5,305       |

## Dataset Conversion

Convert HDB from VIA to COCO format:

```bash
python convert_hdb_to_coco.py
```

Edit `INPUT_DIR` and `OUTPUT_DIR` in the script. Then convert polygons to RLE on HPC:

```python
python -c "
import json
from pycocotools import mask as mask_util
for split in ['train', 'val', 'test']:
    path = f'/work/eaghae1/HDB_COCO/{split}/_annotations.coco.json'
    with open(path) as f:
        coco = json.load(f)
    img_lookup = {img['id']: img for img in coco['images']}
    converted = 0
    for ann in coco['annotations']:
        img = img_lookup[ann['image_id']]
        h, w = img['height'], img['width']
        seg = ann['segmentation']
        if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
            rle = mask_util.merge(mask_util.frPyObjects(seg, h, w))
            rle['counts'] = rle['counts'].decode('utf-8')
            ann['segmentation'] = rle
            ann['area'] = float(mask_util.area(rle))
            converted += 1
    with open(path, 'w') as f:
        json.dump(coco, f)
    print(f'{split}: converted {converted} annotations')
"
```

## Training

### Prerequisites

Install in SAM 3 conda environment:

```bash
pip install submitit fvcore hydra-core omegaconf pycocotools tensorboard scipy torchmetrics wandb --break-system-packages
pip install "numpy>=1.26,<2" "opencv-python<4.10" --break-system-packages
```

### Required Patches to SAM 3 Source Code

Three patches are needed for external fine-tuning:

**1. `sam3/perflib/fused.py`** — Fused CUDA kernel fallback for training:

```python
def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        x = linear(mat1)
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            return torch.nn.functional.relu(x)
        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            return torch.nn.functional.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")
    # ... original fused kernel code for inference ...
```

**2. DDP config** — Add `static_graph: True` to the distributed config in `hdb_building_finetune.yaml`

**3. `sam3/train/transforms/basic_for_api.py` line 212** — Graceful mask resize:

```python
if obj.segment is not None:
    try:
        resized = F.resize(obj.segment[None, None], size)
        if resized.dim() >= 2:
            obj.segment = resized.squeeze(0).squeeze(0)
        else:
            obj.segment = None
    except Exception:
        obj.segment = None
```

### Run Training

```bash
# Copy config to SAM 3 repo
cp hdb_building_finetune.yaml /work/eaghae1/sam3_repo/sam3/train/configs/

# Set partition and account
sed -i 's/account: null/account: hpc_rapid_dt/' /work/eaghae1/sam3_repo/sam3/train/configs/hdb_building_finetune.yaml
sed -i 's/partition: null/partition: gpu/' /work/eaghae1/sam3_repo/sam3/train/configs/hdb_building_finetune.yaml

# Submit training (from login node)
cd /work/eaghae1/sam3_repo
source activate /work/eaghae1/conda_envs/sam3
export HF_HOME=/work/eaghae1/.cache/huggingface
export HYDRA_FULL_ERROR=1

python sam3/train/train.py -c configs/hdb_building_finetune --use-cluster 1
```

Training takes ~6.5 hours on 4 A100 GPUs.

## Results

| Epoch | AP@50  | AP@50:95 | AR@100 |
|-------|--------|----------|--------|
| 5     | 31.9%  | 25.7%    | 33.1%  |
| 10    | 32.6%  | 25.9%    | 33.8%  |
| 15    | 33.1%  | 26.5%    | 34.6%  |
| 20    | 34.0%  | 26.9%    | 34.9%  |
| 25    | 34.3%  | 26.8%    | 35.1%  |
| 30    | 34.6%  | 26.8%    | 35.6%  |

### Semantic Labeling Improvement

| Label      | Off-the-shelf | Fine-tuned |
|------------|--------------|------------|
| Wall       | 0.0%         | **33.0%**  |
| Roof       | 62.1%        | 22.6%      |
| Vegetation | 0.0%         | **2.8%**   |
| Opening    | 0.0%         | **1.1%**   |
| Ground     | 1.2%         | 0.6%       |
| Unknown    | 36.7%        | **3.2%**   |

## Checkpoints

Saved at `/work/eaghae1/sam3_finetune_hdb/checkpoints/`:

- `checkpoint.pt` — Final (epoch 30)
- `checkpoint_5.pt` through `checkpoint_30.pt` — Every 5 epochs

## Logs

- Training stats: `/work/eaghae1/sam3_finetune_hdb/logs/train_stats.json`
- Validation stats: `/work/eaghae1/sam3_finetune_hdb/logs/val_stats.json`
- TensorBoard: `/work/eaghae1/sam3_finetune_hdb/tensorboard/`
