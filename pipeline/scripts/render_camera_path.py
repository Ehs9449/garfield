#!/usr/bin/env python3
"""Render existing Nerfstudio VIEWER camera_path.json with GARField-Gauss.

- Retains original camera poses and per-view vertical FoV from viewer export.
- Applies existing GARField crop (or --no-crop); crop does not inpaint occlusion.
- Saves directly matcher-compatible view_params.json (3x4 c2w; fx, fy, cx, cy).
- Nothing from an existing pipeline/checkpoint is modified.

The camera_path array is resampled, *not* interpolated again.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--camera-path', type=Path, required=True)
    p.add_argument('--gauss-config-txt', type=Path, required=True)
    p.add_argument('--pipeline-config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--frames', type=int, default=96, help='Number of evenly sampled exported camera frames, 0=all')
    p.add_argument('--width', type=int, default=1600, help='Width (aspect ratio follows camera-path JSON)')
    p.add_argument('--fov-deg', type=float, default=None, help='Optional vertical FoV override; by default use viewer-exported FoV')
    p.add_argument('--no-crop', action='store_true', help='Render full model without cropping vegetation')
    p.add_argument('--dry-run', action='store_true', help='Validate/view plan without GPU or writing images')
    p.add_argument('--overwrite', action='store_true', help='Explicitly replace outputs inside the dedicated viewer-path folder')
    return p.parse_args()


def parse_viewer_json(json_path: Path, frames: int, width: int, override_fov=None):
    doc = json.loads(json_path.read_text())
    cameras = doc.get('camera_path')
    if not isinstance(cameras, list) or not cameras:
        raise ValueError('Viewer JSON has no nonempty camera_path array')
    if doc.get('camera_type', 'perspective') != 'perspective':
        raise ValueError('Only perspective camera paths are supported')
    total = len(cameras)
    if frames < 0:
        raise ValueError('Number of frames cannot be negative')
    indices = np.arange(total) if frames == 0 or frames >= total else np.unique(np.rint(np.linspace(0, total-1, frames)).astype(int))
    default_fov = float(doc.get('default_fov', 75))
    default_aspect = float(doc.get('render_width', 1920)) / float(doc.get('render_height', 1080))
    if width < 320 or width > 4096:
        raise ValueError('Render width must be within 320..4096')
    records=[]
    for order, index in enumerate(indices):
        item = cameras[int(index)]
        matrix = np.asarray(item['camera_to_world'], dtype=np.float64)
        if matrix.size != 16:
            raise ValueError(f'Camera frame {index}: expected 16 transform floats, got {matrix.size}')
        matrix = matrix.reshape((4,4))  # Nerfstudio JSON uses row-major flattened 4x4
        if not np.isfinite(matrix).all():
            raise ValueError(f'Camera frame {index}: non-finite transform values')
        if not np.allclose(matrix[3], [0,0,0,1], atol=1e-5):
            raise ValueError(f'Camera frame {index}: bad homogeneous matrix row')
        rot = matrix[:3,:3]
        if not np.allclose(rot.T @ rot, np.eye(3), atol=0.025) or np.linalg.det(rot) < 0.95:
            raise ValueError(f'Camera frame {index}: invalid camera rotation')
        fov = float(override_fov if override_fov is not None else item.get('fov', default_fov))
        aspect = float(item.get('aspect', default_aspect))
        if not 10 < fov < 140 or not .2 < aspect < 5:
            raise ValueError(f'Camera frame {index}: invalid fov/aspect')
        height = int(round(width / aspect))
        # In Nerfstudio's viewer perspective path, FoV is VERTICAL degrees.
        focal = float(height / (2 * math.tan(math.radians(fov)/2)))
        name = f'viewer_{int(index):04d}'
        records.append({
            'view_name': name, 'image_file': f'{name}.jpg',
            'c2w': matrix[:3, :4].tolist(),
            'fx': focal, 'fy': focal, 'cx': width/2.0, 'cy': height/2.0,
            'img_w': width, 'img_h': height,
            'type': 'viewer_camera_path', 'face': 'custom',
            'azimuth': None, 'elevation': float(matrix[2,3]),
            'camera_position': matrix[:3,3].tolist(),
            'look_at': None, 'target_patch_span': 0.0,
            'original_frame_index': int(index),
            'vertical_fov_degrees': fov,
        })
    return doc, records


def crop_settings(config_path: Path):
    import yaml
    cfg=yaml.safe_load(config_path.read_text())
    center=np.asarray(cfg['garfield']['crop_center'], dtype=np.float32)
    scale=np.asarray(cfg['garfield']['crop_scale'], dtype=np.float32)
    if center.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError('Invalid garfield.crop_center/crop_scale')
    return center, scale


def write_contact_sheet(images_dir: Path, dest: Path, ncol=4):
    from PIL import Image, ImageDraw
    images = sorted(images_dir.glob('viewer_*.jpg'))
    if not images: return
    thumb_w, thumb_h, lab_h = 360, 225, 32
    nrows = math.ceil(len(images)/ncol)
    sheet = Image.new('RGB', (ncol*thumb_w, nrows*(thumb_h+lab_h)), (32,34,38))
    for i, path in enumerate(images):
        with Image.open(path) as im:
            thumbnail=im.convert('RGB')
            thumbnail.thumbnail((thumb_w-8,thumb_h-8), Image.Resampling.LANCZOS)
        x=(i % ncol)*thumb_w + (thumb_w-thumbnail.width)//2
        y=(i // ncol)*(thumb_h+lab_h)+(thumb_h-thumbnail.height)//2
        sheet.paste(thumbnail,(x,y))
        ImageDraw.Draw(sheet).text(((i%ncol)*thumb_w+10,(i//ncol)*(thumb_h+lab_h)+thumb_h+5),path.stem,fill=(245,245,245))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=92)
    print(f'Contact sheet: {dest}')


def main():
    a=get_args()
    doc, records = parse_viewer_json(a.camera_path, a.frames, a.width, a.fov_deg)
    print(f'Viewer path: {len(doc["camera_path"])} frames / {len(doc.get("keyframes", []))} keyframes')
    print(f'Selected {len(records)} views, {records[0]["img_w"]}x{records[0]["img_h"]}, vertical FoV {records[0]["vertical_fov_degrees"]:.1f} deg')
    print('First sampled indices:', [r['original_frame_index'] for r in records[:10]])
    print('Last sampled indices:', [r['original_frame_index'] for r in records[-4:]])
    if not a.no_crop:
        center,scale = crop_settings(a.pipeline_config)
        print('Model crop enabled: center', center.tolist(), 'size', scale.tolist())
    else:
        print('No crop: entire model visible')
    if a.dry_run:
        print('Dry-run successful: no GPU loaded, no file writes')
        return
    out=a.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f'{out} already has files; pass --overwrite if intentional')
    pointer_text=a.gauss_config_txt.read_text().strip()
    cfg_path=Path(pointer_text).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Gaussian config not found: {cfg_path}')

    import torch
    import cv2
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras

    print('Loading GARField-Gauss:', cfg_path)
    _, pipeline, _, _ = eval_setup(cfg_path, test_mode='test')
    pipeline.eval()
    if not a.no_crop:
        model=pipeline.model
        model.crop_enabled=True
        tc=torch.as_tensor(center, device=pipeline.device)
        ts=torch.as_tensor(scale, device=pipeline.device)
        model.crop_min=tc-ts/2
        model.crop_max=tc+ts/2
        model.crop_bg_color=torch.tensor([1.,1.,1.], device=pipeline.device)
        print('Crop applied to model for RGB renders')
    out.mkdir(parents=True, exist_ok=True)
    if a.overwrite:
        for p in out.glob('viewer_*.jpg'):
            p.unlink()  # Only this experiment's dedicated view files
    shutil.copyfile(a.camera_path, out/'camera_path_source.json')
    # Preserve camera and crop settings for reproducibility.
    settings={'camera_path':str(a.camera_path), 'gauss_config':str(cfg_path),
              'source_frames':len(doc['camera_path']), 'selected_frames':len(records),
              'crop_enabled':not a.no_crop, 'crop_center':center.tolist() if not a.no_crop else None,
              'crop_scale':scale.tolist() if not a.no_crop else None,
              'render_width':a.width, 'fov_override':a.fov_deg}
    (out/'render_settings.json').write_text(json.dumps(settings,indent=2))

    for num, rec in enumerate(records, 1):
        c2w = torch.as_tensor([rec['c2w']], dtype=torch.float32, device=pipeline.device)
        def field(value): return torch.tensor([[value]], dtype=torch.float32, device=pipeline.device)
        cam=Cameras(camera_to_worlds=c2w,
                    fx=field(rec['fx']), fy=field(rec['fy']),
                    cx=field(rec['cx']), cy=field(rec['cy']),
                    width=torch.tensor([[rec['img_w']]],device=pipeline.device),
                    height=torch.tensor([[rec['img_h']]],device=pipeline.device))
        with torch.inference_mode():
            render=pipeline.model.get_outputs_for_camera(cam)
        rgb=render['rgb'].detach().cpu().numpy()
        if rgb.shape != (rec['img_h'],rec['img_w'],3):
            raise RuntimeError(f'{rec["view_name"]}: bad image shape: {rgb.shape}')
        rgb=np.uint8(np.clip(rgb,0,1)*255)
        filename=out/rec['image_file']
        ok=cv2.imwrite(str(filename),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,96])
        if not ok:
            raise IOError(f'Could not write {filename}')
        print(f'[{num}/{len(records)}] {filename.name} (viewer frame {rec["original_frame_index"]})',flush=True)
    (out/'view_params.json').write_text(json.dumps(records,indent=2))
    with (out/'view_manifest.csv').open('w', newline='') as f:
        w=csv.writer(f)
        w.writerow(['view_name','image_file','original_frame_index','img_w','img_h','vertical_fov_degrees',
                    'eye_x','eye_y','eye_z'])
        for rec in records:
            w.writerow([rec['view_name'], rec['image_file'],rec['original_frame_index'],rec['img_w'],rec['img_h'],
                        rec['vertical_fov_degrees'],*rec['camera_position']])
    contact_sheet = out.parent/'closeview_rgb_contact_sheet.jpg'
    write_contact_sheet(out, contact_sheet)
    print(f'DONE: {len(records)} RGB renders and matching camera metadata: {out}')
    print('No SAM3 inference or clustering was run. All existing pipeline outputs unchanged.')


if __name__ == '__main__':
    main()
