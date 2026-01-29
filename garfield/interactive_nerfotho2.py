#!/usr/bin/env python3
"""
Interactive NeRFOrtho renderer with crop support for GARFIELD models.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import os

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils import colormaps
import mediapy as media


class InteractiveNeRFOrtho:
    """Interactive orthographic renderer with cropping."""
    
    def __init__(self, config_path: Path, data_path: Path = None, 
                 crop_center: list = None, crop_scale: list = None):
        """Initialize with config and enable cropping."""
        
        # Default crop parameters from your JSON
        if crop_center is None:
            crop_center = [0.02, -0.05, -0.15]
        if crop_scale is None:
            crop_scale = [1.0, 0.91, 0.19]
        
        # Change working directory for data path
        original_cwd = os.getcwd()
        
        if data_path and data_path.exists():
            os.chdir("/home/eaghae1")
            print(f"Changed working directory to: /home/eaghae1")
        
        try:
            print(f"Loading model from: {config_path}")
            config, self.pipeline, checkpoint_path, step = eval_setup(config_path, test_mode='test')
            self.device = self.pipeline.device
            print("✓ Model loaded successfully")
            
            # ENABLE CROPPING IN THE MODEL
            print(f"Enabling crop with center={crop_center}, scale={crop_scale}")
            
            # Calculate crop bounds
            crop_center_tensor = torch.tensor(crop_center, device=self.device)
            crop_scale_tensor = torch.tensor(crop_scale, device=self.device)
            crop_min = crop_center_tensor - crop_scale_tensor / 2
            crop_max = crop_center_tensor + crop_scale_tensor / 2
            
            # Set crop parameters in the model
            self.pipeline.model.crop_enabled = True
            self.pipeline.model.crop_min = crop_min
            self.pipeline.model.crop_max = crop_max
            self.pipeline.model.crop_bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            
            print(f"✓ Crop enabled: min={crop_min.tolist()}, max={crop_max.tolist()}")
            
        finally:
            os.chdir(original_cwd)
            print(f"Restored working directory to: {original_cwd}")
    
    def render_orthographic(self,
                           plane_x: float, plane_y: float, plane_z: float,
                           look_x: float, look_y: float, look_z: float,
                           up_x: float, up_y: float, up_z: float,
                           width: float, height: float,
                           img_width: int, img_height: int,
                           near: float, far: float):
        """Render orthographic view with cropping applied."""
        
        # Setup plane
        plane_center = torch.tensor([plane_x, plane_y, plane_z], dtype=torch.float32)
        look_dir = torch.tensor([look_x, look_y, look_z], dtype=torch.float32)
        up_vec = torch.tensor([up_x, up_y, up_z], dtype=torch.float32)
        
        # Normalize and orthogonalize
        look_dir = look_dir / torch.norm(look_dir)
        right = torch.cross(look_dir, up_vec)
        right = right / torch.norm(right)
        up_vec = torch.cross(right, look_dir)
        up_vec = up_vec / torch.norm(up_vec)
        
        # Create grid
        u = torch.linspace(-width/2, width/2, img_width)
        v = torch.linspace(-height/2, height/2, img_height)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        # Ray origins on plane
        origins = (
            plane_center.unsqueeze(0).unsqueeze(0) +
            uu.unsqueeze(-1) * right.unsqueeze(0).unsqueeze(0) +
            vv.unsqueeze(-1) * up_vec.unsqueeze(0).unsqueeze(0)
        )
        
        # All rays parallel
        directions = look_dir.unsqueeze(0).unsqueeze(0).expand(img_width, img_height, -1)
        
        # Flatten
        num_rays = img_width * img_height
        origins = origins.reshape(num_rays, 3).to(self.device)
        directions = directions.reshape(num_rays, 3).to(self.device)
        
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=torch.ones((num_rays, 1), device=self.device) * (width * height / num_rays),
            camera_indices=torch.zeros((num_rays, 1), dtype=torch.int32, device=self.device),
            nears=torch.ones((num_rays, 1), device=self.device) * near,
            fars=torch.ones((num_rays, 1), device=self.device) * far,
            metadata={},
            times=None
        )
        
        # Render (cropping will be applied in the model)
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        
        # Extract image
        rgb = outputs["rgb"].reshape(img_height, img_width, 3).cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        
        depth = None
        if "depth" in outputs:
            depth = outputs["depth"].reshape(img_height, img_width).cpu().numpy()
            
        return rgb, depth


def main():
    parser = argparse.ArgumentParser(
        description="Interactive NeRFOrtho with Cropping"
    )
    
    # Required
    parser.add_argument("--config", type=Path, 
                       default=Path("/home/eaghae1/outputs/unnamed/garfield/2025-11-25_083351/config.yml"))
    parser.add_argument("--output", type=Path, required=True)
    
    # Crop parameters (from your JSON)
    parser.add_argument("--crop-center", nargs=3, type=float,
                       default=[0.02, -0.05, -0.15],
                       help="Crop center [x y z]")
    parser.add_argument("--crop-scale", nargs=3, type=float,
                       default=[1.0, 0.91, 0.19],
                       help="Crop scale [x y z]")
    
    # Plane position
    parser.add_argument("--px", type=float, default=0.02)
    parser.add_argument("--py", type=float, default=-0.05)  
    parser.add_argument("--pz", type=float, default=2.0)
    
    # Look direction
    parser.add_argument("--lx", type=float, default=0.0)
    parser.add_argument("--ly", type=float, default=0.0)
    parser.add_argument("--lz", type=float, default=-1.0)
    
    # Up vector
    parser.add_argument("--ux", type=float, default=0.0)
    parser.add_argument("--uy", type=float, default=1.0)
    parser.add_argument("--uz", type=float, default=0.0)
    
    # Size
    parser.add_argument("--width", type=float, default=1.5)
    parser.add_argument("--height", type=float, default=0.5)
    
    # Resolution
    parser.add_argument("--res", type=int, default=1080)
    
    # Clipping
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=10.0)
    
    # Presets
    parser.add_argument("--preset", type=str, choices=['n', 's', 'e', 'w', 't'])
    parser.add_argument("--dist", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Apply preset
    if args.preset:
        cx, cy, cz = args.crop_center  # Use crop center for presets
        if args.preset == 'n':  # north
            args.px, args.py, args.pz = cx, cy + args.dist, cz
            args.lx, args.ly, args.lz = 0, -1, 0
            args.ux, args.uy, args.uz = 0, 0, 1
        elif args.preset == 's':  # south
            args.px, args.py, args.pz = cx, cy - args.dist, cz
            args.lx, args.ly, args.lz = 0, 1, 0
            args.ux, args.uy, args.uz = 0, 0, 1
        elif args.preset == 'e':  # east
            args.px, args.py, args.pz = cx + args.dist, cy, cz
            args.lx, args.ly, args.lz = -1, 0, 0
            args.ux, args.uy, args.uz = 0, 0, 1
        elif args.preset == 'w':  # west
            args.px, args.py, args.pz = cx - args.dist, cy, cz
            args.lx, args.ly, args.lz = 1, 0, 0
            args.ux, args.uy, args.uz = 0, 0, 1
        elif args.preset == 't':  # top
            args.px, args.py, args.pz = cx, cy, cz + args.dist
            args.lx, args.ly, args.lz = 0, 0, -1
            args.ux, args.uy, args.uz = 0, 1, 0
    
    print(f"\nRendering with crop enabled:")
    print(f"  Crop center: {args.crop_center}")
    print(f"  Crop scale: {args.crop_scale}")
    print(f"  Plane: [{args.px:.2f}, {args.py:.2f}, {args.pz:.2f}]")
    print(f"  Look: [{args.lx:.2f}, {args.ly:.2f}, {args.lz:.2f}]")
    print(f"  Size: {args.width} x {args.height}")
    print(f"  Resolution: {args.res}x{args.res}")
    
    # Create renderer with crop enabled
    renderer = InteractiveNeRFOrtho(
        args.config, 
        Path("/home/eaghae1/data/PFTdrone"),
        crop_center=args.crop_center,
        crop_scale=args.crop_scale
    )
    
    # Render
    print("\nRendering image...")
    rgb, depth = renderer.render_orthographic(
        args.px, args.py, args.pz,
        args.lx, args.ly, args.lz,
        args.ux, args.uy, args.uz,
        args.width, args.height,
        args.res, args.res,
        args.near, args.far
    )
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    media.write_image(args.output, rgb)
    print(f"✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
