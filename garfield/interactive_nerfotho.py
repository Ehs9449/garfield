#!/usr/bin/env python3
"""
Interactive NeRFOrtho with crop support - Alternative approach.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import os

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.model_components import renderers
from nerfstudio.utils import colormaps
import mediapy as media


class InteractiveNeRFOrtho:
    """Interactive orthographic renderer with cropping."""
    
    def __init__(self, config_path: Path, data_path: Path = None, 
                 crop_center: list = None, crop_scale: list = None):
        """Initialize with config and crop settings."""
        
        # Default crop parameters
        if crop_center is None:
            crop_center = [0.02, -0.05, -0.15]
        if crop_scale is None:
            crop_scale = [1.0, 0.91, 0.19]
        
        self.crop_center = crop_center
        self.crop_scale = crop_scale
        
        # Create OrientedBox for cropping
        self.crop_obb = OrientedBox.from_params(
            pos=tuple(crop_center),
            rpy=(0.0, 0.0, 0.0),
            scale=tuple(crop_scale)
        )
        
        # Background color
        self.bg_color = torch.tensor([1.0, 1.0, 1.0])
        
        # Change working directory
        original_cwd = os.getcwd()
        
        if data_path and data_path.exists():
            os.chdir("/home/eaghae1")
            print(f"Changed working directory to: /home/eaghae1")
        
        try:
            print(f"Loading model from: {config_path}")
            config, self.pipeline, checkpoint_path, step = eval_setup(config_path, test_mode='test')
            self.device = self.pipeline.device
            
            # Move to device
            self.crop_obb = OrientedBox(
                R=self.crop_obb.R.to(self.device),
                T=self.crop_obb.T.to(self.device),
                S=self.crop_obb.S.to(self.device)
            )
            self.bg_color = self.bg_color.to(self.device)
            
            # Try to set crop in the model if it has these attributes
            if hasattr(self.pipeline.model, 'crop_obb'):
                self.pipeline.model.crop_obb = self.crop_obb
            if hasattr(self.pipeline.model, 'crop_enabled'):
                self.pipeline.model.crop_enabled = True
            if hasattr(self.pipeline.model, 'crop_bg_color'):
                self.pipeline.model.crop_bg_color = self.bg_color
            
            print("✓ Model loaded successfully")
            print(f"✓ Crop: center={crop_center}, scale={crop_scale}")
            
        finally:
            os.chdir(original_cwd)
    
    def render_orthographic(self,
                           plane_x: float, plane_y: float, plane_z: float,
                           look_x: float, look_y: float, look_z: float,
                           up_x: float, up_y: float, up_z: float,
                           width: float, height: float,
                           img_width: int, img_height: int,
                           near: float, far: float):
        """Render orthographic view."""
        
        # Setup plane
        plane_center = torch.tensor([plane_x, plane_y, plane_z], dtype=torch.float32)
        look_dir = torch.tensor([look_x, look_y, look_z], dtype=torch.float32)
        up_vec = torch.tensor([up_x, up_y, up_z], dtype=torch.float32)
        
        # Normalize
        look_dir = look_dir / torch.norm(look_dir)
        right = torch.cross(look_dir, up_vec, dim=0)
        right = right / torch.norm(right)
        up_vec = torch.cross(right, look_dir, dim=0)
        up_vec = up_vec / torch.norm(up_vec)
        
        # Create grid
        u = torch.linspace(-width/2, width/2, img_width)
        v = torch.linspace(-height/2, height/2, img_height)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        # Ray origins
        origins = (
            plane_center.unsqueeze(0).unsqueeze(0) +
            uu.unsqueeze(-1) * right.unsqueeze(0).unsqueeze(0) +
            vv.unsqueeze(-1) * up_vec.unsqueeze(0).unsqueeze(0)
        )
        
        # Parallel rays
        directions = look_dir.unsqueeze(0).unsqueeze(0).expand(img_width, img_height, -1)
        
        # Flatten
        num_rays = img_width * img_height
        origins = origins.reshape(num_rays, 3).to(self.device)
        directions = directions.reshape(num_rays, 3).to(self.device)
        
        # Add crop info to metadata
        metadata = {
            "crop_enabled": True,
            "crop_obb": self.crop_obb,
            "crop_bg_color": self.bg_color
        }
        
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=torch.ones((num_rays, 1), device=self.device) * (width * height / num_rays),
            camera_indices=torch.zeros((num_rays, 1), dtype=torch.int32, device=self.device),
            nears=torch.ones((num_rays, 1), device=self.device) * near,
            fars=torch.ones((num_rays, 1), device=self.device) * far,
            metadata=metadata,
            times=None
        )
        
        # Render with background color override
        with torch.no_grad():
            with renderers.background_color_override_context(self.bg_color):
                # First try with obb_box parameter (in case it works)
                try:
                    outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(
                        ray_bundle, obb_box=self.crop_obb
                    )
                except TypeError:
                    # Fallback: just use regular call (model might check metadata)
                    outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        
        # Apply crop masking manually if needed
        if "rgb" in outputs:
            # Check if we need to mask based on accumulation
            acc = outputs.get("accumulation", torch.ones_like(outputs["rgb"][..., :1]))
            
            # If model didn't apply crop, we can try to mask here
            # (though this is less ideal than having the model do it)
            if acc.max() > 0.99:  # If scene is fully opaque, crop might not be applied
                # Get ray positions at median depth
                if "depth" in outputs:
                    depths = outputs["depth"].unsqueeze(-1)
                    positions = origins + directions * depths
                    
                    # Check which positions are inside crop box
                    inside = self.crop_obb.within(positions)
                    inside = inside.reshape(img_height, img_width, 1)
                    
                    # Apply mask
                    outputs["rgb"] = outputs["rgb"] * inside + self.bg_color * (~inside)
                    if "accumulation" in outputs:
                        outputs["accumulation"] = outputs["accumulation"] * inside
        
        # Extract image
        rgb = outputs["rgb"].reshape(img_height, img_width, 3).cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        
        depth = None
        if "depth" in outputs:
            depth = outputs["depth"].reshape(img_height, img_width).cpu().numpy()
            
        return rgb, depth


def main():
    parser = argparse.ArgumentParser(description="NeRFOrtho with Crop")
    
    # Required
    parser.add_argument("--config", type=Path, 
                       default=Path("/home/eaghae1/outputs/unnamed/garfield/2025-11-25_083351/config.yml"))
    parser.add_argument("--output", type=Path, required=True)
    
    # Crop
    parser.add_argument("--crop-center", nargs=3, type=float,
                       default=[0.02, -0.05, -0.15])
    parser.add_argument("--crop-scale", nargs=3, type=float,
                       default=[1.0, 0.91, 0.19])
    
    # Plane
    parser.add_argument("--px", type=float, default=0.02)
    parser.add_argument("--py", type=float, default=-0.05)  
    parser.add_argument("--pz", type=float, default=2.0)
    
    # Look
    parser.add_argument("--lx", type=float, default=0.0)
    parser.add_argument("--ly", type=float, default=0.0)
    parser.add_argument("--lz", type=float, default=-1.0)
    
    # Up
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
        cx, cy, cz = args.crop_center
        if args.preset == 'n':
            args.px, args.py, args.pz = cx, cy + args.dist, cz
            args.lx, args.ly, args.lz = 0, -1, 0
            args.ux, args.uy, args.uz = 0, 0, 1
        elif args.preset == 's':
            args.px, args.py, args.pz = cx, cy - args.dist, cz
            args.lx, args.ly, args.lz = 0, 1, 0
            args.ux, args.uy, args.uz = 0, 0, 1
        elif args.preset == 'e':
            args.px, args.py, args.pz = cx + args.dist, cy, cz
            args.lx, args.ly, args.lz = -1, 0, 0
            args.ux, args.uy, args.uz = 0, 0, 1
        elif args.preset == 'w':
            args.px, args.py, args.pz = cx - args.dist, cy, cz
            args.lx, args.ly, args.lz = 1, 0, 0
            args.ux, args.uy, args.uz = 0, 0, 1
        elif args.preset == 't':
            args.px, args.py, args.pz = cx, cy, cz + args.dist
            args.lx, args.ly, args.lz = 0, 0, -1
            args.ux, args.uy, args.uz = 0, 1, 0
    
    print(f"\nRendering:")
    print(f"  Crop: {args.crop_center}, scale: {args.crop_scale}")
    print(f"  Plane: [{args.px:.2f}, {args.py:.2f}, {args.pz:.2f}]")
    print(f"  Look: [{args.lx:.2f}, {args.ly:.2f}, {args.lz:.2f}]")
    
    # Create renderer
    renderer = InteractiveNeRFOrtho(
        args.config,
        Path("/home/eaghae1/data/PFTdrone"),
        crop_center=args.crop_center,
        crop_scale=args.crop_scale
    )
    
    # Render
    print("\nRendering...")
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
