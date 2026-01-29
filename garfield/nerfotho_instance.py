#!/usr/bin/env python3
"""
Interactive NeRFOrtho renderer with GARFIELD instance segmentation colors.
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class InteractiveNeRFOrthoInstance:
    """Orthographic renderer with GARFIELD instance colors."""
    
    def __init__(self, config_path: Path, data_path: Path = None):
        """Initialize with config."""
        
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
            
            # Set scale for GARFIELD instance features
            if hasattr(self.pipeline.model, 'scale_slider'):
                # Set to a specific scale value for consistent segmentation
                self.pipeline.model.scale_slider.value = 1.0  # Adjust this value as needed
                print(f"✓ Set GARFIELD scale to: {self.pipeline.model.scale_slider.value}")
            
        finally:
            os.chdir(original_cwd)
    
    def instance_features_to_colors(self, instance_features, method='pca'):
        """
        Convert high-dimensional instance features to RGB colors.
        
        Args:
            instance_features: Tensor of shape [H, W, D] where D is feature dimension
            method: 'pca', 'first3', or 'colormap'
        
        Returns:
            RGB image of shape [H, W, 3]
        """
        H, W, D = instance_features.shape
        features_flat = instance_features.reshape(-1, D)
        
        if method == 'pca':
            # Use PCA to reduce to 3 dimensions
            pca = PCA(n_components=3)
            features_3d = pca.fit_transform(features_flat.cpu().numpy())
            # Normalize to [0, 1]
            features_3d = (features_3d - features_3d.min()) / (features_3d.max() - features_3d.min() + 1e-6)
            colors = features_3d.reshape(H, W, 3)
            
        elif method == 'first3':
            # Just take first 3 dimensions and normalize
            features_3d = features_flat[:, :3].cpu().numpy()
            features_3d = (features_3d - features_3d.min()) / (features_3d.max() - features_3d.min() + 1e-6)
            colors = features_3d.reshape(H, W, 3)
            
        elif method == 'colormap':
            # Use magnitude of features with a colormap
            feature_magnitude = torch.norm(features_flat, dim=1).cpu().numpy()
            feature_magnitude = (feature_magnitude - feature_magnitude.min()) / (feature_magnitude.max() - feature_magnitude.min() + 1e-6)
            colormap = cm.get_cmap('viridis')
            colors = colormap(feature_magnitude.reshape(H, W))[..., :3]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return colors
    
    def render_orthographic(self,
                           plane_x: float, plane_y: float, plane_z: float,
                           look_x: float, look_y: float, look_z: float,
                           up_x: float, up_y: float, up_z: float,
                           width: float, height: float,
                           img_width: int, img_height: int,
                           near: float, far: float,
                           scale: float = 1.0,
                           color_method: str = 'pca'):
        """Render orthographic view with instance colors."""
        
        # Set the scale for GARFIELD
        if hasattr(self.pipeline.model, 'scale_slider'):
            self.pipeline.model.scale_slider.value = scale
        
        # Setup plane
        plane_center = torch.tensor([plane_x, plane_y, plane_z], dtype=torch.float32)
        look_dir = torch.tensor([look_x, look_y, look_z], dtype=torch.float32)
        up_vec = torch.tensor([up_x, up_y, up_z], dtype=torch.float32)
        
        # Normalize and orthogonalize
        look_dir = look_dir / torch.norm(look_dir)
        right = torch.cross(look_dir, up_vec, dim=0)
        right = right / torch.norm(right)
        up_vec = torch.cross(right, look_dir, dim=0)
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
        
        # Render
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        
        # Get RGB for reference
        rgb = outputs["rgb"].reshape(img_height, img_width, 3).cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        
        # Get instance features and convert to colors
        instance_colors = None
        if "instance" in outputs:
            instance_features = outputs["instance"].reshape(img_height, img_width, -1)
            instance_colors = self.instance_features_to_colors(instance_features, method=color_method)
            print(f"✓ Rendered instance features with shape: {instance_features.shape}")
        else:
            print("⚠ Warning: 'instance' not found in outputs. Available keys:", outputs.keys())
        
        depth = None
        if "depth" in outputs:
            depth = outputs["depth"].reshape(img_height, img_width).cpu().numpy()
        
        return rgb, instance_colors, depth


def main():
    parser = argparse.ArgumentParser(description="NeRFOrtho with GARFIELD Instance Colors")
    
    # Required
    parser.add_argument("--config", type=Path, 
                       default=Path("/home/eaghae1/outputs/unnamed/garfield/2025-11-25_083351/config.yml"))
    parser.add_argument("--output", type=Path, required=True)
    
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
    
    # GARFIELD specific
    parser.add_argument("--scale", type=float, default=1.0,
                       help="GARFIELD grouping scale (0-2)")
    parser.add_argument("--color-method", type=str, default='pca',
                       choices=['pca', 'first3', 'colormap'],
                       help="Method to convert features to colors")
    
    # Presets
    parser.add_argument("--preset", type=str, choices=['n', 's', 'e', 'w', 't'])
    parser.add_argument("--dist", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Apply preset
    if args.preset:
        cx, cy, cz = 0.02, -0.05, -0.15  # crop center
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
    
    print(f"\nRendering GARFIELD Instance Orthographic View:")
    print(f"  Plane: [{args.px:.2f}, {args.py:.2f}, {args.pz:.2f}]")
    print(f"  Scale: {args.scale}")
    print(f"  Color method: {args.color_method}")
    
    # Create renderer
    renderer = InteractiveNeRFOrthoInstance(
        args.config,
        Path("/home/eaghae1/data/PFTdrone")
    )
    
    # Render
    print("\nRendering...")
    rgb, instance_colors, depth = renderer.render_orthographic(
        args.px, args.py, args.pz,
        args.lx, args.ly, args.lz,
        args.ux, args.uy, args.uz,
        args.width, args.height,
        args.res, args.res,
        args.near, args.far,
        args.scale, args.color_method
    )
    
    # Save outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save RGB
    rgb_path = args.output.parent / f"{args.output.stem}_rgb.png"
    media.write_image(rgb_path, rgb)
    print(f"✓ Saved RGB: {rgb_path}")
    
    # Save instance colors if available
    if instance_colors is not None:
        instance_path = args.output.parent / f"{args.output.stem}_instance.png"
        media.write_image(instance_path, instance_colors)
        print(f"✓ Saved Instance: {instance_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
