#!/usr/bin/env python3
"""
Interactive NeRFOrtho with GARFIELD instance segmentation.
Using correct scale parameter for fine-grained segmentation.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import os
from collections import defaultdict

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RayBundle
import mediapy as media
from sklearn.decomposition import PCA
import matplotlib.cm as cm


class InteractiveNeRFOrthoGARFIELD:
    """Interactive orthographic renderer with GARFIELD instance colors."""
    
    def __init__(self, config_path: Path, data_path: Path = None):
        """Initialize with config."""
        
        original_cwd = os.getcwd()
        
        if data_path and data_path.exists():
            os.chdir("/home/eaghae1")
            print(f"Changed working directory to: /home/eaghae1")
        
        try:
            print(f"Loading config from: {config_path}")
            
            _, self.pipeline, checkpoint_path, _ = eval_setup(
                config_path, 
                test_mode='test'
            )
            print(f"Loaded checkpoint: {checkpoint_path}")
            
            self.device = self.pipeline.device
            self.pipeline.eval()
            print("✓ Model loaded successfully")
            
            # Get chunk size from config
            self.num_rays_per_chunk = getattr(self.pipeline.model.config, 'eval_num_rays_per_chunk', 4096)
            print(f"  Rays per chunk: {self.num_rays_per_chunk}")
            
            # Check for GARFIELD model
            self.has_instance = False
            if hasattr(self.pipeline.model, 'grouping_field'):
                print("✓ GARFIELD model detected")
                
                if self.pipeline.model.grouping_field.quantile_transformer is not None:
                    print("  ✓ Quantile transformer is initialized")
                    self.has_instance = True
                
                if hasattr(self.pipeline.model, 'scale_slider'):
                    # Set to fine-grained scale like in viewer
                    self.pipeline.model.scale_slider.value = 0.008
                    print(f"  Scale slider set to: {self.pipeline.model.scale_slider.value}")
                    
        finally:
            os.chdir(original_cwd)
    
    def render_orthographic(self,
                           plane_x: float, plane_y: float, plane_z: float,
                           look_x: float, look_y: float, look_z: float,
                           up_x: float, up_y: float, up_z: float,
                           width: float, height: float,
                           img_width: int, img_height: int,
                           near: float, far: float,
                           scale: float = 0.008):  # Changed default to 0.008
        """Render orthographic view with instance colors."""
        
        # Update scale if GARFIELD
        if hasattr(self.pipeline.model, 'scale_slider'):
            self.pipeline.model.scale_slider.value = scale
            print(f"  Using scale: {scale}")
        
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
        
        # Create bounded grid (NeRFOrtho approach)
        u = torch.linspace(-width/2, width/2, img_width)
        v = torch.linspace(-height/2, height/2, img_height)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        # Ray origins on the plane
        origins = (
            plane_center.unsqueeze(0).unsqueeze(0) +
            uu.unsqueeze(-1) * right.unsqueeze(0).unsqueeze(0) +
            vv.unsqueeze(-1) * up_vec.unsqueeze(0).unsqueeze(0)
        )
        
        # All rays parallel (orthographic)
        directions = look_dir.unsqueeze(0).unsqueeze(0).expand(img_width, img_height, -1)
        
        # Flatten
        num_rays = img_width * img_height
        origins_flat = origins.reshape(num_rays, 3)
        directions_flat = directions.reshape(num_rays, 3)
        
        # Process in chunks
        outputs_lists = defaultdict(list)
        
        print(f"  Processing {num_rays} rays in chunks of {self.num_rays_per_chunk}...")
        
        for i in range(0, num_rays, self.num_rays_per_chunk):
            start_idx = i
            end_idx = min(i + self.num_rays_per_chunk, num_rays)
            
            chunk_origins = origins_flat[start_idx:end_idx].to(self.device)
            chunk_directions = directions_flat[start_idx:end_idx].to(self.device)
            chunk_size = end_idx - start_idx
            
            ray_bundle = RayBundle(
                origins=chunk_origins,
                directions=chunk_directions,
                pixel_area=torch.ones((chunk_size, 1), device=self.device) * (width * height / num_rays),
                camera_indices=torch.zeros((chunk_size, 1), dtype=torch.int32, device=self.device),
                nears=torch.ones((chunk_size, 1), device=self.device) * near,
                fars=torch.ones((chunk_size, 1), device=self.device) * far,
                metadata={},
                times=None
            )
            
            with torch.no_grad():
                outputs = self.pipeline.model.forward(ray_bundle=ray_bundle)
            
            for output_name, output in outputs.items():
                if isinstance(output, torch.Tensor):
                    outputs_lists[output_name].append(output.cpu())
            
            if (i // self.num_rays_per_chunk) % 10 == 0:
                print(f"    Processed {i}/{num_rays} rays...")
        
        # Combine outputs
        combined_outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            try:
                combined_outputs[output_name] = torch.cat(outputs_list)
            except:
                pass
        
        print(f"  Available outputs: {list(combined_outputs.keys())}")
        
        # Process RGB
        rgb = None
        if "rgb" in combined_outputs:
            rgb = combined_outputs["rgb"].reshape(img_height, img_width, 3).numpy()
            rgb = np.clip(rgb, 0, 1)
        
        # Process instance with PCA for better visualization
        instance_colors = None
        if "instance" in combined_outputs:
            instance_features = combined_outputs["instance"].reshape(img_height, img_width, -1).numpy()
            print(f"  Instance features shape: {instance_features.shape}")
            
            # Use PCA to reduce to 3 dimensions for RGB
            H, W, D = instance_features.shape
            features_flat = instance_features.reshape(-1, D)
            
            # Apply PCA
            pca = PCA(n_components=3)
            features_3d = pca.fit_transform(features_flat)
            
            # Normalize each channel to [0, 1]
            for i in range(3):
                channel = features_3d[:, i]
                if channel.max() > channel.min():
                    features_3d[:, i] = (channel - channel.min()) / (channel.max() - channel.min())
            
            instance_colors = features_3d.reshape(H, W, 3)
            print(f"  ✓ Generated instance visualization using PCA")
            print(f"  Feature variance explained: {pca.explained_variance_ratio_}")
        
        # Process depth
        depth = None
        if "depth" in combined_outputs:
            depth = combined_outputs["depth"].reshape(img_height, img_width).numpy()
        
        return rgb, instance_colors, depth


def main():
    parser = argparse.ArgumentParser(description="Interactive NeRFOrtho with GARFIELD Instance")
    
    parser.add_argument("--config", type=Path, 
                       default=Path("/home/eaghae1/outputs/unnamed/garfield/2025-11-14_214010/config.yml"),
                       help="Path to config.yml")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path")
    
    # Plane parameters
    parser.add_argument("--px", type=float, default=0.02)
    parser.add_argument("--py", type=float, default=-0.05)  
    parser.add_argument("--pz", type=float, default=2.0)
    parser.add_argument("--lx", type=float, default=0.0)
    parser.add_argument("--ly", type=float, default=0.0)
    parser.add_argument("--lz", type=float, default=-1.0)
    parser.add_argument("--ux", type=float, default=0.0)
    parser.add_argument("--uy", type=float, default=1.0)
    parser.add_argument("--uz", type=float, default=0.0)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=10.0)
    
    # GARFIELD scale - now defaults to 0.008 for fine segmentation
    parser.add_argument("--scale", type=float, default=0.008,
                       help="GARFIELD grouping scale (default 0.008 for fine segmentation)")
    
    args = parser.parse_args()
    
    print(f"\nRendering Interactive NeRFOrtho:")
    print(f"  Scale: {args.scale} (fine-grained segmentation)")
    
    # Create renderer
    renderer = InteractiveNeRFOrthoGARFIELD(
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
        args.scale
    )
    
    # Save outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    if rgb is not None:
        rgb_path = args.output.parent / f"{args.output.stem}_rgb.png"
        media.write_image(rgb_path, rgb)
        print(f"\n✓ Saved RGB: {rgb_path}")
    
    if instance_colors is not None:
        instance_path = args.output.parent / f"{args.output.stem}_instance_s{args.scale:.3f}.png"
        media.write_image(instance_path, instance_colors)
        print(f"✓ Saved Instance: {instance_path}")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
