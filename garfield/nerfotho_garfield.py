#!/usr/bin/env python3
"""
NeRFOrtho orthographic renderer with GARFIELD instance segmentation.
Combines bounded plane method from NeRFOrtho paper with GARFIELD's instance features.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import os
from PIL import Image
from typing import Optional

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RayBundle
import mediapy as media


class NeRFOrthoGARFIELD:
    """NeRFOrtho renderer with GARFIELD instance segmentation."""
    
    def __init__(self, config_path: Path, data_path: Path = None):
        """Initialize with config."""
        
        original_cwd = os.getcwd()
        
        if data_path and data_path.exists():
            os.chdir("/home/eaghae1")
            print(f"Changed working directory to: /home/eaghae1")
        
        try:
            print(f"Loading model from: {config_path}")
            _, self.pipeline, _, _ = eval_setup(config_path, test_mode='test')
            self.device = self.pipeline.device
            self.pipeline.eval()
            print("✓ Model loaded successfully")
            
            # Check for GARFIELD components
            if hasattr(self.pipeline.model, 'grouping_field'):
                print("✓ GARFIELD model detected")
                # Check if we can set active cluster
                self.num_clusters = getattr(self.pipeline.model, 'num_clusters', 1)
                print(f"  Number of clusters: {self.num_clusters}")
            
            # Set scale slider if available
            if hasattr(self.pipeline.model, 'scale_slider'):
                self.pipeline.model.scale_slider.value = 1.0
                print(f"  Set scale to: {self.pipeline.model.scale_slider.value}")
                
        finally:
            os.chdir(original_cwd)
    
    def create_orthographic_rays(self,
                                plane_x: float, plane_y: float, plane_z: float,
                                look_x: float, look_y: float, look_z: float,
                                up_x: float, up_y: float, up_z: float,
                                width: float, height: float,
                                img_width: int, img_height: int,
                                near: float, far: float):
        """
        Create orthographic rays using NeRFOrtho bounded plane method.
        """
        
        # Setup plane center and orientation
        plane_center = torch.tensor([plane_x, plane_y, plane_z], dtype=torch.float32)
        look_dir = torch.tensor([look_x, look_y, look_z], dtype=torch.float32)
        up_vec = torch.tensor([up_x, up_y, up_z], dtype=torch.float32)
        
        # Normalize and create orthonormal basis
        look_dir = look_dir / torch.norm(look_dir)
        right = torch.cross(look_dir, up_vec, dim=0)
        right = right / torch.norm(right)
        up_vec = torch.cross(right, look_dir, dim=0)
        up_vec = up_vec / torch.norm(up_vec)
        
        # Create bounded grid on plane (NeRFOrtho approach)
        u = torch.linspace(-width/2, width/2, img_width)
        v = torch.linspace(-height/2, height/2, img_height)
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        # Ray origins distributed on the plane
        origins = (
            plane_center.unsqueeze(0).unsqueeze(0) +
            uu.unsqueeze(-1) * right.unsqueeze(0).unsqueeze(0) +
            vv.unsqueeze(-1) * up_vec.unsqueeze(0).unsqueeze(0)
        )
        
        # All rays parallel (orthographic projection)
        directions = look_dir.unsqueeze(0).unsqueeze(0).expand(img_width, img_height, -1)
        
        # Flatten for RayBundle
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
        
        return ray_bundle
    
    def render_view(self, ray_bundle: RayBundle, img_width: int, img_height: int, 
                    cluster_idx: Optional[int] = None):
        """
        Render a view with optional cluster selection for GARFIELD.
        """
        
        with torch.no_grad():
            # Set active cluster if GARFIELD model supports it
            if cluster_idx is not None and hasattr(self.pipeline.model, 'set_active_cluster'):
                self.pipeline.model.set_active_cluster(cluster_idx)
                print(f"  Rendering cluster {cluster_idx}")
            
            # Get outputs
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
            
        # Process RGB
        rgb = outputs["rgb"].reshape(img_height, img_width, 3).cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        
        # Process instance features if available
        instance = None
        if "instance" in outputs:
            instance_features = outputs["instance"].reshape(img_height, img_width, -1)
            # Convert to visualization (using magnitude for simplicity)
            instance_mag = torch.norm(instance_features, dim=-1).cpu().numpy()
            instance_mag = (instance_mag - instance_mag.min()) / (instance_mag.max() - instance_mag.min() + 1e-6)
            # Apply colormap
            import matplotlib.cm as cm
            colormap = cm.get_cmap('viridis')
            instance = colormap(instance_mag)[..., :3]
            print(f"  ✓ Rendered instance features")
        
        # Process depth if available
        depth = None
        if "depth" in outputs:
            depth = outputs["depth"].reshape(img_height, img_width).cpu().numpy()
            
        return rgb, instance, depth
    
    def render_orthographic_view(self,
                                plane_x: float, plane_y: float, plane_z: float,
                                look_x: float, look_y: float, look_z: float,
                                up_x: float, up_y: float, up_z: float,
                                width: float, height: float,
                                img_width: int, img_height: int,
                                near: float, far: float,
                                cluster_idx: Optional[int] = None):
        """
        Complete pipeline: create rays and render.
        """
        
        # Create orthographic rays
        ray_bundle = self.create_orthographic_rays(
            plane_x, plane_y, plane_z,
            look_x, look_y, look_z,
            up_x, up_y, up_z,
            width, height,
            img_width, img_height,
            near, far
        )
        
        # Render
        return self.render_view(ray_bundle, img_width, img_height, cluster_idx)


def main():
    parser = argparse.ArgumentParser(description="NeRFOrtho with GARFIELD Instance")
    
    # Required
    parser.add_argument("--config", type=Path, 
                       default=Path("/home/eaghae1/outputs/unnamed/garfield/2025-11-25_083351/config.yml"))
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory or file path")
    
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
    
    # Plane size
    parser.add_argument("--width", type=float, default=1.5)
    parser.add_argument("--height", type=float, default=0.5)
    
    # Resolution
    parser.add_argument("--res", type=int, default=1080)
    
    # Ray parameters
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=10.0)
    
    # GARFIELD specific
    parser.add_argument("--cluster", type=int, default=None,
                       help="Specific cluster to render (if GARFIELD)")
    parser.add_argument("--all-clusters", action="store_true",
                       help="Render all clusters separately")
    
    # Presets
    parser.add_argument("--preset", type=str, choices=['n', 's', 'e', 'w', 't'])
    parser.add_argument("--dist", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Apply preset if specified
    if args.preset:
        cx, cy, cz = 0.02, -0.05, -0.15  # center
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
    
    print(f"\nNeRFOrtho GARFIELD Rendering:")
    print(f"  Plane: [{args.px:.2f}, {args.py:.2f}, {args.pz:.2f}]")
    print(f"  Look: [{args.lx:.2f}, {args.ly:.2f}, {args.lz:.2f}]")
    print(f"  Size: {args.width} x {args.height}")
    
    # Initialize renderer
    renderer = NeRFOrthoGARFIELD(
        args.config,
        Path("/home/eaghae1/data/PFTdrone")
    )
    
    # Determine what to render
    clusters_to_render = []
    if args.all_clusters and hasattr(renderer, 'num_clusters'):
        clusters_to_render = list(range(renderer.num_clusters))
    elif args.cluster is not None:
        clusters_to_render = [args.cluster]
    else:
        clusters_to_render = [None]  # Render without cluster selection
    
    # Render each cluster
    for cluster_idx in clusters_to_render:
        print(f"\nRendering" + (f" cluster {cluster_idx}" if cluster_idx is not None else "") + "...")
        
        rgb, instance, depth = renderer.render_orthographic_view(
            args.px, args.py, args.pz,
            args.lx, args.ly, args.lz,
            args.ux, args.uy, args.uz,
            args.width, args.height,
            args.res, args.res,
            args.near, args.far,
            cluster_idx
        )
        
        # Determine output paths
        if cluster_idx is not None:
            suffix = f"_cluster{cluster_idx:03d}"
        else:
            suffix = ""
        
        # Save outputs
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        # RGB
        rgb_path = args.output.parent / f"{args.output.stem}{suffix}_rgb.png"
        Image.fromarray((rgb * 255).astype(np.uint8)).save(rgb_path)
        print(f"  ✓ Saved RGB: {rgb_path}")
        
        # Instance if available
        if instance is not None:
            instance_path = args.output.parent / f"{args.output.stem}{suffix}_instance.png"
            Image.fromarray((instance * 255).astype(np.uint8)).save(instance_path)
            print(f"  ✓ Saved Instance: {instance_path}")
        
        # Depth if available
        if depth is not None:
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            depth_path = args.output.parent / f"{args.output.stem}{suffix}_depth.png"
            Image.fromarray((depth_norm * 255).astype(np.uint8)).save(depth_path)
            print(f"  ✓ Saved Depth: {depth_path}")
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
