"""
Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from typing_extensions import TypeVar

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from rich.progress import Console

CONSOLE = Console(width=120)

import h5py
import os
import os.path as osp
import os, json, numpy as np
import torch

import numpy as np
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from garfield.img_group_model import ImgGroupModelConfig, ImgGroupModel
from garfield.garfield_pixel_sampler import GarfieldPixelSampler


@dataclass
class GarfieldDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: GarfieldDataManager)
    """The datamanager class to use."""
    img_group_model: ImgGroupModelConfig = field(default_factory=lambda: ImgGroupModelConfig())
    """The SAM model to use. This can be any other model that outputs masks..."""


TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)


class GarfieldDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """
    Tacking on grouping info to the normal VanillaDataManager.
    """

    config: GarfieldDataManagerConfig
    train_pixel_sampler: Optional[GarfieldPixelSampler] = None

    def __init__(
        self,
        config: GarfieldDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )
        self.img_group_model: ImgGroupModel = self.config.img_group_model.setup(device=self.device)

        # This is where all the group data + statistics is stored.
        # Note that this can get quite big (~10GB if 300 images, ...)
        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        self.sam_data_path = Path(cache_dir) / "sam_data.hdf5"

        self.pixel_level_keys = None
        self.scale_3d = None
        self.group_cdf = None
        self.scale_3d_statistics = None

        # try to load a lightweight index on init so next_group can lazy-load per-image npz files.
        # If a full hdf5 is present but very large, prefer lazy loading instead of populating nested tensors.
        try:
            # If there is an hdf5 sam_data file, decide whether to load it fully or lazily.
            if osp.exists(self.sam_data_path):
                # inexpensive probe: check number of images in file (if prefix exists)
                try:
                    with h5py.File(self.sam_data_path, "r") as f:
                        prefix = self.img_group_model.config.model_type
                        if prefix in f and "pixel_level_keys" in f[prefix].keys():
                            num_entries = len(list(f[prefix]["pixel_level_keys"].keys()))
                        else:
                            num_entries = 0
                except Exception:
                    num_entries = 0

                # threshold: if many images, use lazy loading to avoid OOM
                LAZY_LOAD_THRESHOLD = 200  # tune: lower if your memory is smaller
                if num_entries == 0 or num_entries > LAZY_LOAD_THRESHOLD:
                    # prefer lazy per-image loading (uses per-image npz index if available)
                    # create sam_cache dir if present and load lazy index
                    _ = self.load_sam_data_lazy()
                else:
                    # dataset small enough: attempt the existing full load path to populate nested tensors
                    _ = self.load_sam_data()
            else:
                # no sam_data.hdf5: try lazy index if it exists on disk
                _ = self.load_sam_data_lazy()
        except Exception:
            # never crash the constructor; fallback to lazy mode
            self.pixel_level_keys = None
            self.scale_3d = None
            self.group_cdf = None
            self.load_sam_data_lazy()
        # --- Prevent full-image RAM caching which causes OOM on large datasets ---
        # Many nerfstudio dataparser implementations either expose:
        #  - dataparser.cache_images (bool), or
        #  - dataparser.config.cache_images (bool)
        # Setting these to False prevents the dataparser from keeping all undistorted
        # images in RAM; images will be read/processed on demand instead.
        try:
            if hasattr(self, "dataparser"):
                # prefer explicit attribute if available
                if hasattr(self.dataparser, "cache_images"):
                    self.dataparser.cache_images = False
                # some code paths use a config object
                if hasattr(self.dataparser, "config") and hasattr(self.dataparser.config, "cache_images"):
                    self.dataparser.config.cache_images = False
        except Exception:
            # best-effort — do not break initialization if dataparser shape differs
            pass

    def load_sam_data(self) -> bool:
        """
        Loads the SAM data (masks, 3D scales, etc.) through hdf5.
        If the file doesn't exist, returns False.
        """
        prefix = self.img_group_model.config.model_type
        if not osp.exists(self.sam_data_path):
            return False

        # quick probe to avoid huge memory allocation
        try:
            with h5py.File(self.sam_data_path, "r") as f:
                if prefix not in f:
                    return False
                num_entries = len(list(f[prefix]["pixel_level_keys"].keys()))
        except Exception:
            return False

        # If very many images, prefer lazy loading to avoid OOM
        LAZY_LOAD_THRESHOLD = 200  # same threshold as in __init__; tune to your memory
        if num_entries > LAZY_LOAD_THRESHOLD:
            return False

        # If we're here, dataset is small enough — proceed with the original load behavior
        sam_data = h5py.File(self.sam_data_path, "r")
        if prefix not in sam_data.keys():
            return False

        sam_data = sam_data[prefix]

        pixel_level_keys_list, scales_3d_list, group_cdf_list = [], [], []

        num_entries = len(sam_data["pixel_level_keys"].keys())
        for i in range(num_entries):
            pixel_level_keys_list.append(
                torch.from_numpy(sam_data["pixel_level_keys"][str(i)][...])
            )
        self.pixel_level_keys = torch.nested.nested_tensor(pixel_level_keys_list)
        del pixel_level_keys_list

        for i in range(num_entries):
            scales_3d_list.append(torch.from_numpy(sam_data["scale_3d"][str(i)][...]))
            self.scale_3d = torch.nested.nested_tensor(scales_3d_list)
            self.scale_3d_statistics = torch.cat(scales_3d_list)
            del scales_3d_list

            for i in range(num_entries):
                group_cdf_list.append(torch.from_numpy(sam_data["group_cdf"][str(i)][...]))
            self.group_cdf = torch.nested.nested_tensor(group_cdf_list)
            del group_cdf_list

            return True

        return False

    def save_sam_data(self, pixel_level_keys, scale_3d, group_cdf):
        """Save the SAM grouping data to hdf5."""
        prefix = self.img_group_model.config.model_type
        # make the directory if it doesn't exist
        if not osp.exists(self.sam_data_path.parent):
            os.makedirs(self.sam_data_path.parent)

        # Append, not overwrite -- in case of multiple runs with different settings.
        with h5py.File(self.sam_data_path, "a") as f:
            for i in range(len(pixel_level_keys)):
                f.create_dataset(f"{prefix}/pixel_level_keys/{i}", data=pixel_level_keys[i])
                f.create_dataset(f"{prefix}/scale_3d/{i}", data=scale_3d[i])
                f.create_dataset(f"{prefix}/group_cdf/{i}", data=group_cdf[i])

    @staticmethod
    def create_pixel_mask_array(masks: torch.Tensor):
        """
        Create per-pixel data structure for grouping supervision.
        pixel_mask_array[x, y] = [m1, m2, ...] means that pixel (x, y) belongs to masks m1, m2, ...
        where Area(m1) < Area(m2) < ... (sorted by area).
        """
        max_masks = masks.sum(dim=0).max().item()
        image_shape = masks.shape[1:]
        pixel_mask_array = torch.full(
            (max_masks, image_shape[0], image_shape[1]), -1, dtype=torch.int
        ).to(masks.device)

        for m, mask in enumerate(masks):
            mask_clone = mask.clone()
            for i in range(max_masks):
                free = pixel_mask_array[i] == -1
                masked_area = mask_clone == 1
                right_index = free & masked_area
                if len(pixel_mask_array[i][right_index]) != 0:
                    pixel_mask_array[i][right_index] = m
                mask_clone[right_index] = 0
        pixel_mask_array = pixel_mask_array.permute(1, 2, 0)

        return pixel_mask_array

    def _calculate_3d_groups(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        point: torch.Tensor,
        max_scale: float = 2.0,
    ):
        """
        Calculate the set of groups and their 3D scale for each pixel, and the cdf.
        Returns:
            - pixel_level_keys: [H, W, max_masks]
            - scale: [num_masks, 1]
            - mask_cdf: [H, W, max_masks]
        max_masks is the maximum number of masks that was assigned to a pixel in the image,
         padded with -1s. mask_cdf does *not* include the -1s.
        Refer to the main paper for more details.
        """
        image_shape = rgb.shape[:2]
        depth = depth.view(-1, 1)  # (H*W, 1)
        point = point.view(-1, 3)  # (H*W, 3)

        def helper_return_no_masks():
            # Fail gracefully when no masks are found.
            # Create dummy data (all -1s), which will be ignored later.
            # See: `get_loss_dict_group` in `garfield_model.py`
            pixel_level_keys = torch.full(
                (image_shape[0], image_shape[1], 1), -1, dtype=torch.int
            )
            scale = torch.Tensor([0.0]).view(-1, 1)
            mask_cdf = torch.full(
                (image_shape[0], image_shape[1], 1), 1, dtype=torch.float
            )
            return (pixel_level_keys, scale, mask_cdf)

        # Calculate SAM masks
        masks = self.img_group_model((rgb.numpy() * 255).astype(np.uint8))

        # If no masks are found, return dummy data.
        if len(masks) == 0:
            return helper_return_no_masks()

        sam_mask = []
        scale = []

        # For all 2D groups,
        # 1) Denoise the masks (through eroding)
        all_masks = torch.stack(
            # [torch.from_numpy(_["segmentation"]).to(self.device) for _ in masks]
            [torch.from_numpy(_).to(self.device) for _ in masks]
        )
        # erode all masks using 3x3 kernel
        eroded_masks = torch.conv2d(
            all_masks.unsqueeze(1).float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3).to("cuda"),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

        # 2) Calculate 3D scale
        # Don't include groups with scale > max_scale (likely to be too noisy to be useful)
        for i in range(len(masks)):
            curr_mask = eroded_masks[i]
            curr_mask = curr_mask.flatten()
            curr_points = point[curr_mask]
            extent = (curr_points.std(dim=0) * 2).norm()
            if extent.item() < max_scale:
                sam_mask.append(curr_mask.reshape(image_shape))
                scale.append(extent.item())

        # If no masks are found, after postprocessing, return dummy data.
        if len(sam_mask) == 0:
            return helper_return_no_masks()

        sam_mask = torch.stack(sam_mask)  # (num_masks, H, W)
        scale = torch.Tensor(scale).view(-1, 1).to(self.device)  # (num_masks, 1)

        # Calculate "pixel level keys", which is a 2D array of shape (H, W, max_masks)
        # Each pixel has a list of group indices that it belongs to, in order of increasing scale.
        pixel_level_keys = self.create_pixel_mask_array(
            sam_mask
        ).long()  # (H, W, max_masks)

        # Calculate group sampling CDF, to bias sampling towards smaller groups
        # Be careful to not include -1s in the CDF (padding, or unlabeled pixels)
        # Inversely proportional to log of mask size.
        mask_inds, counts = torch.unique(pixel_level_keys, return_counts=True)
        mask_sorted = torch.argsort(counts)
        mask_inds, counts = mask_inds[mask_sorted], counts[mask_sorted]
        counts[0] = 0  # don't include -1
        probs = counts / counts.sum()  # [-1, 0, ...]
        mask_probs = torch.gather(probs, 0, pixel_level_keys.reshape(-1) + 1).view(
            pixel_level_keys.shape
        )
        mask_log_probs = torch.log(mask_probs)
        never_masked = mask_log_probs.isinf()
        mask_log_probs[never_masked] = 0.0
        mask_log_probs = mask_log_probs / (
            mask_log_probs.sum(dim=-1, keepdim=True) + 1e-6
        )
        mask_cdf = torch.cumsum(mask_log_probs, dim=-1)
        mask_cdf[never_masked] = 1.0

        return (pixel_level_keys.cpu(), scale.cpu(), mask_cdf.cpu())

    def next_group(self, ray_bundle: RayBundle, batch: Dict[str, Any]):
        """Returns the rays' mask and 3D scales for grouping (memory-safe lazy loads)."""
        indices = batch["indices"].long().detach().cpu()
        npximg = self.train_pixel_sampler.num_rays_per_image
        img_ind = indices[:, 0]
        x_ind = indices[:, 1]
        y_ind = indices[:, 2]

        mask_id = torch.zeros((indices.shape[0],), device=self.device)
        scale = torch.zeros((indices.shape[0],), device=self.device)

        random_vec_sampling = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)
        random_vec_densify = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)

        # helper to get per-image arrays (either from in-memory nested tensor or from disk)
        def _get_image_sam_arrays(i):
            # If fully loaded in memory (old behaviour)
            if (
                getattr(self, "pixel_level_keys", None) is not None
                and getattr(self, "scale_3d", None) is not None
                and getattr(self, "group_cdf", None) is not None
            ):
                per_pixel_keys = self.pixel_level_keys[i]  # nested tensor leaf
                per_scale = self.scale_3d[i]
                per_cdf = self.group_cdf[i]
                return per_pixel_keys, per_scale, per_cdf

            # Else try lazy load from disk using datamanager helper
            try:
                arrs = self.load_sam_for_image(int(i))
            except Exception:
                arrs = None
            if arrs is None or arrs[0] is None:
                # fallback: single dummy mask (very small, safe)
                per_pixel_keys = torch.full((1, 1, 1), -1, dtype=torch.int)
                per_scale = torch.zeros((1, 1))
                per_cdf = torch.ones((1, 1))
                return per_pixel_keys, per_scale, per_cdf

            pixel_level_keys_np, scale_3d_np, group_cdf_np = arrs
            per_pixel_keys = torch.from_numpy(pixel_level_keys_np).long()
            per_scale = torch.from_numpy(scale_3d_np).float()
            per_cdf = torch.from_numpy(group_cdf_np).float()
            return per_pixel_keys, per_scale, per_cdf

        # process rays in chunks of npximg
        for i in range(0, indices.shape[0], npximg):
            img_idx = int(img_ind[i].item())
            per_pixel_index, per_scale_tensor, per_group_cdf = _get_image_sam_arrays(img_idx)
            per_pixel_index_slice = per_pixel_index[x_ind[i : i + npximg], y_ind[i : i + npximg]]
            random_index = torch.sum(
                random_vec_sampling.view(-1, 1)
                > per_group_cdf[x_ind[i : i + npximg], y_ind[i : i + npximg]],
                dim=-1,
            )

            # handle single-mask case
            if per_pixel_index_slice.dim() == 1 or per_pixel_index_slice.shape[-1] == 1:
                per_pixel_mask = per_pixel_index_slice.squeeze()
                per_pixel_mask_ = per_pixel_mask.clone()
            else:
                per_pixel_mask = torch.gather(
                    per_pixel_index_slice, 1, random_index.unsqueeze(-1)
                ).squeeze()
                per_pixel_mask_ = torch.gather(
                    per_pixel_index_slice,
                    1,
                    torch.max(random_index.unsqueeze(-1) - 1, torch.tensor([0], dtype=torch.long)),
                ).squeeze()

            mask_id[i : i + npximg] = per_pixel_mask.to(self.device)

            # interval scale supervision
            # gather scales for chosen masks
            curr_scale = per_scale_tensor[per_pixel_mask.long()].squeeze()
            if curr_scale.dim() == 0:
                curr_scale = curr_scale.unsqueeze(0)
            curr_scale = curr_scale.clone()
            # densify where random_index == 0
            mask0 = (random_index == 0)
            if mask0.any():
                curr_scale[mask0] = (
                    per_scale_tensor[per_pixel_mask.long()][mask0]
                    * random_vec_densify[mask0]
                ).squeeze()
            # handle other indices if multiple masks per pixel
            max_cdf_len = per_group_cdf.shape[-1] if per_group_cdf.ndim == 3 else 1
            for j in range(1, max_cdf_len):
                mask_j = (random_index == j)
                if mask_j.sum() == 0:
                    continue
                val = (
                    per_scale_tensor[per_pixel_mask_.long()][mask_j]
                    + (
                        per_scale_tensor[per_pixel_mask.long()][mask_j]
                        - per_scale_tensor[per_pixel_mask_.long()][mask_j]
                    )
                    * random_vec_densify[mask_j]
                )
                curr_scale[mask_j] = val.squeeze()

            scale[i : i + npximg] = curr_scale.to(self.device)

        batch["mask_id"] = mask_id
        batch["scale"] = scale
        batch["nPxImg"] = npximg
        ray_bundle.metadata["scale"] = batch["scale"]



    def save_sam_data_single(self, idx, pixel_level_keys, scale_3d, group_cdf):
        """
        Save per-image grouping output to disk as compressed npz.
        Call: self.save_sam_data_single(i, pixel_level_keys, scale_3d, group_cdf)
        """
        import os, json, numpy as _np

        # ensure cache dir
        cache_dir = getattr(self, "sam_cache_dir", None)
        if cache_dir is None:
            data_root = getattr(self.dataparser, "data", "data")
            cache_dir = os.path.join(str(data_root), "sam_cache")
            self.sam_cache_dir = cache_dir
        os.makedirs(self.sam_cache_dir, exist_ok=True)

        # build output file name
        out_name = os.path.join(self.sam_cache_dir, f"sam_{int(idx):06d}.npz")

        # Convert tensors to CPU numpy (single-image conversion)
        def _to_np(x):
            try:
                return x.detach().cpu().numpy()
            except Exception:
                return _np.array(x)

        _np.savez_compressed(
            out_name,
            pixel_level_keys=_to_np(pixel_level_keys),
            scale_3d=_to_np(scale_3d),
            group_cdf=_to_np(group_cdf),
        )

        # update index (append entry)
        idx_file = os.path.join(self.sam_cache_dir, "index.json")
        try:
            if os.path.exists(idx_file):
                with open(idx_file, "r") as f:
                    idxd = json.load(f)
            else:
                idxd = {}
            idxd[str(int(idx))] = os.path.basename(out_name)
            with open(idx_file, "w") as f:
                json.dump(idxd, f)
        except Exception:
            # best-effort index writing; don't crash training
            pass

    def save_sam_index(self):
        """
        Ensure index exists; no-op if save_sam_data_single already wrote index.json entries.
        """
        cache_dir = getattr(self, "sam_cache_dir", None)
        if cache_dir is None:
            return
        idx_file = os.path.join(cache_dir, "index.json")
        if not os.path.exists(idx_file):
            # create empty index mapping if none
            with open(idx_file, "w") as f:
                json.dump({}, f)

    def load_sam_data_lazy(self):
        """
        Load only the index (list of per-image files). This function does not load the npz mask arrays.
        Subsequent code should read per-image npz files on demand.
        """
        cache_dir = getattr(self, "sam_cache_dir", None)
        if cache_dir is None:
            data_root = getattr(self.dataparser, "data", "data")
            cache_dir = os.path.join(str(data_root), "sam_cache")
            self.sam_cache_dir = cache_dir
        idx_file = os.path.join(self.sam_cache_dir, "index.json")
        if not os.path.exists(idx_file):
            # no precomputed masks available
            self.sam_index = {}
            return False
        try:
            with open(idx_file, "r") as f:
                self.sam_index = json.load(f)
            return True
        except Exception:
            self.sam_index = {}
            return False

    def load_sam_for_image(self, idx):
        """
        datamanager method: returns pixel_level_keys, scale_3d, group_cdf for a single image
        index by loading its npz file.
        """
        import os, numpy as _np, json as _json
        cache_dir = getattr(self, "sam_cache_dir", None)
        if cache_dir is None:
            data_root = getattr(self.dataparser, "data", "data")
            cache_dir = os.path.join(str(data_root), "sam_cache")
            self.sam_cache_dir = cache_dir
        idx_file = os.path.join(self.sam_cache_dir, "index.json")
        if not os.path.exists(idx_file):
            return None, None, None
        try:
            with open(idx_file, "r") as f:
                idxd = _json.load(f)
        except Exception:
            return None, None, None
        fn = idxd.get(str(int(idx)))
        if fn is None:
            return None, None, None
        data = _np.load(os.path.join(self.sam_cache_dir, fn), allow_pickle=True)
        return data["pixel_level_keys"], data["scale_3d"], data["group_cdf"]

