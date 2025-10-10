#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from . import base 

import pandas as pd 
import numpy as np
import csv, json

from tqdm import tqdm
from typing import Optional

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device  = 'cpu'
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o

def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device = 'cpu'
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)

def random_rotation(
    dtype: Optional[torch.dtype] = None, device = 'cpu'
) -> torch.Tensor:
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(1, dtype, device)[0]


def random_rotation_matrix(device='cpu'):
        """
        Create a random rotation matrix using PyTorch.
        """
        angle = torch.rand(1, device=device) * 2 * np.pi
        axis = torch.randn(3, device=device)
        axis = axis / torch.norm(axis)
        
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Rodrigues' rotation formula
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=device)
        
        I = torch.eye(3, device=device)
        R = I + sin_angle * K + (1 - cos_angle) * torch.mm(K, K)
        
        return R

def random_mirror_matrix(device='cpu'):
        """
        Create a random mirror matrix using PyTorch.
        """
        if torch.rand(1, device=device) < 0.75:
            axis = torch.randint(0, 3, (1,), device=device)
            M = torch.eye(3, device=device)
            M[axis, axis] = -1
        else:
            M = torch.eye(3, device=device)
        return M

def apply_point_cloud_dropout(pc, dropout_ratio_range=(0.01, 0.15)):
    """
    Randomly drop points from the point cloud to improve robustness.
    """
    dropout_ratio = torch.empty(1).uniform_(*dropout_ratio_range)
    n_points = pc.shape[0]
    n_keep = int(n_points * (1 - dropout_ratio))
    
    indices = torch.randperm(n_points)[:n_keep]
    pc_dropped = pc[indices]
    
    # Pad back to original size with duplicated points if needed
    if pc_dropped.shape[0] < n_points:
        pad_indices = torch.randint(0, pc_dropped.shape[0], (n_points - pc_dropped.shape[0],))
        pc_dropped = torch.cat([pc_dropped, pc_dropped[pad_indices]], dim=0)
    
    return pc_dropped

def apply_point_cloud_noise(pc, noise_std=0.01):
    """
    Add Gaussian noise to point cloud coordinates.
    """
    noise = torch.randn_like(pc) * noise_std
    pc_noisy = pc + noise
    pc_noisy = torch.clamp(pc_noisy, -1.0, 1.0)
    return pc_noisy

def apply_query_point_noise(xyz, noise_std=0.005):
    """
    Add noise only to points that will remain in bounds after noise.
    """
    noise = torch.randn_like(xyz) * noise_std
    xyz_noisy = xyz + noise
    
    # Create mask for valid points (those that stay in bounds)
    valid_mask = (xyz_noisy > -1.0).all(dim=-1) & (xyz_noisy < 1.0).all(dim=-1)
    
    # Only apply noise where valid
    xyz_augmented = torch.where(valid_mask.unsqueeze(-1), xyz_noisy, xyz)
    
    return xyz_augmented

def apply_transformation(points, transform, normals=None):
    """
    Apply a transformation matrix to points and optionally normals.
    Args:
        points: (N, 3) or (B, N, 3) tensor
        transform: (3, 3) or (B, 3, 3) tensor
        normals: Optional (N, 3) or (B, N, 3) tensor
    """
    # Handle batched or single transformations
    if points.dim() == 2:  # Single sample (N, 3)
        transformed_points = torch.mm(points, transform.T)
    else:  # Batched (B, N, 3)
        transformed_points = torch.bmm(points, transform.transpose(-2, -1))
    
    transformed_normals = None
    if normals is not None:
        # Normalize normals first
        normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
        
        if normals.dim() == 2:  # Single sample
            transformed_normals = torch.mm(normals, transform.T)
        else:  # Batched
            transformed_normals = torch.bmm(normals, transform.transpose(-2, -1))
        
        # Renormalize after transformation
        transformed_normals = F.normalize(transformed_normals, p=2, dim=-1, eps=1e-6)
    
    return transformed_points, transformed_normals

class SdfLoader(base.Dataset):

    def __init__(
        self,
        data_source, # path to points sampled around surface
        split_file, # json filepath which contains train/test classes and meshes 
        grid_source=None, # path to grid points; grid refers to sampling throughout the unit cube instead of only around the surface; necessary for preventing artifacts in empty space
        samples_per_mesh=16000,
        pc_size=1024,
        modulation_path=None, # used for third stage of training; needs to be set in config file when some modulation training had been filtered
        use_npy=True,
        augment=False,
        randomize_near_surface_ratio=True,  # New parameter
        near_surface_ratio_range=(0.5, 0.95)  # New parameter
    ):
        self.use_npy = use_npy
        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.augment = augment
        self.epoch_multiplier = 100 if self.augment else 1
        self.grid_source = grid_source
        self.current_epoch = 0
        self.surface_percentage = 0.7
        
        # Get filenames
        self.gt_filenames = self.get_instance_filenames(
            data_source, split_file, 
            gt_filename="sdf_data.npy",
            filter_modulation_path=modulation_path
        )
        
        subsample = len(self.gt_filenames)
        self.gt_filenames = self.gt_filenames[0:subsample]
        
        # Load grid filenames if needed
        if grid_source:
            self.grid_filenames = self.get_instance_filenames(
                grid_source, split_file, 
                gt_filename="grid_gt.npy", 
                filter_modulation_path=modulation_path
            )
            self.grid_filenames = self.grid_filenames[0:subsample]
            assert len(self.grid_filenames) == len(self.gt_filenames)
        
        # Load all data into memory
        self.gt_data = self._load_all_files(self.gt_filenames, "GT files")
        
        if grid_source:
            self.grid_data = self._load_all_files(self.grid_filenames, "Grid files")

        self.randomize_near_surface_ratio = randomize_near_surface_ratio
        self.near_surface_ratio_range = near_surface_ratio_range
    
    def _load_all_files(self, filenames, desc="Files"):
        """Load all files into memory with progress bar."""
        print(f"Loading all {len(filenames)} {desc} into memory...")
        data_list = []
        
        with tqdm(filenames) as pbar:
            for i, f in enumerate(pbar):
                pbar.set_description(f"{desc} loaded: {i}/{len(filenames)}")
                
                # Load file based on extension
                if f.endswith('.npy'):
                    # Optional: use mmap_mode='r' for memory-mapped arrays if files are huge
                    data = torch.from_numpy(np.load(f))
                else:
                    # Load CSV (fixing the duplicate read bug)
                    data = torch.from_numpy(pd.read_csv(f, sep=',', header=None).values)
                
                data_list.append(data)
        
        return data_list

    def augment_data(self, xyz, sdf_gt, pc):
        """
        Apply augmentation to SDF data.
        """
        device = xyz.device if torch.is_tensor(xyz) else 'cpu'
        
        # Convert to tensors if needed
        if not torch.is_tensor(xyz):
            xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
            sdf_gt = torch.tensor(sdf_gt, dtype=torch.float32, device=device)
            pc = torch.tensor(pc, dtype=torch.float32, device=device)
        
        # Get random rotation matrix
        R = random_rotation(dtype=torch.float32, device=device)
        
        # rotate
        xyz, _ = apply_transformation(xyz, R)
        pc, _ = apply_transformation(pc, R)
        
        # Get random mirror matrix
        if torch.rand(1) < 0.5:
            M = random_mirror_matrix(device=device)
            # mirror
            xyz, _ = apply_transformation(xyz, M)
            pc, _ = apply_transformation(pc, M)

        # 3. Query noise with rejection (SAFE)
        #if torch.rand(1) < 0.5:
        #    xyz = apply_query_point_noise(xyz, noise_std=0.003)

        # 4. Point cloud augmentations (SAFE for conditioning)
        if torch.rand(1) < 0.4:
            pc = apply_point_cloud_dropout(pc, dropout_ratio_range=(0.05, 0.15))

        if torch.rand(1) < 0.5:
            pc = apply_point_cloud_noise(pc, noise_std=0.005)
        
        # Note: Pure rotation and mirroring don't change SDF values for normalized shapes
        # But if you want to add scale, you'd need to adjust SDF values too:
        # scale = torch.exp(torch.randn(1, device=device) * 0.1)
        # xyz_aug = xyz_aug * scale
        # pc_aug = pc_aug * scale
        # sdf_gt = sdf_gt * scale
        
        # Ensure points stay in [-1, 1] range
        #xyz = torch.clamp(xyz, -1, 1)
        pc = torch.clamp(pc, -1, 1)
        
        return xyz, sdf_gt, pc
        
    def __getitem__(self, idx): 
        idx %= len(self.gt_filenames)
        # For float 
        #near_surface_count = int(self.samples_per_mesh*self.surface_percentage) if self.grid_source else self.samples_per_mesh
        near_surface_count = int(self.samples_per_mesh*0.7) if self.grid_source else self.samples_per_mesh
        # Sample from pre-loaded data (no file I/O here!)
        pc, sdf_xyz, sdf_gt = self.labeled_sampling(
            self.gt_data[idx], 
            near_surface_count, 
            self.pc_size, 
            load_from_path=False
        )
        
        # Add grid samples if available
        if self.grid_source is not None:
            grid_count = self.samples_per_mesh - near_surface_count
            _, grid_xyz, grid_gt = self.labeled_sampling(
                self.grid_data[idx], 
                grid_count, 
                pc_size=0, 
                load_from_path=False
            )
            # each getitem is one batch so no batch dimension, only N, 3 for xyz or N for gt 
            # for 16000 points per batch, near surface is 11200, grid is 4800
            # Concatenate near-surface and grid samples
            sdf_xyz = torch.cat((sdf_xyz, grid_xyz))
            sdf_gt = torch.cat((sdf_gt, grid_gt))
        
        # augment the dataset
        if self.augment:
            sdf_xyz, sdf_gt, pc = self.augment_data(sdf_xyz, sdf_gt, pc)
        
        data_dict = {
                        "xyz":sdf_xyz.float().squeeze(),
                        "gt_sdf":sdf_gt.float().squeeze(), 
                        "point_cloud":pc.float().squeeze(),
                    }

        return data_dict

    def __len__(self):
        return len(self.gt_filenames) * self.epoch_multiplier



    
