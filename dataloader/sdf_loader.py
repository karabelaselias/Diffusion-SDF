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
        augment=False
    ):
        self.use_npy = use_npy
        self.samples_per_mesh = samples_per_mesh
        self.pc_size = pc_size
        self.augment = augment
        self.epoch_multiplier = 100 if self.augment else 1
        self.grid_source = grid_source
        
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
        R = random_rotation_matrix(device=device)
        
        # Get random mirror matrix
        M = random_mirror_matrix(device=device)
        
        # Combine transformations
        transform = torch.mm(R, M)
        
        # Apply transformations
        xyz_aug, _ = apply_transformation(xyz, transform)
        pc_aug, _ = apply_transformation(pc, transform)
        
        # Note: Pure rotation and mirroring don't change SDF values for normalized shapes
        # But if you want to add scale, you'd need to adjust SDF values too:
        # scale = torch.exp(torch.randn(1, device=device) * 0.1)
        # xyz_aug = xyz_aug * scale
        # pc_aug = pc_aug * scale
        # sdf_gt = sdf_gt * scale
        
        # Ensure points stay in [-1, 1] range
        xyz_aug = torch.clamp(xyz_aug, -1.0, 1.0)
        pc_aug = torch.clamp(pc_aug, -1.0, 1.0)
        
        return xyz_aug, sdf_gt, pc_aug
        
    def __getitem__(self, idx): 
        idx %= len(self.gt_filenames)
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



    
