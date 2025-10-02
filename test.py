#!/usr/bin/env python3

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from torch_scatter import scatter_add
import matplotlib.pyplot as plt

import os
import json, csv
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import trimesh
import warnings

# add paths in model/__init__.py for new models
from models import * 
from utils import mesh, evaluate
from utils.reconstruct import *
from diff_utils.helpers import * 
#from metrics.evaluation_metrics import *#compute_all_metrics
#from metrics import evaluation_metrics

from dataloader.pc_loader import PCloader

def fix_compiled_state_dict(state_dict):
    """Fix state dict keys from compiled models (remove _orig_mod wrapper)"""
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove _orig_mod from the keys
        new_key = key.replace("._orig_mod", "")
        new_state_dict[new_key] = value
    return new_state_dict

@torch.no_grad()
def test_plane_occupancy(args, specs):
    """
    Diagnostic test to visualize and analyze plane feature sparsity
    Run this separately from main training/testing pipeline
    """
    
    # Load model and data
    test_split = json.load(open(specs["TestSplit"]))
    #test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize", 1024), return_filename=True)
    test_dataset = PCloader(specs["DataSource"], test_split, pc_size=24000, return_filename=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)
    
    # Load the trained model
    #ckpt = "{}.ckpt".format(args.resume) if args.resume == 'last' else "epoch={}.ckpt".format(args.resume)
    ckpt = "{}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
    model = CombinedModel.load_from_checkpoint(resume, specs=specs).cuda().eval()
    
    # Create diagnostic directory
    diagnostic_dir = os.path.join(args.exp_dir, "plane_diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    
    # Collect statistics across multiple samples
    all_stats = {'xz': [], 'xy': [], 'yz': []}
    
    # Process first 10 samples for statistics
    for idx, (point_cloud, filename) in enumerate(test_dataloader):
        if idx >= 10:
            break
            
        pc = point_cloud.cuda()
        
        # Get the ConvPointNet module
        pointnet = model.sdf_model.pointnet
        
        # Analyze each plane
        for plane in ['xz', 'xy', 'yz']:
            # Get normalized coordinates
            xy = pointnet.normalize_coordinate(pc.clone(), plane=plane, padding=pointnet.padding)
            
            # Get indices (this shows which cells points map to)
            index = pointnet.coordinate2index(xy, pointnet.reso_plane)
            
            # Count occupancy per cell
            batch_size = pc.shape[0]
            ones = torch.ones(batch_size, 1, pc.shape[1], device=pc.device)
            occupancy = scatter_add(ones, index, dim=2, dim_size=pointnet.reso_plane**2)
            
            # Reshape to 2D grid
            occupancy_grid = occupancy[0, 0].reshape(pointnet.reso_plane, pointnet.reso_plane)
            
            # Calculate statistics
            total_cells = pointnet.reso_plane ** 2
            empty_cells = (occupancy_grid == 0).sum().item()
            empty_ratio = empty_cells / total_cells
            
            # Find average points per non-empty cell
            non_empty_mask = occupancy_grid > 0
            avg_density = occupancy_grid[non_empty_mask].mean().item() if non_empty_mask.any() else 0
            max_density = occupancy_grid.max().item()
            
            stats = {
                'empty_ratio': empty_ratio,
                'avg_density': avg_density,
                'max_density': max_density,
                'empty_cells': empty_cells,
                'total_cells': total_cells
            }
            all_stats[plane].append(stats)
            
            # Save visualization for first sample
            if idx == 0:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 1. Occupancy heatmap
                im1 = axes[0].imshow(occupancy_grid.cpu().numpy(), cmap='hot', interpolation='nearest')
                axes[0].set_title(f'{plane} Plane Occupancy\n{empty_ratio:.1%} empty cells')
                axes[0].set_xlabel(f'Max density: {max_density:.0f} points/cell')
                plt.colorbar(im1, ax=axes[0])
                
                # 2. Binary occupancy (empty vs non-empty)
                binary_occupancy = (occupancy_grid > 0).float()
                axes[1].imshow(binary_occupancy.cpu().numpy(), cmap='binary', interpolation='nearest')
                axes[1].set_title(f'Binary Occupancy Map\n{empty_cells}/{total_cells} empty')
                
                # 3. Log-scale occupancy to see patterns better
                log_occupancy = torch.log(occupancy_grid + 1)
                im3 = axes[2].imshow(log_occupancy.cpu().numpy(), cmap='viridis', interpolation='nearest')
                axes[2].set_title(f'Log-scale Occupancy\nAvg density: {avg_density:.2f}')
                plt.colorbar(im3, ax=axes[2])
                
                plt.suptitle(f'Plane {plane.upper()} - Point Cloud Size: {pc.shape[1]}')
                plt.tight_layout()
                plt.savefig(os.path.join(diagnostic_dir, f'{plane}_occupancy_sample0.png'), dpi=150)
                plt.close()
                
                # Additional diagnostic: Show where artifacts likely appear
                # Near axes, one coordinate is close to 0
                if plane == 'xz':  # Y-axis problematic
                    problem_rows = occupancy_grid[31:33, :].sum(dim=0)  # Middle rows
                elif plane == 'xy':  # Z-axis problematic
                    problem_rows = occupancy_grid[:, 31:33].sum(dim=1)  # Middle columns
                else:  # yz plane, X-axis problematic
                    problem_rows = occupancy_grid[31:33, :].sum(dim=0)
                
                # Plot cross-section
                plt.figure(figsize=(10, 3))
                plt.bar(range(len(problem_rows)), problem_rows.cpu().numpy())
                plt.title(f'{plane} Plane: Density near problematic axis')
                plt.xlabel('Grid cell index')
                plt.ylabel('Point count')
                plt.savefig(os.path.join(diagnostic_dir, f'{plane}_axis_cross_section.png'), dpi=150)
                plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PLANE OCCUPANCY ANALYSIS RESULTS")
    print("="*60)
    
    for plane in ['xz', 'xy', 'yz']:
        stats = all_stats[plane]
        avg_empty = np.mean([s['empty_ratio'] for s in stats])
        avg_density = np.mean([s['avg_density'] for s in stats])
        avg_max = np.mean([s['max_density'] for s in stats])
        
        print(f"\n{plane.upper()} Plane Statistics (across {len(stats)} samples):")
        print(f"  Empty cells: {avg_empty:.1%}")
        print(f"  Avg points per occupied cell: {avg_density:.2f}")
        print(f"  Avg max density: {avg_max:.1f}")
        
        # Theoretical expectation
        expected_empty = np.exp(-specs.get("PCsize", 1024) / (pointnet.reso_plane ** 2))
        print(f"  Theoretical empty ratio (Poisson): {expected_empty:.1%}")
    
    # Save detailed statistics to file
    with open(os.path.join(diagnostic_dir, 'occupancy_stats.txt'), 'w') as f:
        f.write("Detailed Plane Occupancy Statistics\n")
        f.write("="*60 + "\n\n")
        
        for plane in ['xz', 'xy', 'yz']:
            f.write(f"{plane.upper()} Plane:\n")
            for i, stats in enumerate(all_stats[plane]):
                f.write(f"  Sample {i}: {stats['empty_ratio']:.1%} empty, "
                       f"avg density {stats['avg_density']:.2f}, "
                       f"max {stats['max_density']:.0f}\n")
            f.write("\n")
    
    print(f"\nDiagnostics saved to: {diagnostic_dir}")
    
    # Test hypothesis: Check correlation between sparsity and artifact locations
    print("\n" + "="*60)
    print("HYPOTHESIS TEST: Sparsity at Artifact Locations")
    print("="*60)
    
    # For the first sample, check SDF smoothness vs occupancy
    point_cloud, _ = test_dataset[0]
    pc = point_cloud.unsqueeze(0).cuda()
    
    # Sample along axes where artifacts typically appear
    test_sdf_smoothness_vs_occupancy(model, pc, pointnet, diagnostic_dir)


def test_sdf_smoothness_vs_occupancy(model, pc, pointnet, save_dir):
    """
    Test if SDF roughness correlates with plane sparsity
    """
    results = []
    
    # Test queries at different distances from axes
    for offset in [0.0, 0.05, 0.1, 0.2]:
        # Query line parallel to X-axis at offset from origin
        num_points = 100
        query = torch.zeros(1, num_points, 3).cuda()
        query[0, :, 0] = torch.linspace(-0.4, 0.4, num_points)
        query[0, :, 1] = offset  # Y offset
        query[0, :, 2] = offset  # Z offset
        
        # Get SDF predictions
        with torch.no_grad():
            # Get shape features
            shape_features = model.sdf_model.pointnet(pc, query)
            # Concatenate coordinates and features (as SdfDecoder expects)
            combined_input = torch.cat([query, shape_features], dim=-1)
            # Get SDF prediction
            sdf = model.sdf_model.model(combined_input).squeeze()
        
        # Compute roughness (TV norm)
        first_diff = sdf[1:] - sdf[:-1]
        second_diff = first_diff[1:] - first_diff[:-1]
        roughness = second_diff.abs().mean().item()
        
        # Check occupancy near this query line for each plane
        occupancies = {}
        for plane in ['xz', 'xy', 'yz']:
            xy = pointnet.normalize_coordinate(query.clone(), plane=plane, padding=pointnet.padding)
            index = pointnet.coordinate2index(xy, pointnet.reso_plane)
            
            # Count unique cells touched
            unique_cells = len(torch.unique(index))
            occupancies[plane] = unique_cells / num_points
        
        results.append({
            'offset': offset,
            'roughness': roughness,
            'occupancy_xz': occupancies['xz'],
            'occupancy_xy': occupancies['xy'],
            'occupancy_yz': occupancies['yz']
        })
        
        print(f"Offset {offset:.2f}: roughness={roughness:.4f}, "
              f"occupancy=[xz:{occupancies['xz']:.2f}, "
              f"xy:{occupancies['xy']:.2f}, yz:{occupancies['yz']:.2f}]")
    
    # Plot correlation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    offsets = [r['offset'] for r in results]
    roughness = [r['roughness'] for r in results]
    
    axes[0].plot(offsets, roughness, 'o-', label='SDF Roughness')
    axes[0].set_xlabel('Distance from axis')
    axes[0].set_ylabel('Roughness (2nd derivative)')
    axes[0].set_title('SDF Smoothness vs Distance from Axis')
    axes[0].grid(True)
    
    # Plot occupancy
    for plane in ['xz', 'xy', 'yz']:
        occ = [r[f'occupancy_{plane}'] for r in results]
        axes[1].plot(offsets, occ, 'o-', label=f'{plane} plane')
    
    axes[1].set_xlabel('Distance from axis')
    axes[1].set_ylabel('Cell coverage ratio')
    axes[1].set_title('Plane Occupancy vs Distance from Axis')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roughness_vs_occupancy.png'), dpi=150)
    plt.close()
    
    print("\nCorrelation test complete. Check roughness_vs_occupancy.png")

@torch.no_grad()
def test_modulations(args, specs):
    
    # load dataset, dataloader, model checkpoint
    test_split = json.load(open(specs["TestSplit"]))
    test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize",1024), return_filename=True)
    #test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize",8192), return_filename=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

    specs['compile_model'] = False
    specs['augment_training'] = False
    
    #ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    ckpt = "{}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)

    # Load checkpoint and fix compiled model keys
    checkpoint = torch.load(resume)
    checkpoint['state_dict'] = fix_compiled_state_dict(checkpoint['state_dict'])
    #checkpoint['state_dict'] = fix_compiled_state_dict(checkpoint['state_dict'])
    
    #model = CombinedModel.load_from_checkpoint(resume, specs=specs).cuda().eval()
    model = CombinedModel(specs) #.load_from_checkpoint(resume, specs=specs).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda().eval()

    # filename for logging chamfer distances of reconstructed meshes
    cd_file = os.path.join(recon_dir, "cd.csv") 

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))

            point_cloud, filename = data # filename = path to the csv file of sdf data
            filename = filename[0] # filename is a tuple

            cls_name = filename.split("/")[-3]
            mesh_name = filename.split("/")[-2]
            outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
            os.makedirs(outdir, exist_ok=True)
            mesh_filename = os.path.join(outdir, "reconstruct")
            
            # given point cloud, create modulations (e.g. 1D latent vectors)
            plane_features = model.sdf_model.pointnet.get_plane_features(point_cloud.cuda())  # tuple, 3 items with ([1, D, resolution, resolution])
            plane_features = torch.cat(plane_features, dim=1) # ([1, D*3, resolution, resolution])
            recon = model.vae_model.generate(plane_features) # ([1, D*3, resolution, resolution])
            #print("mesh filename: ", mesh_filename)
            # N is the grid resolution for marching cubes; set max_batch to largest number gpu can hold
            mesh.create_mesh(model.sdf_model, recon, mesh_filename, N=384, max_batch=2**16, from_plane_features=True)

            # load the created mesh (mesh_filename), and compare with input point cloud
            # to calculate and log chamfer distance 
            mesh_log_name = cls_name+"/"+mesh_name
            try:
                evaluate.main(point_cloud, mesh_filename, cd_file, mesh_log_name)
            except Exception as e:
                print(e)


            # save modulation vectors for training diffusion model for next stage
            # filter based on the chamfer distance so that all training data for diffusion model is clean 
            # would recommend visualizing some reconstructed meshes and manually determining what chamfer distance threshold to use
            try:
                # skips modulations that have chamfer distance > 0.0018
                # the filter also weighs gaps / empty space higher
                if not filter_threshold(mesh_filename, point_cloud, 0.006): 
                    continue
                outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)
                features = model.sdf_model.pointnet.get_plane_features(point_cloud.cuda())
                features = torch.cat(features, dim=1) # ([1, D*3, resolution, resolution])
                latent = model.vae_model.get_latent(features) # (1, D*3)
                np.savetxt(os.path.join(outdir, "latent.txt"), latent.cpu().numpy())
            except Exception as e:
                print(e)


           
@torch.no_grad()
def test_generation(args, specs):

    # load model 
    if args.resume == 'finetune': # after second stage of training 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # loads the sdf and vae models
            model = CombinedModel.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False) 

            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k,v in ckpt['state_dict'].items():
                new_key = k.replace("diffusion_model.", "") # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)

            model = model.cuda().eval()
    else:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs).cuda().eval()

    conditional = specs["diffusion_model_specs"]["cond"] 

    if not conditional:
        samples = model.diffusion_model.generate_unconditional(args.num_samples)
        plane_features = model.vae_model.decode(samples)
        for i in range(len(plane_features)):
            plane_feature = plane_features[i].unsqueeze(0)
            mesh.create_mesh(model.sdf_model, plane_feature, recon_dir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            
    else:
        # load dataset, dataloader, model checkpoint
        test_split = json.load(open(specs["TestSplit"]))
        test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize",1024), return_filename=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

        with tqdm(test_dataloader) as pbar:
            for idx, data in enumerate(pbar):
                pbar.set_description("Files generated: {}/{}".format(idx, len(test_dataloader)))

                point_cloud, filename = data # filename = path to the csv file of sdf data
                filename = filename[0] # filename is a tuple

                cls_name = filename.split("/")[-3]
                mesh_name = filename.split("/")[-2]
                outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)

                # filter, set threshold manually after a few visualizations
                if args.filter:
                    threshold = 0.08
                    tmp_lst = []
                    count = 0
                    while len(tmp_lst)<args.num_samples:
                        count+=1
                        samples, perturbed_pc = model.diffusion_model.generate_from_pc(point_cloud.cuda(), batch=args.num_samples, save_pc=outdir, return_pc=True) # batch should be set to max number GPU can hold
                        plane_features = model.vae_model.decode(samples)
                        # predicting the sdf values of the point cloud
                        perturbed_pc_pred = model.sdf_model.forward_with_plane_features(plane_features, perturbed_pc.repeat(args.num_samples, 1, 1))
                        consistency = F.l1_loss(perturbed_pc_pred, torch.zeros_like(perturbed_pc_pred), reduction='none')
                        loss = reduce(consistency, 'b ... -> b', 'mean', b = consistency.shape[0]) # one value per generated sample 
                        #print("consistency shape: ", consistency.shape, loss.shape, consistency[0].mean(), consistency[1].mean(), loss) # cons: [B,N]; loss: [B]
                        thresh_idx = loss<=threshold
                        tmp_lst.extend(plane_features[thresh_idx])

                        if count > 5: # repeat this filtering process as needed 
                            break
                    # skip the point cloud if cannot produce consistent samples or 
                    # just use the samples that are produced if comparing to other methods
                    if len(tmp_lst)<1: 
                        continue
                    plane_features = tmp_lst[0:min(10,len(tmp_lst))]

                else:
                    # for each point cloud, the partial pc and its conditional generations are all saved in the same directory 
                    samples, perturbed_pc = model.diffusion_model.generate_from_pc(point_cloud.cuda(), batch=args.num_samples, save_pc=outdir, return_pc=True)
                    plane_features = model.vae_model.decode(samples)
                
                for i in range(len(plane_features)):
                    plane_feature = plane_features[i].unsqueeze(0)
                    mesh.create_mesh(model.sdf_model, plane_feature, outdir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            


    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--specs", "-s", required=True,
        help="The config specification for the experiment"
    )
    arg_parser.add_argument(
        "--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument("--num_samples", "-n", default=5, type=int, help='number of samples to generate and reconstruct')

    arg_parser.add_argument("--filter", default=False, help='whether to filter when sampling conditionally')

    # Add new argument for diagnostic mode
    arg_parser.add_argument("--diagnostic", default=None, 
                           choices=['plane_occupancy'],
                           help="Run diagnostic tests")

    args = arg_parser.parse_args()
    specs = json.load(open(args.specs))
    print(specs["Description"])


    recon_dir = os.path.join(args.exp_dir, "recon")
    os.makedirs(recon_dir, exist_ok=True)

    # Run diagnostic if specified
    if args.diagnostic == 'plane_occupancy':
        test_plane_occupancy(args, specs)
    elif specs['training_task'] == 'modulation':
        latent_dir = os.path.join(args.exp_dir, "modulations")
        os.makedirs(latent_dir, exist_ok=True)
        test_modulations(args, specs)
    elif specs['training_task'] == 'combined':
        test_generation(args, specs)