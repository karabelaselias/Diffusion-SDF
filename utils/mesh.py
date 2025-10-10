#!/usr/bin/env python3

import logging
import math
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import trimesh



# N: resolution of grid; 256 is typically sufficient 
# max batch: as large as GPU memory will allow
# shape_feature is either point cloud, mesh_idx (neuralpull), or generated latent code (deepsdf)
def create_mesh(
    model, 
    shape_feature, 
    filename, 
    N=256, 
    max_batch=1000000, 
    level_set=0.0, 
    occupancy=False, 
    point_cloud=None, 
    from_plane_features=False, 
    from_pc_features=False,
    decode_from_latent=False,
    diffdmc = None
):
    start_time = time.time()
    ply_filename = filename

    model.eval()

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    coords = torch.linspace(-1, 1, N)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
    cube = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    #cube = create_cube(N).float().cuda()
    #cube_points = cube.shape[0]
    
    # add some noise to the cube values
    #cube[:, 0:3] += torch.randn_like(cube[:, 0:3]) * 0.002
    
    # Process in batches with no_grad to save memory
    shape_feature = shape_feature.cuda()
    sdf_values = []
    
    with torch.no_grad():
        for i in range(0, len(cube), max_batch):
            query = cube[i:i+max_batch].cuda().unsqueeze(0)
            if decode_from_latent:
                # VecSet decoding path
                pred_sdf = model.decode_from_latent(shape_feature, query)
            elif from_plane_features:
                pred_sdf = model.forward_with_plane_features(shape_feature, query)
            else:
                pred_sdf = model(shape_feature, query)
            sdf_values.append(pred_sdf.squeeze())
    if occupancy:
        sdf_values = sdf_values - 0.5
    sdf_values = torch.cat(sdf_values).reshape(N, N, N)
        
    convert_sdf_samples_to_ply(
        sdf_values,
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        level_set,
        diffdmc=diffdmc
    )


# create cube from (-1,-1,-1) to (1,1,1) and uniformly sample points for marching cube
def create_cube(N):

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long().float() / N) % N
    samples[:, 0] = ((overall_index.long().float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    return samples

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    level_set=0.0,
    diffdmc = None
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    #diffdmc = DiffDMC(dtype=torch.float32).cuda()
    #verts, faces = diffdmc(pytorch_3d_sdf_tensor.float().cuda(), None, isovalue=voxel_size)
    #print(verts)

    if diffdmc is not None:
        try:
            verts, faces = diffdmc(pytorch_3d_sdf_tensor, isovalue=level_set, normalize=False)
            verts = verts.detach().cpu().numpy()
            faces = faces.detach().cpu().numpy()
        except Exception as e:
            print("skipping {}; error: {}".format(ply_filename_out, e))
            return
        verts = voxel_size * verts - 1.0
    else:
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.detach().cpu().numpy()
        # use marching_cubes_lewiner or marching_cubes depending on pytorch version 
        try:
            verts, faces, _, _ = skimage.measure.marching_cubes(
                numpy_3d_sdf_tensor, level=level_set, spacing=[voxel_size] * 3
            )
        except Exception as e:
            print("skipping {}; error: {}".format(ply_filename_out, e))
            return
        verts = verts - 1.0

    verts_tuple = np.zeros(len(verts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i, vert in enumerate(verts):
        verts_tuple[i] = tuple(vert)
    
    faces_tuple = np.zeros(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    for i, face in enumerate(faces):
        faces_tuple[i] = (face,)

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")
    plyfile.PlyData([el_verts, el_faces]).write(ply_filename_out)
    
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes

    #mesh_points = np.zeros_like(verts)
    #mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    #mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    #mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    #num_verts = verts.shape[0]
    #num_faces = faces.shape[0]

    #verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    #for i in range(0, num_verts):
    #    verts_tuple[i] = tuple(mesh_points[i, :])

    #faces_building = []
    #for i in range(0, num_faces):
    #    faces_building.append(((faces[i, :].tolist(),)))
    #faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    #el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    #el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    #ply_data = plyfile.PlyData([el_verts, el_faces])
    #ply_data.write(ply_filename_out)


