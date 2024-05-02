from arrgh import arrgh
import torch
import math
import numpy as np
import cv2
import os


def gaussian_interpolation(positions, resolution, grid_vals): # , patch_size=30
    num_coords, coord_dim = positions.shape
    k = math.sqrt(math.log(2))
    sigma = 2.0#/resolution
    patch_size = resolution

    x = resolution * positions
    pos = torch.floor(x).long()
    chunk_size = 8192
    num_chunks = math.ceil(num_coords/chunk_size)
    FINAL_VALS = torch.zeros([num_coords, 3], device=positions.device)

    if num_chunks > 1:
        for i in range(num_chunks):
            chunk_pos = pos[i*chunk_size:(i+1)*chunk_size]
            chunk_x = x[i*chunk_size:(i+1)*chunk_size]
            add_corners = torch.meshgrid(
                    torch.linspace(-patch_size//2, patch_size//2-1, patch_size),
                    torch.linspace(-patch_size//2, patch_size//2-1, patch_size),
                )
            add_corners = torch.stack(add_corners, dim=-1).view(1, -1, 2).long().to(positions.device)
            corners = chunk_pos.unsqueeze(1) + add_corners

            difference = (chunk_x.view(-1, 1, coord_dim) - corners)# / resolution
            forb_norm = torch.sum(difference**2, dim=-1)
            gau_weights = torch.exp(-0.5*forb_norm/(sigma**2))
            gau_weights_norm = gau_weights.sum(-1, keepdim=True)
            gau_weights /= gau_weights_norm

            corners = torch.clamp(corners, 0, resolution-1)
            corner_idxes = corners[..., 0] * resolution + corners[..., 1]
            corner_idxes = corner_idxes.view(-1).type(torch.int)

            corner_grid_vals = torch.index_select(grid_vals, dim=0, index=corner_idxes)
            corner_grid_vals = corner_grid_vals.view(chunk_size, patch_size**2, 3)

            final_vals = torch.sum(corner_grid_vals * gau_weights.unsqueeze(-1), dim=1)

            FINAL_VALS[i*chunk_size:(i+1)*chunk_size] = final_vals
    
    else:
        add_corners = torch.meshgrid(
                torch.linspace(-patch_size//2, patch_size//2-1, patch_size),
                torch.linspace(-patch_size//2, patch_size//2-1, patch_size),
            )
        add_corners = torch.stack(add_corners, dim=-1).view(1, -1, 2).long().to(positions.device)
        corners = pos.unsqueeze(1) + add_corners

        difference = (x.view(-1, 1, coord_dim) - corners) / resolution
        forb_norm = torch.sum(difference**2, dim=-1)
        gau_weights = torch.exp(-0.5*forb_norm/(sigma**2))
        gau_weights_norm = gau_weights.sum(-1, keepdim=True)
        gau_weights /= gau_weights_norm

        corners = torch.clamp(corners, 0, resolution-1)
        corner_idxes = corners[..., 0] * resolution + corners[..., 1]
        corner_idxes = corner_idxes.view(-1).type(torch.int)

        corner_grid_vals = torch.index_select(grid_vals, dim=0, index=corner_idxes)
        corner_grid_vals = corner_grid_vals.view(num_coords, patch_size**2, 3)

        FINAL_VALS = torch.sum(corner_grid_vals * gau_weights.unsqueeze(-1), dim=1)
    
    return FINAL_VALS


def sinc_interpolation(positions, resolution, grid_vals):#, patch_size=63):
    num_coords, coord_dim = positions.shape
    patch_size = 25

    x = resolution * positions
    pos = torch.floor(x).long()
    FINAL_VALS = torch.zeros([num_coords, 3], device=positions.device)

    add_corners = torch.meshgrid(
                torch.linspace(-patch_size//2+1, patch_size//2, patch_size),
                torch.linspace(-patch_size//2+1, patch_size//2, patch_size),
            )
    add_corners = torch.stack(add_corners, dim=-1).view(1, -1, 2).long().to(positions.device)
    corners = pos.unsqueeze(1) + add_corners

    difference = (x.view(num_coords, 1, coord_dim) - corners)
    sinc_weights = torch.sinc(difference).prod(dim=-1)

    # corners = torch.clamp(corners, 0, resolution-1)
    corners[corners > resolution-1] -= resolution
    corners[corners < 0] += resolution
    corner_idxes = corners[..., 0] * resolution + corners[..., 1]
    corner_idxes = corner_idxes.view(-1).type(torch.int)

    corner_grid_vals = torch.index_select(grid_vals, dim=0, index=corner_idxes)
    corner_grid_vals = corner_grid_vals.view(num_coords, patch_size**2, 3)

    FINAL_VALS = torch.sum(corner_grid_vals * sinc_weights.unsqueeze(-1), dim=1)
    
    return FINAL_VALS


def linear_interpolation(positions, resolution, grid_vals):#, patch_size=63):
    num_coords, coord_dim = positions.shape
    patch_size = 2

    x = resolution * positions
    pos = torch.floor(x).long()
    FINAL_VALS = torch.zeros([num_coords, 3], device=positions.device)

    add_corners = torch.meshgrid(
                torch.linspace(-patch_size//2+1, patch_size//2, patch_size),
                torch.linspace(-patch_size//2+1, patch_size//2, patch_size),
            )
    add_corners = torch.stack(add_corners, dim=-1).view(1, -1, 2).long().to(positions.device)
    corners = pos.unsqueeze(1) + add_corners

    difference = (x.view(num_coords, 1, coord_dim) - corners)
    linear_weights = (1-torch.abs(difference)).prod(dim=-1)
    # arrgh(difference, linear_weights)
    # arrgh(grid_vals, x, pos, add_corners, corners, difference, sinc_weights)

    # corners = torch.clamp(corners, 0, resolution-1)
    corners[corners > resolution-1] -= resolution
    corners[corners < 0] += resolution
    corner_idxes = corners[..., 0] * resolution + corners[..., 1]
    corner_idxes = corner_idxes.view(-1).type(torch.int)

    corner_grid_vals = torch.index_select(grid_vals, dim=0, index=corner_idxes)
    corner_grid_vals = corner_grid_vals.view(num_coords, patch_size**2, 3)

    FINAL_VALS = torch.sum(corner_grid_vals * linear_weights.unsqueeze(-1), dim=1)
    
    return FINAL_VALS