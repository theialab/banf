import os
import math
from arrgh import arrgh

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from einops import rearrange
from skimage.io import imsave
from PIL import Image
import scipy

from dataloader import get_loader
from utils import bilinear_interpolation, psnr_score, get_fft, lanczos_interpolation, visualize
from interpolate import sinc_interpolation
import torch.nn.functional as F


class DenseGrid(nn.Module):
    def __init__(self, base_res=16, max_res=256, num_lod=8, interpolation_type="closest", zero_init=False):
        super().__init__()
        self.zero_init = zero_init
        self.num_lod = num_lod
        self.feat_dim = 2  # feature dim size
        self.codebook = nn.ParameterList([])
        self.interpolation_type = interpolation_type  # bilinear
        growth_factor = np.exp((math.log(max_res) - math.log(base_res)) / (num_lod - 1))

        self.LODS = [int(base_res*growth_factor**L) for L in range(num_lod)]
        self.init_feature_structure()

    def init_feature_structure(self):
        for LOD in self.LODS:
            fts = torch.zeros(LOD**2, self.feat_dim)
            if not self.zero_init:
                fts += torch.randn_like(fts) * 1e-2
            else:
                fts += torch.randn_like(fts) * 1e-6
            fts = nn.Parameter(fts)
            self.codebook.append(fts)

    def forward(self, pts):
        feats = []
        # Iterate in every level of detail resolution
        for i, res in enumerate(self.LODS):
            if self.interpolation_type == "closest":
                x = pts[:, 0] * (res - 1)
                x = torch.floor(x).int()

                y = pts[:, 1] * (res - 1)
                y = torch.floor(y).int()

                features = self.codebook[i][(x + y * res).long()]
            elif self.interpolation_type == "bilinear":
                features = bilinear_interpolation(res, self.codebook[i], pts, "NGLOD")

            else:
                raise NotImplementedError

            feats.append((torch.unsqueeze(features, dim=-1)))
        all_features = torch.cat(feats, -1)
        return all_features.reshape(-1, self.num_lod*self.feat_dim)


class MLP(nn.Module):
    def __init__(
        self, grid_structure, input_dim, hidden_dim, output_dim, num_hidden_layers=3
    ):
        super().__init__()
        self.module_list = torch.nn.ModuleList()
        self.module_list.append(torch.nn.Linear(input_dim, hidden_dim, bias=True))
        for i in range(num_hidden_layers):
            osize = hidden_dim if i < num_hidden_layers - 1 else output_dim
            self.module_list.append(torch.nn.ReLU())
            self.module_list.append(torch.nn.Linear(hidden_dim, osize, bias=True))
        self.model = torch.nn.Sequential(*self.module_list)
        self.grid_structure = grid_structure

    def forward(self, coords):
        h, w, c = coords.shape
        coords = rearrange(coords, "h w c -> (h w) c")
        feat = self.grid_structure(coords)
        out = self.model(feat)
        out = rearrange(out, "(h w) c -> h w c", h=h, w=w)

        return out


class MLP_grid_wrapper(nn.Module):
    def __init__(self, NN_model, grid_res):
        super().__init__()
        self.grid_res = grid_res
        self.model = NN_model
        self.blur = False
    
    def sinc_interpolate_grid(self, grid, coords):
        in_shape = coords.shape[0]
        grid = grid.permute(2, 3, 1, 0).squeeze().view(-1, 3)
        coords = coords.view(-1, 2)

        out = sinc_interpolation(coords, self.grid_res, grid)
        out = out.reshape(in_shape, in_shape, 3)
        # arrgh(out)
        return out

    def get_nngrid(self,):
        grid_pts = torch.meshgrid(
            torch.arange(0, 1, 1/self.grid_res),
            torch.arange(0, 1, 1/self.grid_res),
        )
        grid_pts = torch.stack(grid_pts, dim=-1)#.cuda()
        grid_vals = self.model(grid_pts)[None]
        return grid_vals.permute(0, 3, 1, 2) # [1, 3, H, W]

    def forward(self, coords, interpolate=True):
        nn_grid = self.get_nngrid()
        if interpolate:
            vals = self.sinc_interpolate_grid(nn_grid, coords)
            return vals
        else:
            return nn_grid.squeeze().permute(1, 2, 0)


class MixMode(nn.Module):
    def __init__(self, model1, model2, model3):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
    
    def pred1(self, x, interpolate=True):
        return self.model1(x, interpolate)
    
    def pred2(self, x, interpolate=True):
        return self.model2(x, interpolate)
    
    def pred3(self, x, interpolate=True):
        return self.model3(x, interpolate)