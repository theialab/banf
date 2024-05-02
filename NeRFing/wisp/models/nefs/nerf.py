# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
from typing import Dict, Any, Optional
from wisp.ops.geometric import sample_unif_sphere
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import BLASGrid, HashGrid, TriplanarGrid
from typing import Optional
import torch.nn.functional as F
import torch.nn as nn
from local_scripts.sh import eval_sh


class NeuralRadianceField(BaseNeuralField):
    """Model for encoding Neural Radiance Fields (Mildenhall et al. 2020), e.g., density and view dependent color.
    Different to the original NeRF paper, this implementation uses feature grids for a
    higher quality and more efficient implementation, following later trends in the literature,
    such as Neural Sparse Voxel Fields (Liu et al. 2020), Instant Neural Graphics Primitives (Muller et al. 2022)
    and Variable Bitrate Neural Fields (Takikawa et al. 2022).
    """

    def __init__(self,
                 grid: BLASGrid,
                 # embedder args
                 pos_embedder: str = 'none',    # options: 'none', 'identity', 'positional'
                 view_embedder: str = 'none',   # options: 'none', 'identity', 'positional'
                 pos_multires: int = 10,
                 view_multires: int = 4,
                 position_input: bool = False,
                 # decoder args
                 activation_type: str = 'relu', #  options: 'none', 'relu', 'sin', 'fullsort', 'minmax'
                 layer_type: str = 'linear',    # 'linear', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 bias: bool = False,
                 # pruning args
                 prune_density_decay: Optional[float] = (0.01 * 512) / np.sqrt(3),
                 prune_min_density: Optional[float] = 0.6,
                 cur_res: int = 10042301,
                 ):
        """
        Creates a new NeRF instance, which maps 3D input coordinates + view directions to RGB + density.

        This neural field consists of:
         * A feature grid (backed by an acceleration structure to boost raymarching speed)
         * Color & density decoders
         * Optional: positional embedders for input position coords & view directions, concatenated to grid features.

         This neural field also supports:
          * Aggregation of multi-resolution features (more than one LOD) via summation or concatenation
          * Pruning scheme for HashGrids

        Args:
            grid: (BLASGrid): represents feature grids in Wisp. BLAS: "Bottom Level Acceleration Structure",
                to signify this structure is the backbone that captures
                a neural field's contents, in terms of both features and occupancy for speeding up queries.
                Notable examples: OctreeGrid, HashGrid, TriplanarGrid, CodebookGrid.

            pos_embedder (str): Type of positional embedder to use for input coordinates.
                Options:
                 - 'none': No positional input is fed into the density decoder.
                 - 'identity': The sample coordinates are fed as is into the density decoder.
                 - 'positional': The sample coordinates are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the density decoder.
            view_embedder (str): Type of positional embedder to use for view directions.
                Options:
                 - 'none': No positional input is fed into the color decoder.
                 - 'identity': The view directions are fed as is into the color decoder.
                 - 'positional': The view directions are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the color decoder.
            pos_multires (int): Number of frequencies used for 'positional' embedding of pos_embedder.
                 Used only if pos_embedder is 'positional'.
            view_multires (int): Number of frequencies used for 'positional' embedding of view_embedder.
                 Used only if view_embedder is 'positional'.
            position_input (bool): If True, the input coordinates will be passed into the decoder.
                 For 'positional': the input coordinates will be concatenated to the embedded coords.
                 For 'none' and 'identity': the embedder will behave like 'identity'.
            activation_type (str): Type of activation function to use in BasicDecoder:
                 'none', 'relu', 'sin', 'fullsort', 'minmax'.
            layer_type (str): Type of MLP layer to use in BasicDecoder:
                 'none' / 'linear', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'.
            hidden_dim (int): Number of neurons in hidden layers of both decoders.
            num_layers (int): Number of hidden layers in both decoders.
            bias (bool): Whether to use bias in the decoders.
            prune_density_decay (Optional[float]): Decay rate of density per "prune step",
                 using the pruning scheme from Muller et al. 2022. Used only for grids which support pruning.
            prune_min_density (Optional[float]): Minimal density allowed for "cells" before they get pruned during a "prune step".
                 Used within the pruning scheme from Muller et al. 2022. Used only for grids which support pruning.
        """
        super().__init__()
        self.grid = grid[0]
        self.grid2 = grid[1]

        self.pos_embedder_type = pos_embedder
        self.view_embedder_type = view_embedder
        # Init Embedders
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires,
                                                                   include_input=position_input)
        self.view_embedder, self.view_embed_dim = self.init_embedder(view_embedder, view_multires,
                                                                     include_input=True)

        # Init Decoder
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bias = bias
        self.decoder_density, self.decoder_color = \
            self.init_decoders(activation_type, layer_type, num_layers, hidden_dim)

        self.prune_density_decay = prune_density_decay
        self.prune_min_density = prune_min_density
        self.cur_res = cur_res

        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None, include_input=False):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none' and not include_input:
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity' or (embedder_type == 'none' and include_input):
            embedder, embed_dim = torch.nn.Identity(), 3    # Assumes pos / view input is always 3D
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies, include_input=include_input)
        elif embedder_type == 'tcnn':
            import tinycudann as tcnn
            embedder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                    ],
                },
            )
            embed_dim = 16
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralRadianceField: {embedder_type}')
        return embedder, embed_dim

    def init_decoders(self, activation_type, layer_type, num_layers, hidden_dim):
        """Initializes the decoder object.
        """
        self.decoder_density64 = BasicDecoder(input_dim=self.density_net_input_dim(),
                                       output_dim=16,
                                       activation=get_activation_class(activation_type),
                                       bias=self.bias,
                                       layer=get_layer_class(layer_type),
                                       num_layers=num_layers,
                                       hidden_dim=hidden_dim,
                                       skip=[])
        if self.decoder_density64.lout.bias is not None:
            self.decoder_density64.lout.bias.data[0] = 1.0
        
        self.decoder_density128 = BasicDecoder(input_dim=self.density_net_input_dim(),
                                       output_dim=16,
                                       activation=get_activation_class(activation_type),
                                       bias=self.bias,
                                       layer=get_layer_class(layer_type),
                                       num_layers=num_layers,
                                       hidden_dim=hidden_dim,
                                       skip=[])
        self.decoder_density128.lout.bias.data[0] = 1

        self.decoder_color64 = BasicDecoder(input_dim=self.color_net_input_dim(),
                                     output_dim=27, #3,
                                     activation=get_activation_class(activation_type),
                                     bias=self.bias,
                                     layer=get_layer_class(layer_type),
                                     num_layers=num_layers + 1,
                                     hidden_dim=hidden_dim,
                                     skip=[])
        
        self.decoder_color128 = BasicDecoder(input_dim=self.color_net_input_dim(),
                                     output_dim=27, #3,
                                     activation=get_activation_class(activation_type),
                                     bias=self.bias,
                                     layer=get_layer_class(layer_type),
                                     num_layers=num_layers + 1,
                                     hidden_dim=hidden_dim,
                                     skip=[])
        return None, None

    def prune(self):
        """Prunes the blas based on current state.
        """
        if self.prune_density_decay is None or self.prune_min_density is None:
            return
        if self.grid is not None:
            if isinstance(self.grid, (HashGrid, TriplanarGrid)):
                density_decay = self.prune_density_decay
                min_density = self.prune_min_density

                self.grid.occupancy = self.grid.occupancy.cuda()
                self.grid.occupancy = self.grid.occupancy * density_decay
                points = self.grid.dense_points.cuda()
                res = 2.0**self.grid.blas.max_level
                samples = torch.rand(points.shape[0], 3, device=points.device)
                samples = points.float() + samples
                samples = samples / res
                samples = samples * 2.0 - 1.0
                sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
                with torch.no_grad():
                    density = self.forward(coords=samples, ray_d=sample_views, channels="density")
                self.grid.occupancy = torch.stack([density[:, 0], self.grid.occupancy], -1).max(dim=-1)[0]

                mask = self.grid.occupancy > min_density

                _points = points[mask]

                if _points.shape[0] == 0:
                    return

                if hasattr(self.grid.blas.__class__, "from_quantized_points"):
                    self.grid.blas = self.grid.blas.__class__.from_quantized_points(_points, self.grid.blas.max_level)
                else:
                    raise Exception(f"The BLAS {self.grid.blas.__class__.__name__} does not support initialization " 
                                     "from_quantized_points, which is required for pruning.")

            else:
                raise NotImplementedError(f'Pruning not implemented for grid type {self.grid.__class__.__name__}')

    def register_forward_functions(self):
        """Registers the forward function to call per requested channel.
        """
        self._register_forward_function(self.rgba, ["density", "rgb"])

    def get_density_grid(self, coords, lod_idx, resolution, den_model, grid):
        grid_pts = torch.meshgrid(
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution),
            torch.linspace(-1, 1, resolution),
        )
        grid_pts = torch.stack(grid_pts, dim=-1).view(-1, 3).to(coords.device)

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = grid.interpolate(grid_pts, lod_idx).reshape(resolution**3, self.effective_feature_dim())

        density_feats_grid = den_model(feats)

        density_feats = density_feats_grid.T.view(-1, resolution, resolution, resolution)[None, ...]
        density_feats[..., 0:1] = torch.relu(density_feats[..., 0:1])
        density_feats = interpolate_grid(density_feats, coords/2 + 0.5)
        density = torch.relu(density_feats[...,0:1])

        return density, density_feats_grid
    
    def get_color(self, coords, ray_d, density_feats_grid, resolution, color_model):
        color_feat = density_feats_grid[:, 1:]
        sh_grid = color_model(color_feat)
        sh_grid = sh_grid.T.view(-1, resolution, resolution, resolution)[None, ...]
        sh_vals = interpolate_grid(sh_grid, coords/2 + 0.5)
        sh_vals = sh_vals.reshape(-1, 3, 9)
        colors = eval_sh(2, sh_vals, ray_d)

        return colors

    def rgba(self, coords, ray_d, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, 3]
            ray_d (torch.FloatTensor): tensor of shape [batch, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, 3]
                - Density tensor of shape [batch, 1]
        """
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, _ = coords.shape
        
        if self.cur_res == 64:
            density64, density_feats_grid64 = self.get_density_grid(
                coords, lod_idx, resolution=32, den_model=self.decoder_density64, grid=self.grid)
            density = density64
        
        elif self.cur_res == 128:
            with torch.no_grad():
                density64, density_feats_grid64 = self.get_density_grid(
                    coords, lod_idx, resolution=32, den_model=self.decoder_density64, grid=self.grid)
            
            density128, density_feats_grid128 = self.get_density_grid(
                coords, lod_idx, resolution=64, den_model=self.decoder_density128, grid=self.grid2)
            density = density64 + density128

        ### SH color
        if self.cur_res == 64:
            colors64 = self.get_color(coords, ray_d, density_feats_grid64, 
                resolution=32, color_model=self.decoder_color64)
            colors = colors64
        
        elif self.cur_res == 128:
            with torch.no_grad():
                colors64 = self.get_color(coords, ray_d, density_feats_grid64, 
                    resolution=32, color_model=self.decoder_color64)
            
            colors128 = self.get_color(coords, ray_d, density_feats_grid128,
                resolution=64, color_model=self.decoder_color128)
            colors = colors64 + colors128
        
        return dict(rgb=colors, density=density)

    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    def density_net_input_dim(self):
        return self.effective_feature_dim() + self.pos_embed_dim

    def color_net_input_dim(self):
        return 15 #+ self.view_embed_dim

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        properties = {
            "Grid": self.grid,
            "Pos. Embedding": self.pos_embedder,
            "View Embedding": self.view_embedder,
            "Decoder (density)": self.decoder_density,
            "Decoder (color)": self.decoder_color
        }
        if self.prune_density_decay is not None:
            properties['Pruning Density Decay'] = self.prune_density_decay
        if self.prune_min_density is not None:
            properties['Pruning Min Density'] = self.prune_min_density
        return properties


def interpolate_grid(grid, in_tensor):
    ### Grid shape: [1, N_feat]
    num_feat = grid.shape[1]
    shape = in_tensor.shape[:-1]
    in_tensor = in_tensor.reshape(1, 1, 1, -1, 3)
    ind_norm = in_tensor.flip((-1,)) * 2 - 1
    out = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
    out = out.reshape(num_feat,-1).T.reshape(*shape,num_feat)

    return out