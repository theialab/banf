# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Type, Union, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    PeriodicVolumeEncoding,
    TensorVMEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig
# from banf.base_field import Field, FieldConfig
from banf.banf_mlp import Geo_MLP, Color_MLP
from arrgh import arrgh

from nerfstudio.model_components.ray_samplers import save_points

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SigmoidDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Sigmoid density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        # negtive sdf will have large density
        return alpha * torch.sigmoid(-sdf * alpha)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@dataclass
class BanfFieldConfig(FieldConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: BanfField)
    num_layers: int = 8
    """Number of layers for geometric network"""
    hidden_dim: int = 256
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 64
    """Dimension of geometric feature"""
    num_layers_color: int = 4
    """Number of layers for color network"""
    hidden_dim_color: int = 256
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of appearance embedding"""
    use_appearance_embedding: bool = False
    """Dimension of appearance embedding"""
    bias: float = 0.8
    """sphere size of geometric initializaion"""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear laer"""
    use_grid_feature: bool = False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.1
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
    """feature grid encoding type"""
    position_encoding_max_degree: int = 6
    """positional encoding max degree"""
    use_diffuse_color: bool = False
    """whether to use diffuse color as in ref-nerf"""
    use_specular_tint: bool = False
    """whether to use specular tint as in ref-nerf"""
    use_reflections: bool = False
    """whether to use reflections as in ref-nerf"""
    use_n_dot_v: bool = False
    """whether to use n dot v as in ref-nerf"""
    rgb_padding: float = 0.001
    """Padding added to the RGB outputs"""
    off_axis: bool = False
    """whether to use off axis encoding from mipnerf360"""
    use_numerical_gradients: bool = False
    """whether to use numercial gradients"""
    num_levels: int = 16
    """number of levels for multi-resolution hash grids"""
    max_res: int = 2048
    """max resolution for multi-resolution hash grids"""
    base_res: int = 16
    """base resolution for multi-resolution hash grids"""
    log2_hashmap_size: int = 19
    """log2 hash map size for multi-resolution hash grids"""
    hash_features_per_level: int = 2
    """number of features per level for multi-resolution hash grids"""
    hash_smoothstep: bool = True
    """whether to use smoothstep for multi-resolution hash grids"""
    use_position_encoding: bool = True
    """whether to use positional encoding as input for geometric network"""
    color_continuous: bool = False
    cur_mode = "64"
    compute_laplacian_loss: bool = False


class BanfField(Field):
    """_summary_

    Args:
        Field (_type_): _description_
    """

    config: BanfFieldConfig

    def __init__(
        self,
        config: BanfFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # TODO do we need aabb here?
        self.aabb = Parameter(aabb, requires_grad=False)

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor

        self.num_levels = self.config.num_levels
        self.base_res = self.config.base_res 
        self.log2_hashmap_size = self.config.log2_hashmap_size 
        self.features_per_level = self.config.hash_features_per_level 
        use_hash = True
        smoothstep = self.config.hash_smoothstep

        self.max_res = self.config.max_res 
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))

        # feature encodings
        self.encoding1 = create_hash_encoding(
            self.num_levels, self.features_per_level, self.log2_hashmap_size, self.base_res, self.growth_factor, smoothstep
        )

        self.encoding2 = create_hash_encoding(
            self.num_levels, self.features_per_level, self.log2_hashmap_size, self.base_res, self.growth_factor, smoothstep
        )

        self.encoding3 = create_hash_encoding(
            self.num_levels, self.features_per_level, self.log2_hashmap_size, self.base_res, self.growth_factor, smoothstep
        )

        self.encoding4 = create_hash_encoding(
            self.num_levels, self.features_per_level, self.log2_hashmap_size, self.base_res, self.growth_factor, smoothstep
        )

        self.hash_encoding_mask = torch.ones(
            self.num_levels * self.features_per_level,
            dtype=torch.float32,
        )

        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=self.config.position_encoding_max_degree,
            min_freq_exp=0.0,
            max_freq_exp=self.config.position_encoding_max_degree - 1,
            include_input=False,
            off_axis=self.config.off_axis,
        )

        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        self.geo64 = Geo_MLP(self, init_factor=1)
        self.geo128 = Geo_MLP(self, init_factor=0)
        self.geo256 = Geo_MLP(self, init_factor=0)
        self.geoinf = Geo_MLP(self, init_factor=0)
        self.grid_sdfs_dict = {}

        self.color64 = Color_MLP(self, init_factor=1, color_continuous=self.config.color_continuous)
        self.color128 = Color_MLP(self, init_factor=1, color_continuous=self.config.color_continuous)
        self.color256 = Color_MLP(self, init_factor=1, color_continuous=self.config.color_continuous)
        self.colorinf = Color_MLP(self, init_factor=1, color_continuous=self.config.color_continuous)
        
        self.config.cur_mode="64"

        self.model_list = {
            "64": {
                "resolution": 64,
                "geo": self.geo64,
                "color": self.color64,
                "encoding": self.encoding1,
                "delta": 0.25 / 64,
                "numerical_grad_type": "numerical",
                "interpolation_method": "full_grid"
            },
            "128": {
                "resolution": 128,
                "geo": self.geo128,
                "color": self.color128,
                "encoding": self.encoding2,
                "delta": 0.25 / 128,
                "numerical_grad_type": "numerical",
                "interpolation_method": "full_grid"
            },
            "256": {
                "resolution": 256,
                "geo": self.geo256,
                "color": self.color256,
                "encoding": self.encoding3,
                "delta": 0.25 / 256,
                "numerical_grad_type": "numerical",
                "interpolation_method": "full_grid"
            },
            "inf": {
                "resolution": None,
                "geo": self.geoinf,
                "color": self.colorinf,
                "encoding": self.encoding4,
                "delta": 0.25 / 2048,
                "numerical_grad_type": "numerical",
                "interpolation_method": "none"
            },

        }

        self.use_numerical_gradients = self.config.use_numerical_gradients

        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        # self.laplace_density = SigmoidDensity(init_val=self.config.beta_init)

        # TODO use different name for beta_init for config
        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = SingleVarianceNetwork(init_val=self.config.beta_init)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0

    def create_hash_encoding(self, num_levels, features_per_level, log2_hashmap_size, base_res, growth_factor, smoothstep):
        encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
                "interpolation": "Smoothstep" if smoothstep else "Linear",
            },
        )
        return encoding
    
    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def update_mask(self, level: int):
        self.hash_encoding_mask[:] = 1.0
        self.hash_encoding_mask[level * self.features_per_level:] = 0
    
    def transfer_weights(self, source, destination):
        self.model_list[destination]["color"].load_state_dict(self.model_list[source]["color"].state_dict())
        self.model_list[destination]["encoding"].load_state_dict(self.model_list[source]["encoding"].state_dict())
    
    def get_nngrid(self, resolution, encoding, forward_geo_mlp):
        if str(resolution) not in self.grid_sdfs_dict.keys():
            grid_pts = torch.meshgrid(
                torch.linspace(-1, 1, resolution), #32
                torch.linspace(-1, 1, resolution),
                torch.linspace(-1, 1, resolution),
            )
            grid_pts = torch.stack(grid_pts, dim=-1).view(-1, 3).to(self.aabb.device)
            pe = self.position_encoding(grid_pts)
            grid_pts_norm = (grid_pts + 1.0) / 2.0
            feature = encoding(grid_pts_norm) # [N, 32]

            grid_pts = torch.cat((grid_pts, pe, feature), dim=-1)    
            x = forward_geo_mlp(grid_pts) # [N^3, feature]

            grid_sdfs = x.T.view(-1, resolution, resolution, resolution)[None, ...] # [1, n_feat, res, res ,res]
            self.grid_sdfs_dict[str(resolution)] = grid_sdfs
            return grid_sdfs
        
        else:
            # Cashing the grid
            return self.grid_sdfs_dict[str(resolution)]
    

    def get__nn_grid_with_pts(self, positions, resolution, encoding, forward_geo_mlp, eval=False):
        num_coords, coord_dim = positions.shape

        x = torch.clamp(resolution * positions, 0.0, float(resolution)-1-1e-1)
        pos = torch.floor(x).long()
        x_ = x - pos
        _x = 1.0 - x_

        coeffs = torch.empty([num_coords, 8], device=positions.device)
        coeffs[:, 0] = _x[:, 0] * _x[:, 1] * _x[:, 2]
        coeffs[:, 1] = _x[:, 0] * _x[:, 1] * x_[:, 2]
        coeffs[:, 2] = _x[:, 0] * x_[:, 1] * _x[:, 2]
        coeffs[:, 3] = _x[:, 0] * x_[:, 1] * x_[:, 2]
        coeffs[:, 4] = x_[:, 0] * _x[:, 1] * _x[:, 2]
        coeffs[:, 5] = x_[:, 0] * _x[:, 1] * x_[:, 2]
        coeffs[:, 6] = x_[:, 0] * x_[:, 1] * _x[:, 2]
        coeffs[:, 7] = x_[:, 0] * x_[:, 1] * x_[:, 2]

        grid_pts = torch.empty([num_coords, 8, coord_dim], device=positions.device).long()
        for k in range(8):
            grid_pts[:, k, 0] = pos[:, 0] + ((k & 4) >> 2)
            grid_pts[:, k, 1] = pos[:, 1] + ((k & 2) >> 1)
            grid_pts[:, k, 2] = pos[:, 2] + ((k & 1) >> 0)

        grid_pts = grid_pts.reshape(-1, coord_dim) / resolution
        if eval:
            with torch.no_grad():
                feature = encoding(grid_pts) # [N, 32]

                grid_pts = 2 * grid_pts - 1
                pe = self.position_encoding(grid_pts)

                grid_pts = torch.cat((grid_pts, pe, feature), dim=-1)    
                grid_pts_out = forward_geo_mlp(grid_pts)
        else:
            feature = encoding(grid_pts) # [N, 32]

            grid_pts = 2 * grid_pts - 1
            pe = self.position_encoding(grid_pts)

            grid_pts = torch.cat((grid_pts, pe, feature), dim=-1)    
            grid_pts_out = forward_geo_mlp(grid_pts)

        grid_pts_out = grid_pts_out.reshape(num_coords, 8, -1)

        sdf_out = torch.sum(grid_pts_out * coeffs.unsqueeze(-1), dim=1)
        return sdf_out
    
    def single_geo_forward(self, res_mode, inputs, cur_model_config, no_grad_mode=False):
        encoding = cur_model_config["encoding"]
        forward_geo_mlp = cur_model_config["geo"]
        # delta = cur_model_config["delta"]
        # numerical_grad_type = cur_model_config["numerical_grad_type"]
        interpolation_method = cur_model_config["interpolation_method"]
        res = cur_model_config["resolution"]

        if res_mode == "inf":
            non_scale_pts = inputs * 2 - 1
            feature = encoding(inputs)
            pe = self.position_encoding(non_scale_pts)
            non_scale_pts = torch.cat((non_scale_pts, pe, feature), dim=-1)
            x = self.geoinf(non_scale_pts)
            sdf_inf, geo_feature_inf = torch.split(x, [1, self.config.geo_feat_dim], dim=-1)
            return sdf_inf, geo_feature_inf

        if interpolation_method == "full_grid":
            if no_grad_mode:
                with torch.no_grad():
                    grid_sdfs = self.get_nngrid(res, encoding, forward_geo_mlp)
            else:
                grid_sdfs = self.get_nngrid(res, encoding, forward_geo_mlp)
            x = interpolate_grid(grid_sdfs, inputs) 

        elif interpolation_method == "smart_indexing":
            x = self.get__nn_grid_with_pts(inputs, res, encoding, forward_geo_mlp, eval=no_grad_mode)

        sdf, geo_feature = torch.split(x, [1, self.config.geo_feat_dim], dim=-1)
        return sdf, geo_feature

    def forward_geonetwork(self, inputs):
        """forward the geonetwork"""
        #TODO normalize inputs depending on the whether we model the background or not
        positions = (inputs + 1.0) / 2.0

        assert self.config.cur_mode in self.model_list

        final_sdf = 0
        final_geo_feature = []

        for res_mode, params in self.model_list.items():
            if res_mode != "inf" and self.config.cur_mode != "inf":
                assert int(res_mode) <= int(self.config.cur_mode)
                assert params["resolution"] <= int(self.config.cur_mode)
            
            sdf, geo_feat = self.single_geo_forward(
                res_mode,
                positions,
                params,
                no_grad_mode=self.config.cur_mode != res_mode
            )

            final_sdf += sdf
            final_geo_feature += [geo_feat]

            if self.config.cur_mode == res_mode:
                break
        
        return final_sdf, final_geo_feature 


    def get_sdf(self, ray_samples: RaySamples):
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)

        sdf, _ = self.forward_geonetwork(positions_flat)
        sdf = sdf.view(*ray_samples.frustums.shape, -1)
        # sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def sample_pts(self, x, delta):
        points = torch.stack(
            [
                x + torch.as_tensor([delta, 0.0, 0.0]).to(x),
                x + torch.as_tensor([-delta, 0.0, 0.0]).to(x),
                x + torch.as_tensor([0.0, delta, 0.0]).to(x),
                x + torch.as_tensor([0.0, -delta, 0.0]).to(x),
                x + torch.as_tensor([0.0, 0.0, delta]).to(x),
                x + torch.as_tensor([0.0, 0.0, -delta]).to(x),
                x
            ],
            dim=0,
        )
        return points
    
    def get_grad_and_lapl(self, points_sdf, delta, compute_laplacian_loss):
        gradients = torch.stack(
            [
                0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                0.5 * (points_sdf[4] - points_sdf[5]) / delta,
            ],
            dim=-1,
        )

        if compute_laplacian_loss:
            hessian = torch.stack(
                [
                    (points_sdf[0] + points_sdf[1] - 2 * points_sdf[-1]) / delta**2,
                    (points_sdf[2] + points_sdf[3] - 2 * points_sdf[-1]) / delta**2,
                    (points_sdf[4] + points_sdf[5] - 2 * points_sdf[-1]) / delta**2,
                ],
                dim=-1,
            )
            lapl = hessian.sum(dim=-1)**2
            lapl = lapl.mean()
        else:
            lapl = None
        
        return gradients, lapl

    def gradient(self, x, skip_spatial_distortion=False):
        """compute the gradient of the ray"""
        delta = self.model_list[self.config.cur_mode]["delta"]
        points = self.sample_pts(x, delta)
        sdf, _ = self.forward_geonetwork(points.view(-1, 3))
        sdf = sdf[..., 0].view(7, *x.shape[:-1])
        gradients, lapl = self.get_grad_and_lapl(sdf, delta, self.compute_laplacian_loss)

        return gradients, sdf, lapl

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)

        sdf, geo_feature = self.forward_geonetwork(positions_flat)
        sdf = sdf.view(*ray_samples.frustums.shape, -1)
        # sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        density = self.laplace_density(sdf)
        return density, geo_feature

    def get_alpha(self, ray_samples: RaySamples, sdf=None, gradients=None):
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            assert 0

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        # HF-NeuS
        # # sigma
        # cdf = torch.sigmoid(sdf * inv_s)
        # e = inv_s * (1 - cdf) * (-iter_cos) * ray_samples.deltas
        # alpha = (1 - torch.exp(-e)).clip(0.0, 1.0)

        return alpha

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy
    
    def single_color_forward(self, inputs, cur_h, params):
        if self.config.color_continuous:
            codebook_encoding = params["encoding"](inputs)
            cur_h = cur_h + [codebook_encoding.view(-1, params["encoding"].n_output_dims)]
        cur_h = params['color'](torch.cat(cur_h, dim=-1))
        return cur_h

    def get_colors(self, points, directions, gradients, geo_features, camera_indices):
        """compute colors"""

        # normals = F.normalize(gradients, p=2, dim=-1)
        assert self.config.cur_mode in self.model_list.keys()

        d = self.direction_encoding(directions)
        h = [
            points,
            d,
            gradients,
        ]
        inputs = (points + 1.0) / 2.0

        params = self.model_list[self.config.cur_mode]
        h = h + [geo_features[-1].view(-1, self.config.geo_feat_dim)]
        h = self.single_color_forward(inputs, h, params)

        rgb = self.sigmoid(h)
        rgb = rgb * (1 + 2 * self.config.rgb_padding) - self.config.rgb_padding
        
        return rgb

    def get_outputs(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False):
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)
        # save_points("get_outputs.ply", inputs.detach().cpu().numpy())

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        points_norm = inputs.norm(dim=-1)
        # compute gradient in constracted space
        inputs.requires_grad_(True)
        with torch.enable_grad():
            sdf, geo_feature = self.forward_geonetwork(inputs)

        # if self.use_numerical_gradients:
        if self.model_list[self.config.cur_mode]["numerical_grad_type"] == "numerical":
            gradients, sampled_sdf, lapl = self.gradient(
                inputs,
                skip_spatial_distortion=True,
            )
            sampled_sdf = sampled_sdf.view(-1, *ray_samples.frustums.directions.shape[:-1]).permute(1, 2, 0).contiguous()
        else:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            sampled_sdf = None
            lapl = None

        rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature, camera_indices)

        density = self.laplace_density(sdf)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = F.normalize(gradients, p=2, dim=-1)
        points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)
        
        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.DENSITY: density,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMAL: normals,
                FieldHeadNames.GRADIENT: gradients,
                "points_norm": points_norm,
                "sampled_sdf": sampled_sdf,
                "laplacian": lapl,
            }
        )

        if return_alphas:
            # TODO use mid point sdf for NeuS
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        return outputs

    def forward(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        self.grid_sdfs_dict = {}
        field_outputs = self.get_outputs(ray_samples, return_alphas=return_alphas, return_occupancy=return_occupancy)
        return field_outputs

def interpolate_grid(grid, in_tensor):
    num_feat = grid.shape[1]
    shape = in_tensor.shape[:-1]
    in_tensor = in_tensor.reshape(1, 1, 1, -1, 3)
    ind_norm = in_tensor.flip((-1,)) * 2 - 1
    out = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
    out = out.reshape(num_feat,-1).T.reshape(*shape,num_feat)

    return out
