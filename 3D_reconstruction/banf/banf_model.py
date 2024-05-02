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
Implementation of NeuS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import numpy as np

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import NeuSSampler, save_points
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
# from banf.base_surface import SurfaceModel, SurfaceModelConfig
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from banf.banf_field import BanfFieldConfig
import torch
from torch.nn import Parameter

import torch.nn.functional as F


from arrgh import arrgh


@dataclass
class BanfModelConfig(SurfaceModelConfig):
    """Banf Model Config"""

    _target: Type = field(default_factory=lambda: BanfModel)
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    num_up_sample_steps: int = 4
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    base_variance: float = 64
    """fixed base variance in NeuS sampler, the inv_s will be base * 2 ** iter during upsample"""
    perturb: bool = True
    """use to use perturb for the sampled points"""
    laplacian_loss_mult: float = 0.0
    """fix max s value"""
    max_s: float = None #980.0
    """fixed s value"""
    fixeds: bool = False
    delta_val: float = 0.000125
    big_delta: bool = False
    use_mask: bool = False
    sdf_field: BanfFieldConfig = BanfFieldConfig()


class BanfModel(SurfaceModel):
    """Banf model

    Args:
        config: Banf configuration to instantiate model
    """

    config: BanfModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = NeuSSampler(
            num_samples=self.config.num_samples,
            num_samples_importance=self.config.num_samples_importance,
            num_samples_outside=self.config.num_samples_outside,
            num_upsample_steps=self.config.num_up_sample_steps,
            base_variance=self.config.base_variance,
        )

        self.anneal_end = 50000
        self.max_s = self.config.max_s

        self.collider = AABBBoxCollider(self.scene_box, near_plane=self.scene_box.near)
        if self.config.laplacian_loss_mult > 0:
            self.field.compute_laplacian_loss = True
            self.field.config.compute_laplacian_loss = True
        else:
            self.field.compute_laplacian_loss = False
            self.field.config.compute_laplacian_loss = False
        
        if self.config.use_mask > 0:
            print('USING MASKS FOR TRAINING')


    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)
        # anneal for cos in NeuS
        if self.anneal_end > 0:

            def set_anneal(step):
                anneal = min([1.0, step / self.anneal_end])
                self.field.set_cos_anneal_ratio(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
        
        def set_train_mode(step):
            self.config.laplacian_loss_mult *= 0.9999
            if self.config.big_delta:
                self.field.model_list['64']['delta'] = 2 / 64
                self.field.model_list['128']['delta'] = 2 / 128
                self.field.model_list['256']['delta'] = 2 / 256
                self.field.model_list['inf']['delta'] = 2 / 2048
            else:
                self.field.model_list['64']['delta'] = 0.25 / 64
                self.field.model_list['128']['delta'] = 0.25 / 128
                self.field.model_list['256']['delta'] = 0.25 / 256
                self.field.model_list['inf']['delta'] = 0.25 / 2048

            
            ### Define transition steps
            transition_steps = {
                "64": 20000,
                "128": 40000,
                "256": 60000,
            }

            ### Coarse to fine training
            if step < 3000:
                self.field.model_list['64']['resolution'] = 16
            elif step < 8000:
                self.field.model_list['64']['resolution'] = 32
            elif step < 20000:
                self.field.model_list['64']['resolution'] = 64

            ### Tranfer initilization when starting training next resolution
            if step == transition_steps["64"]:
                self.field.transfer_weights("64", "128")
            if step == transition_steps["128"]:
                self.field.transfer_weights("128", "256")
            if step == transition_steps["256"]:
                self.field.transfer_weights("256", "inf")
            
            ### Switching to next resolution based on current step
            if step < transition_steps["64"]:
                self.field.config.cur_mode = "64"
            elif step < transition_steps["128"]:
                self.field.config.cur_mode = "128"
            elif step < transition_steps["256"]:
                self.field.config.cur_mode = "256"
            else:
                self.field.config.cur_mode = "inf"
        
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_train_mode,
            )
        )

        def set_sval(step):
            beta = np.log((self.max_s - 20) * np.tanh(2 * step / 60000) + 20) / 10
            self.field.deviation_network.variance.data[...] = beta

        if self.config.fixeds:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_sval,
                )
            )

        return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        # save_points("start.ply", ray_samples.frustums.get_start_positions().reshape(-1, 3).detach().cpu().numpy())
        field_outputs = self.field(ray_samples, return_alphas=True)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
            "laplacian": field_outputs['laplacian'],
        }
        return samples_and_field_outputs

    def get_metrics_dict_parent(self, outputs, batch, mask=False) -> Dict:
        metrics_dict = {}
        image = batch["image"].to(self.device)
        if mask:
            mask = batch["fg_mask"].to(self.device).long()
            metrics_dict["psnr"] = self.psnr(outputs["rgb"] * mask, image * mask)
        else:
            metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict
    
    def get_metrics_dict(self, outputs, batch) -> Dict:
        have_mask = "fg_mask" in batch and self.config.use_mask
        metrics_dict = self.get_metrics_dict_parent(outputs, batch, mask=have_mask)
        if self.training:
            # training statics
            metrics_dict["s_val"] = self.field.deviation_network.get_variance().item()
            metrics_dict["inv_s"] = 1.0 / self.field.deviation_network.get_variance().item()

        return metrics_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups = {}
        param_groups["model64"] = list(self.field.geo64.parameters()) + list(self.field.encoding1.parameters()) + list(self.field.color64.parameters())
        param_groups["model128"] = list(self.field.geo128.parameters()) + list(self.field.encoding2.parameters()) + list(self.field.color128.parameters())
        param_groups["model256"] = list(self.field.geo256.parameters()) + list(self.field.encoding3.parameters()) + list(self.field.color256.parameters())
        param_groups["modelinf"] = list(self.field.geoinf.parameters()) + list(self.field.encoding4.parameters()) + list(self.field.colorinf.parameters())
        other_list = []
        for k, v in self.field.named_parameters():
            if ('geo' in k or 'encoding' in k or 'color' in k):
                continue
            else:
                other_list.append(v)
        param_groups["other"] = other_list
        assert len(list(self.field.parameters())) == (len(param_groups["model64"]) + len(param_groups["model128"]) + len(param_groups["model256"]) + len(param_groups["modelinf"]) + len(param_groups["other"]))
        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        else:
            param_groups["field_background"] = list(self.field_background)
        return param_groups
    
    def forward(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        self.field.grid_sdfs_dict = {}

        return self.get_outputs(ray_bundle)
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

            if self.config.laplacian_loss_mult > 0:
                loss_dict["laplacian_loss"] = outputs["laplacian"] * self.config.laplacian_loss_mult
            
            if "fg_mask" in batch and self.config.use_mask:
                final_gt = image * batch["fg_mask"].float().to(self.device)
                loss_dict["rgb_loss"] = self.rgb_loss(final_gt, outputs["rgb"])

                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )

        return loss_dict