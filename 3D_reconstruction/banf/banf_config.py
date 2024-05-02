### This file is used to define the configuration of the BANF method.
### The structure of the configuration is similar to the one used in the SDFStudio repository.
from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import (
    Config,
    SchedulerConfig,
    TrainerConfig,
    ViewerConfig,
)
from nerfstudio.engine.schedulers import NeuSSchedulerConfig
from banf.banf_optim import NeuSSchedulerConfig

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from banf.blender_parser import BlenderDataParserConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from banf.banf_model import BanfModelConfig

method_configs: Dict[str, Config] = {}

descriptions = {
    "banf": "method proposed in the BANF paper",
}

method_configs["banf"] = Config(
    method_name="banf",
    trainer=TrainerConfig(
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=100000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=BanfModelConfig(eval_num_rays_per_chunk=1024),
    ),
    optimizers={
        "model64": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_start=0, warm_up_end=2000, learning_rate_alpha=0.05, max_steps=50000, end=20000),
        },
        "model128": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_start=20000, warm_up_end=22000, learning_rate_alpha=0.05, max_steps=80000, end=40000),
        },
        "model256": {
            "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_start=40000, warm_up_end=42000, learning_rate_alpha=0.05, max_steps=100000, end=60000),
        },
        "modelinf": {
            "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_start=60000, warm_up_end=62000, learning_rate_alpha=0.05, max_steps=120000, end=80000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_start=0, warm_up_end=5000, learning_rate_alpha=0.05, max_steps=100000, end=80000),
        },
        "other": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_start=0, warm_up_end=5000, learning_rate_alpha=0.05, max_steps=100000, end=80000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]