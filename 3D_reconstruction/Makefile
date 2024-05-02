# share same shell, allows multiline, stop on error
.ONESHELL:
# which shell to share
SHELL = /bin/bash
# terminate on first error within a recipe
.SHELLFLAGS += -e

# export DATA_PATH="data/nerf_dataset/"
# export MODEL_DIR="outputs/data-nerf_dataset-lego/banf/"
# export RES=64

train: 
	python train.py \
		--pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 128 \
		--pipeline.model.sdf-field.hidden-dim-color 128 --pipeline.model.sdf-field.num-layers 2 \
		--pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding False \
		--pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False \
		--pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.beta-init 0.3 \
		--trainer.max_num_iterations 60000 --pipeline.datamanager.train_num_rays_per_batch 5000 \
		--trainer.steps_per_save 1000 --trainer.steps_per_eval_image 1000 --trainer.steps_per_eval_batch 60000 \
		--pipeline.model.background-model none --pipeline.model.background_color white \
		--pipeline.model.sdf-field.color-continuous True \
		--pipeline.model.sdf-field.geo-feat-dim 16 \
		--vis tensorboard \
		--pipeline.model.sdf-field.use-numerical-gradients True \
		--pipeline.model.eikonal_loss_mult 0.01 \
		--pipeline.model.laplacian_loss_mult 1e-7 \
		--pipeline.model.fixeds False --pipeline.model.big_delta False \
		--data "${DATA_PATH}/lego"

meshing:
	python mesh_extractor.py \
		--load-config "${MODEL_DIR}"/config.yml \
		--output-path "${MODEL_DIR}"/mesh_"${RES}".ply \
		--create_visibility_mask False \
		--is_occupancy False \
		--resolution ${RES} \
		--cur_mode ${RES}