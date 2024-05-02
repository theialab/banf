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
from interpolate import linear_interpolation
import torch.nn.functional as F
import fire


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
    
    def linear_interpolate_grid(self, grid, coords):
        in_shape = coords.shape[0]
        grid = grid.permute(2, 3, 1, 0).squeeze().view(-1, 3)
        coords = coords.view(-1, 2)

        out = linear_interpolation(coords, self.grid_res, grid)
        out = out.reshape(in_shape, in_shape, 3)
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
            vals = self.linear_interpolate_grid(nn_grid, coords)
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


def main(image_id):
    # Set training params
    folder = "data_triangle"
    data_path = f"{folder}/{image_id}/gt_256.png"
    output_path = f"output_linear_kernel/{image_id}"
    num_imgs_per_iter = 2
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/resolution", exist_ok=True)
    os.makedirs(f"{output_path}/summation", exist_ok=True)
    os.makedirs(f"{output_path}/error", exist_ok=True)

    device = "cpu"

    APPLY_BLUR = False
    max_epochs = 900
    SECOND_START = 150
    THIRD_START = 450

    gt64 = torch.Tensor(np.array(Image.open(f"{folder}/{image_id}/gt_64.png"))/255)
    gt128 = torch.Tensor(np.array(Image.open(f"{folder}/{image_id}/gt_128.png"))/255)
    gt256 = torch.Tensor(np.array(Image.open(f"{folder}/{image_id}/gt_256.png"))/255)

    val_data_loader64 = get_loader(data_path, 64, num_imgs_per_iter)
    val_data_loader128 = get_loader(data_path, 128, num_imgs_per_iter)
    val_data_loader256 = get_loader(data_path, 256, num_imgs_per_iter) # sampling fixed grid points of image
    val_data_loader512 = get_loader(data_path, 512, num_imgs_per_iter)

    train_data_loader64 = get_loader(data_path, 256, num_imgs_per_iter, 128, continuous_sampling=False)
    train_data_loader128 = get_loader(data_path, 256, num_imgs_per_iter, 128, continuous_sampling=False)
    train_data_loader256 = get_loader(data_path, 256, num_imgs_per_iter, 128, continuous_sampling=False) # sampling randomly and conitnuous over image

    smart_grid = DenseGrid(max_res=64, num_lod=5, interpolation_type="bilinear", zero_init=False)  # "closest" or "bilinear"
    model1 = MLP_grid_wrapper(MLP(smart_grid, 10, 32, 3, 2), 64)
    smart_grid = DenseGrid(max_res=128, num_lod=5, interpolation_type="bilinear", zero_init=True)  # "closest" or "bilinear"
    model2 = MLP_grid_wrapper(MLP(smart_grid, 10, 32, 3, 2), 128)
    smart_grid = DenseGrid(max_res=256, num_lod=5, interpolation_type="bilinear", zero_init=True)  # "closest" or "bilinear"
    model3 = MLP_grid_wrapper(MLP(smart_grid, 10, 32, 3, 2), 256)
    # model3 = MLP(DenseGrid(interpolation_type="bilinear", zero_init=True), 8, 32, 3)

    model = MixMode(model1, model2, model3).to(device)
    print("TOTAL NUM OF PARAMS:", sum(p.numel() for p in model.parameters()))
    loss_fn = torch.nn.HuberLoss()
    optimizer1 = torch.optim.RMSprop(model.model1.parameters(), lr=1e-3)
    optimizer2 = torch.optim.RMSprop(model.model2.parameters(), lr=1e-3)
    optimizer3 = torch.optim.RMSprop(model.model3.parameters(), lr=5e-4)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[8*SECOND_START//10], gamma=0.33)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[8*(THIRD_START-SECOND_START)//10], gamma=0.33)
    scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer3, milestones=[8*(max_epochs-THIRD_START)//10], gamma=0.33)

    model.blur = APPLY_BLUR
    loop = tqdm(range(max_epochs))
    for epoch in loop:
        for val_data, val_data64, val_data128, val_data_512, train_data, train_data64, train_data128 in zip(val_data_loader256, val_data_loader64, val_data_loader128, val_data_loader512, train_data_loader256, train_data_loader64, train_data_loader128):
            coords256, values256 = train_data[0][0].to(device), train_data[1][0].to(device)
            coords128, values128 = train_data128[0][0].to(device), train_data128[1][0].to(device)
            coords64, values64 = train_data64[0][0].to(device), train_data64[1][0].to(device)

            if epoch <= SECOND_START:
                optimizer1.zero_grad()
                output64 = model.pred1(coords256)

                loss = loss_fn(output64, values256)
                loss.backward()
                optimizer1.step()
                scheduler1.step(loss.item())

            elif epoch > SECOND_START and epoch <= THIRD_START:
                # break
                with torch.no_grad():
                    output1 = model.pred1(coords256)
                output2 = model.pred2(coords256)
                output128 = (output1 + output2)

                loss = loss_fn(output128, values256)
                loss.backward()
                optimizer2.step()
                scheduler2.step(loss.item())

            else:
                with torch.no_grad():
                    output1 = model.pred1(coords256)
                    output2 = model.pred2(coords256)
                output3 = model.pred3(coords256, interpolate=False)
                output256 = (output1 + output2 + output3)

                loss = loss_fn(output256, values256)
                loss.backward()
                optimizer3.step()
                scheduler3.step(loss.item())

            with torch.no_grad():
                coords512, _ = val_data_512[0][0].to(device), val_data_512[1][0].to(device)
                coords256, _ = val_data[0][0].to(device), val_data[1][0].to(device)
                coords128, _ = val_data128[0][0].to(device), val_data128[1][0].to(device)
                coords64, _ = val_data64[0][0].to(device), val_data64[1][0].to(device)

                if epoch > max_epochs-2:
                    output512_1 = model.pred1(coords512)
                    output512_2 = model.pred2(coords512)
                    output512_3 = model.pred3(coords512)
                    if epoch > SECOND_START and epoch <= THIRD_START:
                        output512 = (output512_1 + output512_2)
                    elif epoch > THIRD_START:
                        output512 = (output512_1 + output512_2 + output512_3)
                    else:
                        output512 = output512_1

                output256_1 = model.pred1(coords256)
                output256_2 = model.pred2(coords256)
                output256_3 = model.pred3(coords256, interpolate=False)
                if epoch > SECOND_START and epoch <= THIRD_START:
                    output256 = (output256_1 + output256_2)
                elif epoch > THIRD_START:
                    output256 = (output256_1 + output256_2 + output256_3)
                else:
                    output256 = output256_1

                output128_1 = model.pred1(coords128)
                output128_2 = model.pred2(coords128, interpolate=False)
                if epoch > SECOND_START:
                    output128 = (output128_1 + output128_2)
                else:
                    output128 = output128_1

                output64 = model.pred1(coords64, interpolate=False)

                def norm_error(pred, gt):
                    error = torch.abs(pred.cpu() - gt.cpu())
                    return (error - error.min())/(error.max() - error.min())

                psnr64 = psnr_score(output64, gt64).item()
                error64 = norm_error(output64, gt64)
                psnr128 = psnr_score(output128, gt128).item()
                error128 = norm_error(output128, gt128)
                psnr256 = psnr_score(output256, gt256).item()
                error256 = norm_error(output256, gt256)
                
                loop.set_description(f"Epoch: {epoch}")
                loop.set_postfix_str(
                    f"PSNR64: {psnr64:.2f} | PSNR128: {psnr128:.2f} | PSNR256: {psnr256:.2f} | Loss: {loss.item():.5f}" #  | PSNR512: {psnr512:.2f}
                )

                visualize(
                    [output256_1.clamp(0, 1), (abs(output256_2)).clamp(0, 1), (abs(output256_3)).clamp(0, 1), output256.clamp(0, 1), values256],
                    names=["Low_Freq", "Med_freq", "High_freq", "Low+Med+High", "GT_High"], 
                    save_path=f"vis_out/{epoch:05d}.png")


    with open(f"{output_path}/psnr_triangle.txt", "w") as f:
        f.write(f"PNSR-64: {psnr64:.3f} \nPNSR-128: {psnr128:.3f} \nPNSR-256: {psnr256:.3f}")
        f.close()

    imsave(f"{output_path}/resolution/64_pred.png", (255*output64.cpu().clamp(0, 1)).type(torch.uint8))
    imsave(f"{output_path}/resolution/128_pred.png", (255*output128.cpu().clamp(0, 1)).type(torch.uint8))
    imsave(f"{output_path}/resolution/256_pred.png", (255*output256.cpu().clamp(0, 1)).type(torch.uint8))
    imsave(f"{output_path}/resolution/512_pred.png", (255*output512.cpu().clamp(0, 1)).type(torch.uint8))

    imsave(f"{output_path}/error/64_pred.png", (255*error64.cpu().clamp(0, 1)).type(torch.uint8))
    imsave(f"{output_path}/error/128_pred.png", (255*error128.cpu().clamp(0, 1)).type(torch.uint8))
    imsave(f"{output_path}/error/256_pred.png", (255*error256.cpu().clamp(0, 1)).type(torch.uint8))

    imsave(f"{output_path}/summation/low.png", (255*output512_1.cpu().clamp(0, 1)).type(torch.uint8))
    imsave(f"{output_path}/summation/med.png", (255*(output512_1+output512_2).cpu().clamp(0, 1)).type(torch.uint8))
    imsave(f"{output_path}/summation/high.png", (255*(output512_1+output512_2+output512_3).cpu().clamp(0, 1)).type(torch.uint8))

    imsave(f"{output_path}/summation/low_freq.png", (255*get_fft(output512_1.cpu(), 250)).astype(np.uint8))
    imsave(f"{output_path}/summation/med_freq.png", (255*get_fft((output512_1+output512_2).cpu(), 250)).astype(np.uint8))
    imsave(f"{output_path}/summation/high_freq.png", (255*get_fft((output512_1+output512_2+output512_3).cpu(), 125)).astype(np.uint8))

if __name__ == "__main__":
    fire.Fire(main)
