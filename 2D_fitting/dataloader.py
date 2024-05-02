import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from skimage.transform import resize
from imageio import imread, imsave
from PIL import Image
import numpy as np

def create_coords(h, w):
    # sampling regularly
    grid_y, grid_x = torch.meshgrid(
        [torch.arange(0, 1, 1/h), torch.arange(0, 1, 1/w)]
    )
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid

def sample_random_points(img, num_samples):
    # sample randomly in the image
    h, w, c = img.shape
    coords = torch.rand((num_samples, num_samples, 2))
    gt = F.grid_sample(img.permute(2,1,0).unsqueeze(0), coords.unsqueeze(0).float() * 2 - 1, mode='bilinear', align_corners=True)

    return coords, gt.squeeze(0).permute(1,2,0)

class ImageDataset(Dataset):
    def __init__(self, image_path, img_dim, trainset_size, pts_per_sample, continuous_sampling):
        self.trainset_size = trainset_size
        self.img_dim = (img_dim, img_dim) if isinstance(img_dim, int) else img_dim
        self.continuous_sampling = continuous_sampling
        self.pts_per_sample = pts_per_sample

        image = Image.open(image_path)
        w, h = image.size
        if (w != self.img_dim[0]) or (h != self.img_dim[1]):
            crop_size = min(w, h)
            w_start = int(0.5*(w - crop_size))
            h_start = int(0.5*(h - crop_size))
            image = image.crop((w_start, h_start, w_start+crop_size, h_start+crop_size))
            image = image.resize(self.img_dim, Image.BILINEAR) #Image.ANTIALIAS Image.BILINEAR
        image = np.array(image)

        image = image.astype("float64") / 255

        self.coords = create_coords(image.shape[0], image.shape[1])

        self.img = torch.from_numpy(image).float()

    def __getitem__(self, idx):
        if self.continuous_sampling:
            coords, gt = sample_random_points(self.img, self.pts_per_sample)
            return coords, gt
        else:
            return self.coords, self.img

    def __len__(self):
        return self.trainset_size

def get_loader(image_path, img_dim, trainset_size, pts_per_sample=None, continuous_sampling=False):
    return DataLoader(
        ImageDataset(image_path, img_dim, trainset_size, pts_per_sample, continuous_sampling),
        batch_size=1,
        num_workers=0,
    )