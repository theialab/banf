from arrgh import arrgh
import torch
import numpy as np
import cv2
import os
# import matplotlib.pyplot as plt


def lanczos_interpolation(positions, resolution, grid_vals):
    num_coords, coord_dim = positions.shape

    x = resolution * positions
    pos = torch.floor(x).long()

    add_corners = torch.tensor([[-1, -1], [0, -1], [1, -1], [2, -1],
                                [-1, 0], [0, 0], [1, 0], [2, 0], 
                                [-1, 1], [0, 1], [1, 1], [2, 1],
                                [-1, 2], [0, 2], [1, 2], [2, 2]]).to(positions.device)
    corners = pos.unsqueeze(1).repeat(1, 16, 1) + add_corners
    out_of_bound = (corners > resolution-1) + (corners < 0)
    in_bound = 1 - (out_of_bound[..., 0] + out_of_bound[..., 1]).type(torch.float)
    in_bound = in_bound.view(num_coords, 16)

    difference = x.view(num_coords, 1, coord_dim) - corners
    lanczos_weights = torch.sinc(difference[..., 0])*torch.sinc(difference[..., 0]/2)
    lanczos_weights *= torch.sinc(difference[..., 1])*torch.sinc(difference[..., 1]/2)
    summ = lanczos_weights.sum(dim=1, keepdim=True)

    corners = torch.clamp(corners, 0, resolution-1)
    corner_idxes = corners[..., 0] * resolution + corners[..., 1]
    corner_idxes = corner_idxes.view(-1)

    corner_grid_vals = torch.index_select(grid_vals, dim=0, index=corner_idxes)
    corner_grid_vals = corner_grid_vals.view(num_coords, 16, 3)

    final_vals = torch.sum(corner_grid_vals * lanczos_weights.unsqueeze(-1), dim=1)
    
    return final_vals


def show_model_stats(model):
    print(model)
    print("Num of params:", sum([param.nelement() for param in model.parameters()]))
    print(
        "Num of params require grad:",
        sum(
            [
                param.nelement()
                for param in model.parameters()
                if param.requires_grad is True
            ]
        ),
    )


def psnr_score(img1, img2):
    img1, img2 = img1.cpu(), img2.cpu()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def visualize(imgs, names=None, save_path=None):

    for idx, im in enumerate(imgs):
        if idx == 0:
            res = im.detach().cpu().numpy()
        else:
            res = np.hstack((res, im.detach().cpu().numpy()))

    res = (res * 255).astype("uint8")
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    
    if names is not None:
        assert len(names) == len(imgs)
        for idx, name in enumerate(names):
            cv2.putText(
                res,
                name,
                (30 + idx * 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, res)
        return

    cv2.imshow("Result", res)
    cv2.waitKey(1)

def get_fft(image, factor):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum = magnitude_spectrum.clip(0, factor)/factor
    # arrgh(magnitude_spectrum)
    return magnitude_spectrum


def bilinear_interpolation(res, grid, points, grid_type):
    """
    Performs bilinear interpolation of points with respect to a grid.

    Parameters:
        grid (numpy.ndarray): A 2D numpy array representing the grid.
        points (numpy.ndarray): A 2D numpy array of shape (n, 2) representing
            the points to interpolate.

    Returns:
        numpy.ndarray: A 1D numpy array of shape (n,) representing the interpolated
            values at the given points.
    """
    PRIMES = [1, 265443567, 805459861]

    # Get the dimensions of the grid
    grid_size, feat_size = grid.shape
    points = points[None]
    _, N, _ = points.shape
    # Get the x and y coordinates of the four nearest points for each input point
    x = points[:, :, 0] * (res - 1)
    y = points[:, :, 1] * (res - 1)

    x1 = torch.floor(torch.clip(x, 0, res - 1 - 1e-5)).int()
    y1 = torch.floor(torch.clip(y, 0, res - 1 - 1e-5)).int()

    x2 = torch.clip(x1 + 1, 0, res - 1).int()
    y2 = torch.clip(y1 + 1, 0, res - 1).int()

    # Compute the weights for each of the four points
    w1 = (x2 - x) * (y2 - y)
    w2 = (x - x1) * (y2 - y)
    w3 = (x2 - x) * (y - y1)
    w4 = (x - x1) * (y - y1)

    if grid_type == "NGLOD":
        # Interpolate the values for each point
        id1 = (x1 + y1 * res).long()
        id2 = (y1 * res + x2).long()
        id3 = (y2 * res + x1).long()
        id4 = (y2 * res + x2).long()

    elif grid_type == "HASH":
        npts = res**2
        if npts > grid_size:
            id1 = ((x1 * PRIMES[0]) ^ (y1 * PRIMES[1])) % grid_size
            id2 = ((x2 * PRIMES[0]) ^ (y1 * PRIMES[1])) % grid_size
            id3 = ((x1 * PRIMES[0]) ^ (y2 * PRIMES[1])) % grid_size
            id4 = ((x2 * PRIMES[0]) ^ (y2 * PRIMES[1])) % grid_size
        else:
            id1 = (x1 + y1 * res).long()
            id2 = (y1 * res + x2).long()
            id3 = (y2 * res + x1).long()
            id4 = (y2 * res + x2).long()
    else:
        print("NOT IMPLEMENTED")

    values = (
        torch.einsum("ab,abc->abc", w1, grid[(id1).long()])
        + torch.einsum("ab,abc->abc", w2, grid[(id2).long()])
        + torch.einsum("ab,abc->abc", w3, grid[(id3).long()])
        + torch.einsum("ab,abc->abc", w4, grid[(id4).long()])
    )
    return values[0]


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d
