import os
import random
from collections import deque

import numpy as np
import torch
import torchvision
import lpips
import wandb 
from matplotlib import pyplot as plt
from einops import rearrange
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2s
from skimage.measure import label, regionprops, find_contours
import re

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_reproducibility_cudnn():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Meter:
    def __init__(self):
        self._deque = deque()
        self._count = 0
        self._total = 0.0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    def reset(self):
        self._deque.clear()
        self._count = 0
        self._total = 0.0

    @property
    def avg(self):
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None


def get_torchvision_unnormalize(mean, std):
    """
    Get unnormalize function. reference: https://github.com/pytorch/vision/issues/528

    Args:
        mean, std (list): Normalization parameters (RGB)

    Returns:
        unnormalize (torchvision.transforms.Normalize): Unnormalize function.
    """
    assert len(mean) == 3
    assert len(std) == 3
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)

    unnormalize = torchvision.transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return unnormalize


def vid_to_wandb(vid_pred, vid_target=None):
    if vid_target is not None: 
        videos = torch.cat([vid_pred, vid_target], dim=-2)
    else: 
        videos = vid_pred

    videos = rearrange(videos, "b c t h w -> t c h (b w)")
    videos = (videos *255).clamp(0,255).numpy().astype(np.uint8)
    temp = np.repeat(videos, 3, axis=1)
    wandb.log({"videos": wandb.Video(temp, fps=2)})

def mask_to_border(mask):
    plt.imsave("mask.png", mask.astype(np.uint8))
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

def extract_seven_concurrent_numbers(text):
    pattern = r'\b\d{7}\b'
    seven_numbers = re.findall(pattern, text)
    return seven_numbers

def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    plt.imsave("border.png", mask.astype(np.uint8))
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]
        print(x1, y1, x2, y2)
        bboxes.append([x1, y1, x2, y2])

    return bboxes


def find_bounding_box(mask):
    # Find rows and columns where mask is non-zero
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Find the minimum and maximum indices of rows and columns
    try:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
    except IndexError as e:
        return 0, 0, 0, 0

    # Return the bounding box coordinates
    return xmin, ymin, xmax, ymax


def find_bounding_box3D(mask):
    # Find non-zero elements indices
    non_zero_indices = torch.nonzero(mask)

    # Get the minimum and maximum coordinates along each axis
    z_min, y_min, x_min = torch.min(non_zero_indices, dim=0)[0]
    z_max, y_max, x_max = torch.max(non_zero_indices, dim=0)[0]

    return z_min.item(), z_max.item(), y_min.item(), y_max.item(), x_min.item(), x_max.item()

def center_crop_bbox(image_size, crop_size):
    """
    Calculates bounding box coordinates for a center crop of a given size.

    Parameters:
    image_size (tuple): The size of the original image in the format (height, width).
    crop_size (int): The size of the center crop.

    Returns:
    tuple: A tuple containing the coordinates of the bounding box in the format (y_min, y_max, x_min, x_max).
    """
    height, width = image_size
    top = int((height - crop_size) / 2)
    left = int((width - crop_size) / 2)
    bottom = top + crop_size
    right = left + crop_size
    return top, bottom, left, right

def calculate_metrics(video_pred, video_tar, metrics, segmented=None):
    def name2function(metric):
        available = ["ssim", "mse", "psnr", "mae", "r2s", "lpips"]
        if metric not in available: 
            raise NameError(f"Metric: {metric} not available. Please chose from {available}")
        if metric == "ssim":
            return ssim
        elif metric == "mse": 
            return mean_squared_error
        elif metric == "psnr":
            return psnr
        elif metric == "mae":
            return mae
        # elif metric == "rmse":
        #     return rmse 
        elif metric == "r2s":
            return r2s
        elif metric == "lpips":
            loss_fn_alex = lpips.LPIPS(net="alex")
            return loss_fn_alex
    
    if segmented is None:
        seg = np.ones_like(video_pred)
    else: 
        seg = segmented


    seg_metrics = ["mse", "mae", "rmse", "r2s", "lpips"]
    crop_metrics = ['ssim', 'psnr']

    results = []
    for metric in metrics:
        if metric in seg_metrics:
            video_pred *= seg
            video_tar *= seg

        frame_metrics= []
        for (frame_pred, frame_tar, frame_seg) in list(zip(video_pred, video_tar, seg)):
            if metric in crop_metrics and segmented is not None:
                w_min, h_min, w_max, h_max = find_bounding_box(frame_seg)
                if (w_max - w_min) < 7 or (h_max - h_min) < 7:
                    continue
                frame_pred = frame_pred[h_min:h_max, w_min:w_max]
                frame_tar = frame_tar[h_min:h_max, w_min:w_max]
            try:
                frame_metric = name2function(metric)(frame_pred, frame_tar)
            except ValueError as e:
                print("Gave data range manually")
                frame_metric = name2function(metric)(frame_pred, frame_tar, data_range=1)
            frame_metrics.append(frame_metric)
        results.append(sum(frame_metrics) / (len(frame_metrics) + np.finfo(np.float64).eps))
        

    return results