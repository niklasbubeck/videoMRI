import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom
import cv2
import re
from tqdm import tqdm
import torchvision
from einops import rearrange
from typing import Union

from .models.network_helpers import normalize_neg_one_to_one, unnormalize_zero_to_one
from .utils import find_bounding_box3D

MEAN_SAX_LV_VALUE = 222.7909
MAX_SAX_VALUE = 487.0
MEAN_4CH_LV_VALUE = 224.8285
MAX_4CH_LV_VALUE = 473.0


def get_torchvision_transforms(cfg, mode):
    assert mode in {'train', 'test'}
    if mode == 'train':
        transforms_cfg = cfg.dataset.train.transforms
    else:
        transforms_cfg = cfg.dataset.test.transforms

    transforms = []
    for t in transforms_cfg:
        if hasattr(torchvision.transforms, t['name']):
            transform_cls = getattr(torchvision.transforms, t['name'])(**t['params'])
        else:
            raise ValueError(f'Tranform {t["name"]} is not defined')
        transforms.append(transform_cls)
    transforms = torchvision.transforms.Compose(transforms)

    return transforms


def normalize_image_with_mean_lv_value(im: Union[np.ndarray, torch.Tensor], mean_value=MEAN_SAX_LV_VALUE,
                                       target_value=0.5) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize such that LV pool has value of 0.5. Assumes min value is 0.0. """
    im = im / (mean_value / target_value)
    im = im.clip(min=0.0, max=1.0)
    return im


class UKBB(Dataset):

    def __init__(self, config, sbj_file=None, transforms=None) -> None:
        """
        Constructor Method
        """

        self.target_resolution = config.dataset.get("res", 256)
        self.root_dir = config.dataset.get("data_path", 256)
        self.transforms = transforms
        self.slice_res = config.dataset.get("slice_res", 8)
        self.normalize = config.dataset.get("normalize", False)
        self.crop_along_bbox = config.dataset.get("crop_along_bbox", False)

        self.fnames = []
        # self.la_fnames =[]
        # self.seg_fnames = []
        # self.meta_fnames = []
        subjects = os.listdir(self.root_dir)
        if sbj_file is not None:
            subjects = self.read_subject_numbers(sbj_file)

        for subject in tqdm(subjects):
            # if len(self.fnames) >= 100:
            #     break
            try:
                self.fnames += glob.glob(f'{self.root_dir}/{subject}/processed_seg_allax.npz', recursive=True)
                # self.la_fnames += glob.glob(f'{self.root_dir}/{subject}/la_2ch.nii.gz', recursive=True)
                # self.seg_fnames += glob.glob(f'{self.root_dir}/{subject}/seg_sa_cropped.nii.gz', recursive=True)
            except:
                ImportError('No data found in the given path')

        subject_list = [self.extract_seven_concurrent_numbers(fname) for fname in tqdm(self.fnames)]
        with open("used_sbj.txt", 'w') as f:
            for item in tqdm(subject_list):
                f.write(str(item) + '\n')
        # some subjects dont have both, check for edge cases
        # sa_subjects = [self.extract_seven_concurrent_numbers(file) for file in self.sa_fnames]
        # la_subjects = [self.extract_seven_concurrent_numbers(file) for file in self.la_fnames]
        # seg_subjects = [self.extract_seven_concurrent_numbers(file) for file in self.seg_fnames]
        # common_subjects = self.common_subjects(la_subjects, sa_subjects)
        # common_subjects = self.common_subjects(common_subjects, seg_subjects)
        # self.sa_fnames = [f'{self.root_dir}/{subject}/sa_cropped.nii.gz' for subject in common_subjects]
        # self.la_fnames = [f'{self.root_dir}/{subject}/la_2ch.nii.gz' for subject in common_subjects]
        # self.seg_fnames = [f'{self.root_dir}/{subject}/seg_sa_cropped.nii.gz' for subject in common_subjects]

        print(f'{len(self.fnames)} files found in {self.root_dir}')
        # print(f'{len(self.la_fnames)} files found in {self.root_dir}/{folder}')
        # assert len(self.sa_fnames) == len(self.la_fnames) == len(self.seg_fnames), f"number of sa {len(self.sa_fnames)} and la {len(self.la_fnames)} and seg {len(self.seg_fnames)} not equal"
        # assert len(self.sa_fnames) != 0, f"Given directory contains 0 images. Please check on the given root: {self.root_dir}"

    def extract_seven_concurrent_numbers(self, text):
        pattern = r'\b\d{7}\b'
        seven_numbers = re.findall(pattern, text)
        return seven_numbers[0]

    def common_subjects(self, la_subjects, sa_subjects):
        # Convert both lists to sets
        la_set = set(la_subjects)
        sa_set = set(sa_subjects)

        # Find the common subjects using set intersection
        common_subjects_set = la_set.intersection(sa_set)

        # Convert the result back to a list
        common_subjects_list = list(common_subjects_set)

        return common_subjects_list

    def read_subject_numbers(self, file_path):
        try:
            with open(file_path, 'r') as file:
                subject_numbers = [line.strip() for line in file.readlines()]
                subject_numbers = [num for num in subject_numbers if num.isdigit() and len(num) == 7]
                return subject_numbers
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, value):
        self._indices = value

        # @property

    # def fnames(self):
    #     return self.targets_fnames

    def load_nifti(self, fname: str):
        nii = nib.load(fname).get_fdata()
        return nii

    def load_meta_patient(self, fname: str):

        file = open(fname, 'r')
        content = file.read()
        config_dict = {}
        lines = content.split("\n")  # split it into lines
        for path in lines:
            split = path.split(": ")
            if len(split) != 2:
                break
            key, value = split[0], split[1]
            config_dict[key] = value

        return config_dict

    def min_max(self, x, min, max):
        # Scale to SA
        std = (x - x.min()) / (x.max() - x.min())
        x = std * (max - min) + min
        return x

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):

        process_npy = np.load(self.fnames[idx])
        sa = process_npy['sax']  # [H, W, S, T]
        la = process_npy['lax']  # [H, W, S, T]
        sa_seg = process_npy['seg_sax']  # [H, W, S, T]

        sa = normalize_image_with_mean_lv_value(sa)
        la = normalize_image_with_mean_lv_value(la)

        h, w, s, t = sa.shape

        # # load the short axis-image
        # sa = self.load_nifti(self.sa_fnames[idx]) # h, w, s, t
        # sa = self.min_max(sa, 0, 1)

        # # load the segmentation short-axis image
        # sa_seg = self.load_nifti(self.seg_fnames[idx]) #h ,w, s, t

        # # load the long-axis image
        # la = self.load_nifti(self.la_fnames[idx]) # h, w, s, t
        # la = self.min_max(la, 0, 1)

        # la_o_h, la_o_w, la_o_s, la_o_t = la.shape

        # # crop la image
        # left = (la_o_w - self.target_resolution) // 2
        # top = (la_o_h - self.target_resolution) // 2
        # right = left + self.target_resolution
        # bottom = top + self.target_resolution
        # la = la[top:bottom, left:right, 0, 0]

        # # error handling for la images smaller than 128
        # la_h, la_w = la.shape
        # if la_h != self.target_resolution or la_w != self.target_resolution:
        #     print(f"Weird stuff: {la_o_h} {la_o_w} --> {la_h} {la_w}")
        #     return self.__getitem__(idx + 1)

        # add channel dimension and float it
        sa = torch.from_numpy(sa).unsqueeze(0).type(torch.float)  # c, h ,w ,s ,t
        la = torch.from_numpy(la).unsqueeze(0).type(torch.float)  # c, h, w
        sa_seg = torch.from_numpy(sa_seg).unsqueeze(0).type(torch.float)  # c, h, w, s, t

        # rearrange
        sa = rearrange(sa, "c h w s t -> c s t h w")
        la = rearrange(la, "c h w s t -> c s t h w")
        sa_seg = rearrange(sa_seg, "c h w s t -> c s t h w")

        # apply transformations
        sa = self.transforms(sa)
        la = self.transforms(la)
        sa_seg = self.transforms(sa_seg)

        if self.slice_res is not None:
            start_slice = (sa.shape[1] - self.slice_res) // 2
            end_slice = start_slice + 8

            sa = sa[:, start_slice:end_slice, ...]
            sa_seg = sa_seg[:, start_slice:end_slice, ...]

        if self.normalize:
            sa = normalize_neg_one_to_one(sa)
            la = normalize_neg_one_to_one(la)

        if self.crop_along_bbox:
            _, _, ymin, ymax, xmin, xmax = find_bounding_box3D(sa_seg[0, 5, ...])

            # bounderis to ensure that every crop is target_res x target_res
            min_x, max_x = self.target_resolution // 2, w - self.target_resolution // 2
            min_y, max_y = self.target_resolution // 2, h - self.target_resolution // 2

            cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
            cx = max(min(cx, max_x), min_x)
            cy = max(min(cy, max_y), min_y)

            x_top_left = max(0, cx - self.target_resolution // 2)
            y_top_left = max(0, cy - self.target_resolution // 2)

            # Calculate the coordinates of the bottom-right corner
            x_bottom_right = min(w, cx + self.target_resolution // 2)
            y_bottom_right = min(h, cy + self.target_resolution // 2)

            sa = sa[..., y_top_left:y_bottom_right, x_top_left:x_bottom_right]
            sa_seg = sa_seg[..., y_top_left:y_bottom_right, x_top_left:x_bottom_right]

        fnames = self.fnames[idx]
        return sa, la, sa_seg, fnames