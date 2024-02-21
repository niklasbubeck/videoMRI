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

class UKBB(Dataset):

    def __init__(self, config, sbj_file=None, transforms=None) -> None:
        """
        Constructor Method
        """

        self.target_resolution = config.dataset.get("res", 256)
        self.root_dir = config.dataset.get("data_path", 256)
        self.transforms = transforms


        self.sa_fnames = []
        self.la_fnames =[]
        self.seg_fnames = []
        # self.meta_fnames = []
        subjects = ["**"]
        if sbj_file:
            subjects = self.read_subject_numbers(sbj_file)

        for subject in tqdm(subjects):
            try:
                    self.sa_fnames += glob.glob(f'{self.root_dir}/{subject}/sa_cropped.nii.gz', recursive=True) 
                    self.la_fnames += glob.glob(f'{self.root_dir}/{subject}/la_2ch.nii.gz', recursive=True) 
                    self.seg_fnames += glob.glob(f'{self.root_dir}/{subject}/seg_sa_cropped.nii.gz', recursive=True)
            except:
                ImportError('No data found in the given path')

        # some subjects dont have both, check for edge cases 
        sa_subjects = [self.extract_seven_concurrent_numbers(file) for file in self.sa_fnames]
        la_subjects = [self.extract_seven_concurrent_numbers(file) for file in self.la_fnames]
        seg_subjects = [self.extract_seven_concurrent_numbers(file) for file in self.seg_fnames]  
        common_subjects = self.common_subjects(la_subjects, sa_subjects)
        common_subjects = self.common_subjects(common_subjects, seg_subjects)
        self.sa_fnames = [f'{self.root_dir}/{subject}/sa_cropped.nii.gz' for subject in common_subjects]
        self.la_fnames = [f'{self.root_dir}/{subject}/la_2ch.nii.gz' for subject in common_subjects]
        self.seg_fnames = [f'{self.root_dir}/{subject}/seg_sa_cropped.nii.gz' for subject in common_subjects]

        print(f'{len(self.sa_fnames)} files found in {self.root_dir}')
        # print(f'{len(self.la_fnames)} files found in {self.root_dir}/{folder}')
        assert len(self.sa_fnames) == len(self.la_fnames) == len(self.seg_fnames), f"number of sa {len(self.sa_fnames)} and la {len(self.la_fnames)} and seg {len(self.seg_fnames)} not equal"
        assert len(self.sa_fnames) != 0, f"Given directory contains 0 images. Please check on the given root: {self.root_dir}"


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

    @property
    def fnames(self):
        return self.targets_fnames

    
    def load_nifti(self, fname:str):
        nii = nib.load(fname).get_fdata()
        return nii

    def load_meta_patient(self, fname:str):

        file = open(fname, 'r')
        content = file.read()
        config_dict = {}
        lines = content.split("\n") #split it into lines
        for path in lines:
            split = path.split(": ")
            if len(split) != 2: 
                break
            key, value = split[0], split[1]
            config_dict[key] = value

        return config_dict

    def min_max(self, x, min, max):
        # Scale to SA 
        std = (x- x.min()) / (x.max() - x.min())
        x = std * (max - min) + min
        return x 

    def __len__(self):
        return len(self.sa_fnames)

    def __getitem__(self, idx):
        
        # load the short axis-image
        sa = self.load_nifti(self.sa_fnames[idx]) # h, w, s, t
        sa = self.min_max(sa, 0, 1) 

        # load the segmentation short-axis image
        sa_seg = self.load_nifti(self.seg_fnames[idx]) #h ,w, s, t 


        # load the long-axis image 
        la = self.load_nifti(self.la_fnames[idx]) # h, w, s, t
        la = self.min_max(la, 0, 1)
    
        la_o_h, la_o_w, la_o_s, la_o_t = la.shape

        # crop la image 
        left = (la_o_w - self.target_resolution) // 2
        top = (la_o_h - self.target_resolution) // 2
        right = left + self.target_resolution
        bottom = top + self.target_resolution
        la = la[top:bottom, left:right, 0, 0]

        # error handling for la images smaller than 128
        la_h, la_w = la.shape
        if la_h != self.target_resolution or la_w != self.target_resolution:
            print(f"Weird stuff: {la_o_h} {la_o_w} --> {la_h} {la_w}")
            return self.__getitem__(idx + 1)
    
        # add channel dimension and float it 
        sa = torch.from_numpy(sa).unsqueeze(0).type(torch.float)         # c, h ,w ,s ,t 
        la = torch.from_numpy(la).unsqueeze(0).type(torch.float)         # c, h, w
        sa_seg = torch.from_numpy(sa_seg).unsqueeze(0).type(torch.float) # c, h, w, s, t 
        
        # rearrange
        sa = rearrange(sa, "c h w s t -> c s t h w")
        la = rearrange(la, "c h w -> c h w")
        sa_seg = rearrange(sa_seg, "c h w s t -> c s t h w")
        
        # apply transformations
        sa = self.transforms(sa)
        la = self.transforms(la)
        sa_seg = self.transforms(sa_seg)

        # return dict with each filename 
        fnames = dict(
            sa = self.sa_fnames[idx],
            la = self.la_fnames[idx],
            sa_seg = self.seg_fnames[idx]
        )

        return sa, la, sa_seg, fnames
