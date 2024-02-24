import torch.nn as nn
import torch
from functools import partial

from scipy.ndimage import zoom
from einops import rearrange, repeat, reduce
from contextlib import contextmanager, nullcontext


from .network_helpers import *
from .encoder import SemanticEncoder, SemanticEncoder3D
from .unet import Unet, Unet3D

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    def preprocess(self,
                    batch, 
                    time_res,
                    slice_res,
                    res, 
                    cond_slices,
                    slice_idx,
                    time_idx=None, 
                    mode="fcfs",
                    **kwargs):
        
        sa, la, seg_sa, fnames = batch
        
        assert mode in ["interpolate", "fcfs", "uniform"], f"dont know the given mode {mode}"
        assert sa.shape == seg_sa.shape, "segmentation and short axis image shapes do not match"
        assert sa.shape[-1] == sa.shape[-2] and sa.shape[-1] in [16, 32, 64, 128, 256], "Can only take care of quadratic images"

        # Resize resolution if necessary
        if sa.shape[-1] != res:
            sa = torch.from_numpy(zoom(sa.numpy(), (1, 1, 1 , 1, res/sa.shape[4], res/sa.shape[5]), order=1)).clamp(0, 1)                      # b, c, s, t, h, w 
            seg_sa = torch.from_numpy(zoom(seg_sa.numpy(), (1, 1, 1, 1, res/seg_sa.shape[4], res/seg_sa.shape[5]), order=1)).clamp(0, 1)       # b, c, s, t, h, w
            la = torch.from_numpy(zoom(la.numpy(), (1, 1, res/seg_sa.shape[4], res/seg_sa.shape[5]), order=1)).clamp(0, 1)                     # b, c, h, w

        # Resize the slices and time dimensions
        if mode == "interpolate":
            sa = torch.from_numpy(zoom(sa.numpy(), (1, 1, 1, 1/time_res, res/sa.shape[4], res/sa.shape[5]))).clamp(0, 1)                      # b, c, s, t, h, w 
            seg_sa = torch.from_numpy(zoom(seg_sa.numpy(), (1, 1, 1, 1/time_res, res/seg_sa.shape[4], res/seg_sa.shape[5]))).clamp(0, 1)  # b, c, s, t, h, w 

        if mode == "fcfs":
            sa = sa[:, :, :, 0::time_res, ...]
            seg_sa = seg_sa[:, :, :, 0::time_res, ...]

        if mode == "uniform": 
            raise NotImplementedError

        # define slice number
        sa = sa[..., slice_idx, :, :, :]      # b, c, t, h ,w
        seg_sa = seg_sa[..., slice_idx, :, :, :]    # b, c, t, h ,w

        # define time frame
        if time_idx is not None: 
            sa = sa[:, :, time_idx, ...]            # b, c, h ,w
            seg_sa = seg_sa[:, :, time_idx, ...]    # b, c, h ,w 

        cond_frames = [ sa[:, :, slice, time_idx, ...] for slice in cond_slices]
        cond_frames = torch.cat(cond_frames, dim=1)

        return sa, la, seg_sa, cond_frames, fnames


    def postprocess(self, input, fourier_mode=None):
        return input


class CascadedDiffusionModel(BaseModel):
    def __init__(self,
                 unets,
                 image_sizes,
                 channels=3,
                 temporal_downsample_factor=1,
                 text_embed_dim=512,
                 condition_on_text = True,
                 auto_normalize_img = True,
                 lowres_sample_noise_level = 0.2,
                 device = 'cuda',
                 **kwargs
                 ) -> None:
        super().__init__()

        self.unets = nn.ModuleList([])
        self.num_unets = len(unets)
        self.image_sizes = image_sizes
        self.temporal_downsample_factor = temporal_downsample_factor
        self.text_embed_dim = text_embed_dim
        self.channels = channels
        self.unet_being_trained_index = -1
        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text
        self.lowres_sample_noise_level = lowres_sample_noise_level

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        assert self.num_unets == len(self.image_sizes) == len(self.temporal_downsample_factor), "given image sizes or temp_downsample_factors do not match the number of unets"   
        assert temporal_downsample_factor[-1] == 1, 'downsample factor of last stage must be 1'
        assert tuple(sorted(temporal_downsample_factor, reverse = True)) == temporal_downsample_factor, 'temporal downsample factor must be in order of descending'

        for i, unet in enumerate(unets):
            assert isinstance(unet, (DiffusionAutoEncoders, DiffusionAutoEncoders3D, Unet3D, Unet)), "selected unet is not available"
            self.unets.append(unet)

        is_video = any([isinstance(unet, (Unet3D, DiffusionAutoEncoders3D)) for unet in self.unets])
        self.is_video = is_video

        self.resize_to = resize_video_to if is_video else resize_image_to
        self.resize_to = partial(self.resize_to, mode = 'nearest')

        # conditioning for cascading 
        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))


        assert lowres_conditions == (False, *((True,) * (self.num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'
        # one temp parameter for keeping track of device

        self.register_buffer('_temp', torch.tensor([0.]), persistent = False)

        # default to device of unets passed in

        self.to(next(self.unets.parameters()).device)


    def _preprocess_cascaded(self, xt, seg , unet_number, **kwargs):
        assert xt.shape == seg.shape, f"shapes of xt and seg do not match with {xt.shape} and {seg.shape}"
        target_image_size    = self.image_sizes[unet_number]
        prev_image_size      = self.image_sizes[unet_number - 1] if unet_number > 0 else None

        batch_size, c, *_, h, w, device, is_video = *xt.shape, xt.device, (xt.ndim == 5)
        frames              = xt.shape[2] if is_video else None
        all_frame_dims      = tuple(safe_get_tuple_index(el, 0) for el in calc_all_frame_dims(self.temporal_downsample_factor, frames))

        target_frame_size   = all_frame_dims[unet_number] if is_video else None
        prev_frame_size     = all_frame_dims[unet_number - 1] if is_video and unet_number >= 0 else None
        frames_to_resize_kwargs = lambda frames: dict(target_frames = frames) if exists(frames) else dict()

        assert xt.shape[1] == self.channels
        assert h >= target_image_size and w >= target_image_size

        # low resolution conditioning
        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = self.resize_to(xt, prev_image_size, **frames_to_resize_kwargs(prev_frame_size), clamp_range = (0,1))
            lowres_cond_img = self.resize_to(lowres_cond_img, target_image_size, **frames_to_resize_kwargs(target_frame_size), clamp_range = (0,1))

        xt = self.resize_to(xt, target_image_size, **frames_to_resize_kwargs(target_frame_size))
        seg = self.resize_to(seg, target_image_size, **frames_to_resize_kwargs(target_frame_size))

        # TODO need encode noise augmentation to lowres_cond_img 
        # Currently done in trainer but shift all noise sampling to model itself 
        
        # Auto normalize from (0,1) to (-1,1)

        return xt, lowres_cond_img, seg


    def preprocess(self, batch, time_res, slice_res, res, cond_slices, slice_idx, time_idx=None, mode="fcfs", **kwargs):
        x0, la, seg, cond, fnames = super().preprocess(batch, time_res, slice_res, res, cond_slices, slice_idx, time_idx, mode, **kwargs)
        return self._preprocess_cascaded(x0, seg, **kwargs), fnames, x0

    def reset_unets_all_one_device(self, device = None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 <= unet_number < len(self.unets), f'invalid unet number {unet_number} must lie between 0 and {len(self.unets)}'
        index = unet_number

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        cpu = torch.device('cpu')

        devices = [module_device(unet) for unet in self.unets]

        self.unets.to(cpu)
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)


    def forward(
            self,
            x0,
            xt,
            t,
            lowres_cond_img,
            lowres_noise_times,
            unet_number=0,
            cond_images=None,
            ignore_time=0.1,
            resize_cond_video_frames=True,
            **kwargs
            ):
        assert xt.shape[-1] == xt.shape[-2], f'the images you pass in must be a square, but received dimensions of {xt.shape[2]}, {xt.shape[-1]}'
        assert xt.dtype == torch.float, f'images tensor needs to be floats but {xt.dtype} dtype found instead'
        self.resize_cond_video_frames = resize_cond_video_frames

        unet = self.get_unet(unet_number)

        return unet.forward(x0, xt, t, lowres_cond_img=lowres_cond_img, lowres_noise_times=lowres_noise_times)      


class DiffusionAutoEncoders(BaseModel):
    def __init__(self, enc_config, unet_config, lowres_cond=False):
        """
        Args:
            cfg: A dict of config.
        """
        super().__init__()
        self.lowres_cond = lowres_cond
        self.encoder = SemanticEncoder(enc_config)
        self.unet = Unet(unet_config)

    def forward(self, x0, xt, t, lowres_cond_img=None, lowres_noise_times=None):
        """
        Args:
            x0 (torch.tensor): A tensor of original image.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            xt (torch.tensor): A tensor of x at time step t.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            t (torch.tensor): A tensor of time steps.
                shape = (batch, )
                dtype = torch.float32

        Returns:
            out (torch.tensor): A tensor of output.
                shape = (batch, channels, height, width)
                dtype = torch.float32
        """

        style_emb = self.encoder(x0)
        out = self.unet(xt, t, style_emb)
        return out

class DiffusionAutoEncoders3D(BaseModel):
    def __init__(self, enc_config, unet_config, lowres_cond=False):
        """
        Args:
            cfg: A dict of config.
        """
        super().__init__()
        self.lowres_cond = lowres_cond
        self.encoder = SemanticEncoder3D(**enc_config)
        self.unet = Unet3D(**unet_config)

    def forward(self, x0, xt, t, lowres_cond_img=None, lowres_noise_times=None):
        """
        Args:
            x0 (torch.tensor): A tensor of original image.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            xt (torch.tensor): A tensor of x at time step t.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            t (torch.tensor): A tensor of time steps.
                shape = (batch, )
                dtype = torch.float32

        Returns:
            out (torch.tensor): A tensor of output.
                shape = (batch, channels, height, width)
                dtype = torch.float32
        """

        style_emb = self.encoder(x0)
        out = self.unet(xt, t, text_embeds=style_emb, lowres_cond_img=lowres_cond_img, lowres_noise_times=lowres_noise_times)
        return out