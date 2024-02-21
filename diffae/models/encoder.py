import math
import operator
import functools
from tqdm.auto import tqdm
from functools import partial, wraps
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from .network_helpers import *
from .network_blocks import *
import torch.nn as nn

from .network_blocks import AttentionBlock, Downsample, ResnetBlock2D


class SemanticEncoder(nn.Module):
    def __init__(self, config):
        """
        Semantic encoder summarizes input images into descriptive vectors.

        Args (dict): A dict of config.
        """
        super().__init__()
        self.network_config = config
        self.image_size = config.image_size
        self.in_chans = config.in_channels
        self.out_chans = config.out_channels
        self.model_chans = config.model_channels
        self.emb_chans = config.emb_channels
        self.chan_mults = config.channel_multipliers
        self.n_res_blocks = config.num_resnet_blocks
        self.res_dropout = config.resnet_dropout
        self.attn_resolution = config.attn_resolution
        self.use_conv_resample = config.use_conv_resample
        self.groups = config.num_groups

        self.time_emb_chans = self.emb_chans
        self.style_emb_chans = self.emb_chans
        self.n_resolutions = len(self.chan_mults)

        self._create_network()

    def _create_network(self):
        """
        Create semantic encoder networks.
        The architecture is same as that of the first half of Unet decoder.
        """
        # e.g.) model_chans = 64, chan_mults = (1, 2, 4, 8) ---> chans = [64, 128, 256, 512]
        chans = [self.model_chans, *map(lambda x: self.model_chans * x, self.chan_mults)]

        # e.g.) image_size = 64, len(chan_mults) = 4 ---> resolutions = [64, 32, 16, 8]
        resolutions = [self.image_size // (2**i) for i in range(self.n_resolutions)]

        self.init_conv = nn.Conv2d(self.in_chans, self.model_chans, 3, 1, 1)

        # downsampling
        self.downs = nn.ModuleList()
        for i_level in range(self.n_resolutions):
            in_c = chans[i_level]
            out_c = chans[i_level + 1]
            resolution = resolutions[i_level]

            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            for i_block in range(self.n_res_blocks):
                res_block.append(ResnetBlock2D(in_c, out_c, self.res_dropout, self.groups))
                if resolution in self.attn_resolution:
                    attn_block.append(AttentionBlock(out_c, groups=self.groups))
                else:
                    attn_block.append(nn.Identity())
                in_c = out_c
            is_last = bool(i_level == self.n_resolutions - 1)
            if is_last:
                downsample = nn.Identity()
            else:
                downsample = Downsample(out_c, use_conv=self.use_conv_resample)

            down = nn.Module()
            down.res_block = res_block
            down.attn_block = attn_block
            down.downsample = downsample
            self.downs.append(down)

        # middle
        mid_chans = chans[-1]
        self.middle = nn.Module()
        self.middle.res_block1 = ResnetBlock2D(mid_chans, mid_chans, self.res_dropout, self.groups)
        self.middle.attn_block1 = AttentionBlock(mid_chans, self.groups)
        self.middle.res_block2 = ResnetBlock2D(mid_chans, mid_chans, self.res_dropout, self.groups)

        self.flatten_block = nn.Sequential(
            nn.GroupNorm(self.groups, mid_chans),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(mid_chans, self.style_emb_chans, 1, 1, 0),
            nn.Flatten(),
        )

    def forward(self, x):
        """
        Args:
            x (torch.tensor): A tensor of original image.
                shape = (batch, channels, height, width)
                dtype = torch.float32

        Returns:
            out (torch.tensor): Style embedding.
                shape = (batch, style_emb_chans)
                dtype = torch.float32
        """
        out = self.init_conv(x)
        for i_level in range(self.n_resolutions):
            for i_block in range(self.n_res_blocks):
                out = self.downs[i_level].res_block[i_block](out)
                out = self.downs[i_level].attn_block[i_block](out)
            out = self.downs[i_level].downsample(out)

        out = self.middle.res_block1(out)
        out = self.middle.attn_block1(out)
        out = self.middle.res_block2(out)

        out = self.flatten_block(out)
        return out



class SemanticEncoder3D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        text_embed_dim = 512,  # can also use it for style_emb from diffae 
        num_resnet_blocks = 1,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        learned_sinu_pos_emb_dim = 16,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        temporal_strides = 1,
        cond_images_channels = 0,
        channels = 3,
        channels_out = None,
        attn_dim_head = 64,
        attn_heads = 8,
        ff_mult = 2.,
        ff_time_token_shift = True,         # this would do a token shift along time axis, at the hidden layer within feedforwards - from successful use in RWKV (Peng et al), and other token shift video transformer works
        lowres_cond = False,                # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns = False,
        layer_attns_depth = 1,
        layer_attns_add_text_cond = True,   # whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
        attend_at_middle = True,            # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        time_rel_pos_bias_depth = 2,
        time_causal_attn = True,
        layer_cross_attns = True,
        use_linear_attn = False,
        use_linear_cross_attn = False,
        cond_on_text = True,
        max_text_len = 256,
        init_dim = None,
        resnet_groups = 8,
        init_conv_kernel_size = 7,          # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        attn_pool_text = True,
        attn_pool_num_latents = 32,
        dropout = 0.,
        memory_efficient = False,
        init_conv_to_final_conv_residual = False,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        self_cond = False,
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = True,       # may address checkboard artifacts
        resize_mode = 'nearest'
    ):
        super().__init__()

        # guide researchers

        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        if dim < 128:
            print_once('The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/')

        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        self.self_cond = self_cond

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)
        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))
        init_dim = default(init_dim, dim)

        # optional image conditioning

        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels

        init_channels += cond_images_channels

        # initial convolution

        self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1) if init_cross_embed else Conv2d(init_channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        # embedding time for log(snr) noise from continuous version

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        # low res aug noise conditioning

        self.lowres_cond = lowres_cond

        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU()
            )

            self.to_lowres_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r = num_time_tokens)
            )

        # normalizations

        self.norm_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text:
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on text encodings

        self.cond_on_text = cond_on_text

        # attention pooling

        self.attn_pool = PerceiverResampler(dim = cond_dim, depth = 2, dim_head = attn_dim_head, heads = attn_heads, num_latents = attn_pool_num_latents) if attn_pool_text else None

        # for classifier free guidance

        self.max_text_len = max_text_len

        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

        # for non-attention based text conditioning at all points in the network where time is also conditioned

        self.to_text_non_attn_cond = None

        if cond_on_text:
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim)
            )

        # attention related params

        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)

        num_layers = len(in_out)

        # temporal attention - attention across video frames

        temporal_peg_padding = (0, 0, 0, 0, 2, 0) if time_causal_attn else (0, 0, 0, 0, 1, 1)
        temporal_peg = lambda dim: Residual(nn.Sequential(Pad(temporal_peg_padding), nn.Conv3d(dim, dim, (3, 1, 1), groups = dim)))

        temporal_attn = lambda dim: RearrangeTimeCentric(Residual(Attention(dim, **{**attn_kwargs, 'causal': time_causal_attn, 'init_zero': True, 'rel_pos_bias': True})))

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        resnet_klass = partial(ResnetBlock3D, **attn_kwargs)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        # temporal downsample config

        temporal_strides = cast_tuple(temporal_strides, num_layers)
        self.total_temporal_divisor = functools.reduce(operator.mul, temporal_strides, 1)

        # downsample klass

        downsample_klass = DownsamplePseudo3D

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # initial resnet block (for memory efficient unet)

        self.init_resnet_block = resnet_klass(init_dim, init_dim, groups = resnet_groups[0], use_gca = use_global_context_attn) if memory_efficient else None

        self.init_temporal_peg = temporal_peg(init_dim)
        self.init_temporal_attn = temporal_attn(init_dim)

        # scale for resnet skip connections

        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_attns_depth, layer_cross_attns, temporal_strides]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, temporal_stride) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_use_linear_cross_attn = not layer_cross_attn and use_linear_cross_attn
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            transformer_block_klass = TransformerBlock if layer_attn else (LinearAttentionTransformerBlock if use_linear_attn else Identity)

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out) if not is_last else Parallel(Conv2d(dim_in, dim_out, 3, padding = 1), Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, current_dim, linear_attn = layer_use_linear_cross_attn, groups = groups),
                nn.ModuleList([ResnetBlock3D(current_dim, current_dim, groups = groups, use_gca = use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim, depth = layer_attn_depth, ff_mult = ff_mult, ff_time_token_shift = ff_time_token_shift, **attn_kwargs),
                temporal_peg(current_dim),
                temporal_attn(current_dim),
                TemporalDownsample(current_dim, stride = temporal_stride) if temporal_stride > 1 else None,
                post_downsample
            ]))

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock3D(mid_dim, mid_dim, groups = resnet_groups[-1])
        self.mid_attn = Residual(Attention(mid_dim, **attn_kwargs)) if attend_at_middle else None
        self.mid_temporal_peg = temporal_peg(mid_dim)
        self.mid_temporal_attn = temporal_attn(mid_dim)
        self.mid_block2 = ResnetBlock3D(mid_dim, mid_dim, groups = resnet_groups[-1])

        self.flatten_block = nn.Sequential(
            nn.GroupNorm(resnet_groups[-1], mid_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            Conv2d(mid_dim, text_embed_dim, 1, 1, 0),
            nn.Flatten(),
        )

        
    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    # methods for returning the full unet config as well as its parameter state

    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # class method for rehydrating the unet from its config and state dict

    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # methods for persisting unet to disk

    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok = True, parents = True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config = config, state_dict = state_dict)
        torch.save(pkg, str(path))

    # class method for rehydrating the unet from file saved with `persist_to_file`

    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        assert 'config' in pkg and 'state_dict' in pkg
        config, state_dict = pkg['config'], pkg['state_dict']

        return SemanticEncoder3D.from_config_and_state_dict(config, state_dict)

    # forward with classifier free guidance

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        *,
        lowres_cond_img = None,
        lowres_noise_times = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        cond_video_frames = None,
        post_cond_video_frames = None,
        self_cond = None,
        cond_drop_prob = 0.,
        ignore_time = False
    ):
        assert x.ndim == 5, 'input to 3d unet must have 5 dimensions (batch, channels, time, height, width)'

        batch_size, frames, device, dtype = x.shape[0], x.shape[2], x.device, x.dtype

        assert ignore_time or divisible_by(frames, self.total_temporal_divisor), f'number of input frames {frames} must be divisible by {self.total_temporal_divisor}'

        # add self conditioning if needed

        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim = 1)

        # add low resolution conditioning, if present

        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)), 'low resolution conditioning noise time must be present'

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

            if exists(cond_video_frames):
                lowres_cond_img = torch.cat((cond_video_frames, lowres_cond_img), dim = 2)
                cond_video_frames = torch.cat((cond_video_frames, cond_video_frames), dim = 1)

            if exists(post_cond_video_frames):
                lowres_cond_img = torch.cat((lowres_cond_img, post_cond_video_frames), dim = 2)
                post_cond_video_frames = torch.cat((post_cond_video_frames, post_cond_video_frames), dim = 1)

        # conditioning on video frames as a prompt

        num_preceding_frames = 0
        if exists(cond_video_frames):
            cond_video_frames_len = cond_video_frames.shape[2]

            assert divisible_by(cond_video_frames_len, self.total_temporal_divisor)

            cond_video_frames = resize_video_to(cond_video_frames, x.shape[-1])
            x = torch.cat((cond_video_frames, x), dim = 2)

            num_preceding_frames = cond_video_frames_len

        # conditioning on video frames as a prompt

        num_succeeding_frames = 0
        if exists(post_cond_video_frames):
            cond_video_frames_len = post_cond_video_frames.shape[2]

            assert divisible_by(cond_video_frames_len, self.total_temporal_divisor)

            post_cond_video_frames = resize_video_to(post_cond_video_frames, x.shape[-1])
            x = torch.cat((post_cond_video_frames, x), dim = 2)

            num_succeeding_frames = cond_video_frames_len

        # condition on input image

        assert not (self.has_cond_image ^ exists(cond_images)), 'you either requested to condition on an image on the unet, but the conditioning image is not supplied, or vice versa'

        if exists(cond_images):
            assert cond_images.ndim == 4, 'conditioning images must have 4 dimensions only, if you want to condition on frames of video, use `cond_video_frames` instead'
            assert cond_images.shape[1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'

            cond_images = repeat(cond_images, 'b c h w -> b c f h w', f = x.shape[2])
            cond_images = resize_video_to(cond_images, x.shape[-1], mode = self.resize_mode)

            x = torch.cat((cond_images, x), dim = 1)

        # ignoring time in pseudo 3d resnet blocks

        conv_kwargs = dict(
            ignore_time = ignore_time
        )

        # initial convolution

        x = self.init_conv(x)

        if not ignore_time:
            x = self.init_temporal_peg(x)
            x = self.init_temporal_attn(x)

        # initial resnet block (for memory efficient unet)

        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, **conv_kwargs)

        # go through the layers of the unet, down and up

        hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn_block, temporal_peg, temporal_attn, temporal_downsample, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, **conv_kwargs)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, **conv_kwargs)
                hiddens.append(x)

            x = attn_block(x)

            if not ignore_time:
                x = temporal_peg(x)
                x = temporal_attn(x)

            hiddens.append(x)

            if exists(temporal_downsample) and not ignore_time:
                x = temporal_downsample(x)

            if exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, **conv_kwargs)
        if exists(self.mid_attn):
            x = rearrange(x, 'b c f h w -> b f h w c')
            x, ps = pack([x], 'b * c')

            x = self.mid_attn(x)

            x, = unpack(x, ps, 'b * c')
            x = rearrange(x, 'b f h w c -> b c f h w')

        if not ignore_time:
            x = self.mid_temporal_peg(x)
            x = self.mid_temporal_attn(x)

        x = self.mid_block2(x, **conv_kwargs)
        out = self.flatten_block(x).unsqueeze(0)

        return out