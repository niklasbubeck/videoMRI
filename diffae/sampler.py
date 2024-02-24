import warnings

import lpips
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import os 
import sys 
from PIL import Image
from matplotlib import pyplot as plt

from .utils import get_betas, calculate_metrics, extract_seven_concurrent_numbers


# class SamplerCascaded:
#     def __init__(self, model, config, device):
#         """
#         Args:
#             model: Diffusion Autoencoder model.
#             cfg (dict): A dict of config.
#         """
#         self.model = model
#         self.config = config

#         self.image_sizes = self.config.cascaded.image_sizes
#         self.temporal_downsample_factor = self.config.cascaded.temporal_downsample_factor
#         self.target_frames = [self.config.dataset.time_res // down_factor for down_factor in self.temporal_downsample_factor]
#         self.device = device
#         # dont put complete model but only necessry unet stage, see CascadingDiffusionModel.get_unet
#         # self.model.to(self.device)

#         self.output_dir = self.config.output_dir
#         self.num_timesteps = config.trainer.timestep_sampler.num_sample_steps

#         # get betas and alphas
#         self.betas = get_betas(config)
#         self.alphas = 1 - self.betas
#         self.alphas_cumprod = self.alphas.cumprod(dim=0).to(self.device)
#         self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
#         self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.zeros(1, device=self.device)], dim=0)
        
#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
#         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
#         self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
#         self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
#         self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             self.lpips_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(self.device)

#     def make_video(self, x, path):
#         x *= 255
#         videos = [Image.fromarray(image) for image in x[0,0,...].cpu().numpy().astype(np.uint8)]
#         videos[0].save(path, save_all=True, append_images=videos[1:], duration=1000/8, loop=0)


#     def sample_interpolated_testdata(self, test_dataset, eta=0.0, iterations=1000000, slice_nr="rand", return_res_dict=False):
#         """Autoencode test data and calculate evaluation metrics.
#         """
#         test_dataset = test_dataset
#         test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

#         keys = ['subject', 'slice_nr', "mse", "psnr", "ssim"]

#         groundtruths = []
#         samples = []

#         df = pd.DataFrame(columns=keys)

#         self.model.eval()
#         for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
#             if iter == iterations:
#                 break
#             for  i in range(len(self.model.unets)):
#                 unet = self.model.get_unet(i)
#                 if i == 0:
#                     if slice_nr == "rand":
#                         slice_nr = np.random.randint(1 , batch[0].shape[2]-1)
#                     prev, next = slice_nr - 1, slice_nr + 1
#                     (x0_prev, _, _), _ , reference_prev = self.model.preprocess(batch, **self.config.dataset, slice_idx=prev, unet_number=i)
#                     (x0_inter, lowres, _),fnames, reference_inter = self.model.preprocess(batch, **self.config.dataset, slice_idx=slice_nr, unet_number=i)
#                     (x0_next, _, _), _, reference_next = self.model.preprocess(batch, **self.config.dataset, slice_idx=next, unet_number=i)

#                     if len(x0_inter.shape) == 4:
#                         dim = 3
#                     elif len(x0_inter.shape) == 5:
#                         dim = 4
#                     batch_size = x0_inter.shape[0]

#                     x0_prev = x0_prev.to(self.device)
#                     x0_inter = x0_inter.to(self.device)
#                     x0_next = x0_next.to(self.device)
#                     reference_prev = reference_prev.to(self.device)
#                     reference_inter = reference_inter.to(self.device)
#                     reference_next = reference_next.to(self.device)

#                     reference_images_prevs = [self.model.resize_to(reference_prev, image_size, target_frame) for (image_size, target_frame) in zip(self.image_sizes, self.target_frames)]
#                     reference_images_inters = [self.model.resize_to(reference_inter, image_size, target_frame) for (image_size, target_frame) in zip(self.image_sizes, self.target_frames)]
#                     reference_images_nexts = [self.model.resize_to(reference_next, image_size, target_frame) for (image_size, target_frame) in zip(self.image_sizes, self.target_frames)]

#                     noise_lows_prev = [torch.randn_like(reference_image, device=self.device) for reference_image in reference_images_prevs]
#                     noise_lows_inter = [torch.randn_like(reference_image, device=self.device) for reference_image in reference_images_inters]
#                     noise_lows_next = [torch.randn_like(reference_image, device=self.device) for reference_image in reference_images_nexts]

#                     t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
#                     t_low = t_low.type(torch.long)
#                     alphas_shape = self.alphas_cumprod[t_low].shape

                    
#                     alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)

#                     lowres_images_prev = [torch.sqrt(alpha_t) * reference_images_prevs[i]+ torch.sqrt(1.0 - alpha_t) * noise for i, noise in enumerate(noise_lows_prev)]
#                     lowres_images_prev[0] = None
#                     lowres_images_inter = [torch.sqrt(alpha_t) * reference_images_inters[i]+ torch.sqrt(1.0 - alpha_t) * noise for i, noise in enumerate(noise_lows_inter)]
#                     lowres_images_inter[0] = None
#                     lowres_images_next = [torch.sqrt(alpha_t) * reference_images_nexts[i]+ torch.sqrt(1.0 - alpha_t) * noise for i, noise in enumerate(noise_lows_next)]
#                     lowres_images_next[0] = None



#                     xt_prevs = [self.encode_stochastic(self.model.get_unet(i), reference_images_prev, lowres_cond_img=lowres_images_prev[i], lowres_noise_times=t_low ) for i, reference_images_prev in enumerate(reference_images_prevs)]
#                     xt_nexts = [self.encode_stochastic(self.model.get_unet(i), reference_images_next, lowres_cond_img=lowres_images_prev[i], lowres_noise_times=t_low ) for i, reference_images_next in enumerate(reference_images_nexts)]

#                     style_emb_prevs = [self.model.get_unet(i).encoder(reference) for i, reference in enumerate(reference_images_prevs)]
#                     style_emb_nexts = [self.model.get_unet(i).encoder(reference) for i, reference in enumerate(reference_images_nexts)]

#                     xt_inters = []
#                     style_emb_inters = []
#                     for (xt_1, xt_2, style_emb_1, style_emb_2) in zip (xt_prevs, xt_nexts, style_emb_prevs, style_emb_nexts):
#                         xt_inter, style_emb_inter = self.only_interpolate(xt_1, xt_2, style_emb_1, style_emb_2, alpha=0.5)
#                         xt_inters.append(xt_inter)
#                         style_emb_inters.append(style_emb_inter)

            

#                 t_low = None
#                 if lowres is not None: 
#                     #scale up if necessary
#                     lowres = self.model.resize_to(lowres, self.image_sizes[i], target_frames=self.target_frames[i])
                    
#                     t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
#                     t_low = t_low.type(torch.long)
#                     noise_low = torch.randn_like(lowres, device=self.device)
#                     alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)
#                     lowres = torch.sqrt(alpha_t) * lowres + torch.sqrt(1.0 - alpha_t) * noise_low
#                 # # Scale up if neccessary
#                 # xt_inter = self.model.resize_to(xt_inter, self.image_sizes[i], target_frames=self.target_frames[i])  
#                 xt_inter = xt_inters[i]
#                 unet = self.model.get_unet(i)
#                 for _t in tqdm(reversed(range(self.num_timesteps)), desc='decoding...', total=self.num_timesteps):
#                     t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
#                     e = unet.unet(xt_inter, t, text_embeds=style_emb_inters[i], lowres_cond_img=lowres, lowres_noise_times=t_low)
#                     alphas_shape = self.alphas_cumprod[t].shape

#                     # Equation 12 of Denoising Diffusion Implicit Models
#                     x0_t = (
#                         torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt_inter
#                         - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
#                     ).clamp(-1, 1)
#                     e = (
#                         (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt_inter - x0_t)
#                         / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
#                     )
#                     sigma = (
#                         eta
#                         * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
#                         * torch.sqrt(1 - self.alphas_cumprod[t] / self.alphas_cumprod_prev[t])
#                     )
#                     xt_inter = (
#                         torch.sqrt(self.alphas_cumprod_prev[t]).view(*alphas_shape + (1,) *dim) * x0_t
#                         + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
#                     )
#                     xt_inter = xt_inter + torch.randn_like(xt_inter) * sigma if _t != 0 else xt_inter

#                 lowres = xt_inter

#             groundtruths.append(reference_inter)
#             samples.append(xt_inter.clamp(0, 1))


#             ref = reference_inter[0,0,...].cpu().numpy()
#             sample = xt_inter[0,0,...].clamp(0,1).cpu().numpy()

#             print(ref.shape, sample.shape)
#             subject = extract_seven_concurrent_numbers(fnames['sa'][0])[0]
#             mse, psnr, ssim = calculate_metrics(ref, sample, keys[2:])
#             results = [subject, slice_nr, mse, psnr, ssim]
#             print(results)
#             df.loc[len(df.index)] = results

#             # try to safe the gifs
            
#             ref, sample = ref*255, sample*255
#             os.makedirs(os.path.join(self.output_dir, "videos", "interpolation", subject), exist_ok=True)
#             videos = [Image.fromarray(image) for image in sample.astype(np.uint8)]
#             videos[0].save(os.path.join(self.output_dir, "videos", "interpolation", subject ,f"sample.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
#             videos = [Image.fromarray(image) for image in ref.astype(np.uint8)]
#             videos[0].save(os.path.join(self.output_dir, "videos", "interpolation", subject ,f"reference.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
#             df.to_csv(os.path.join(self.output_dir, "videos", "interpolation" ,f"data.csv"))

#         res_dict = None 
#         if return_res_dict:
#             res_dict = dict(
#                 gts = groundtruths,
#                 preds = samples 
#             )
        

#         return df, res_dict


#     def sample_testdata(self, test_dataset, eta=0.0, iterations=100000, slice_nr="rand", return_res_dict=False):
#         """Autoencode test data and calculate evaluation metrics.
#         """
#         test_dataset = test_dataset
#         test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

#         keys = ['subject', 'slice_nr', "mse", "psnr", "ssim"]
#         groundtruths = []
#         samples = []
        
#         df = pd.DataFrame(columns=keys)

#         self.model.eval()
#         for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
#             if iter == iterations: 
#                 break
#             if slice_nr == "rand":
#                 slice_nr = np.random.randint(0 , batch[0].shape[2])
            
#             for i in range(len(self.model.unets)):
                
#                 if i == 0:
#                     (x0, lowres, seg), fnames, reference = self.model.preprocess(batch, **self.config.dataset, slice_idx=slice_nr, unet_number=i)
#                     x0 = x0.to(self.device)
#                     reference=reference.to(self.device)
#                     batch_size = x0.shape[0]
#                     if len(x0.shape) == 4:
#                         dim = 3
#                     if len(x0.shape) == 5:
#                         dim = 4

#                     # get all st encoded references
#                     t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
#                     t_low = t_low.type(torch.long)
#                     alphas_shape = self.alphas_cumprod[t_low].shape

                    
#                     alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)


#                     reference_images = [self.model.resize_to(reference, image_size, target_frame) for (image_size, target_frame) in zip(self.image_sizes, self.target_frames)]
#                     # reference_images = [reference_image.to(self.device) for reference_image in reference_images]
#                     noise_lows = [torch.randn_like(reference_image, device=self.device) for reference_image in reference_images]
#                     lowres_images = [torch.sqrt(alpha_t) * reference_images[i]+ torch.sqrt(1.0 - alpha_t) * noise for i, noise in enumerate(noise_lows)]
#                     lowres_images[0] = None
#                     print("test1")
#                     st_encoded_ref = [self.encode_stochastic(self.model.get_unet(i), reference_image, lowres_cond_img=lowres_images[i], lowres_noise_times=t_low) for i, reference_image in enumerate(reference_images)]
#                     print("test2")
            

#                 t_low = None
#                 if lowres is not None: 
#                     #scale up if necessary
#                     lowres = self.model.resize_to(lowres, self.image_sizes[i], target_frames=self.target_frames[i])
                    
#                     t_low = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.num_timesteps * 0.2
#                     t_low = t_low.type(torch.long)
#                     noise_low = torch.randn_like(lowres, device=self.device)
#                     alpha_t = self.alphas_cumprod[t_low].view(*alphas_shape + (1,) *dim)
#                     lowres = torch.sqrt(alpha_t) * lowres + torch.sqrt(1.0 - alpha_t) * noise_low
                
#                 # scale up if necessary
#                 # x0 = self.model.resize_to(x0, self.image_sizes[i], target_frames=self.target_frames[i])
                


#                 # xt = self.encode_stochastic(unet, x0, disable_tqdm=False, lowres_cond_img=lowres, lowres_noise_times=t_low)
#                 xt = st_encoded_ref[i]
#                 xt = torch.randn(*xt.shape, device=self.device)
#                 print(xt)
#                 eta = 0
#                 print("MIN MAX ST ENC: ", xt.min(), xt.max())
#                 unet = self.model.get_unet(i)
#                 style_emb = unet.encoder(reference_images[i])


#                 for _t in tqdm(reversed(range(self.num_timesteps)), desc="decoding ..."):
#                     t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
                    
#                     e = unet.unet(xt, t, text_embeds=style_emb, lowres_cond_img=lowres, lowres_noise_times=t_low)
#                     # Equation 12 of Denoising Diffusion Implicit Models
#                     x0_t = (
#                         torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt
#                         - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
#                     ).clamp(-1, 1)
#                     e = (
#                         (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
#                         / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
#                     )
#                     sigma = (
#                         eta
#                         * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
#                         / torch.sqrt((1 - self.alphas_cumprod[t]) / self.alphas_cumprod_prev[t])
#                     )
#                     xt = (
#                         torch.sqrt(self.alphas_cumprod_prev[t]).view(*alphas_shape + (1,) *dim) * x0_t
#                         + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
#                     )

#                     xt = xt + torch.randn_like(xt) * sigma.view(*alphas_shape + (1,) *dim) if _t != 0 else xt

#                 if i == len(self.model.unets) - 1:
#                     xt = xt.clamp(0, 1)
#                     continue
                
#                 mse, psnr, ssim = calculate_metrics(reference_images[i][0, 0, ...].cpu().numpy(), xt[0, 0, ...].cpu().numpy(), keys[2:])
#                 print(f"mse: {mse}, psnr: {psnr}, ssim: {ssim}")
                
#                 lowres = xt

#             groundtruths.append(reference)
#             samples.append(xt)

#             for i in range(batch_size):
#                 ref = reference[i,0,...].cpu().numpy()
#                 sample = xt[i,0,...].cpu().numpy()
#                 subject = extract_seven_concurrent_numbers(fnames['sa'][i])[0]
#                 mse, psnr, ssim = calculate_metrics(ref, sample, keys[2:])
#                 results = [subject, slice_nr, mse, psnr, ssim]
#                 print(results)
#                 df.loc[len(df.index)] = results

#                 # try to safe the gifs
                
#                 ref, sample = ref*255, sample*255
#                 os.makedirs(os.path.join(self.output_dir, "videos", "reconstruction", subject), exist_ok=True)
#                 videos = [Image.fromarray(image) for image in sample.astype(np.uint8)]
#                 videos[0].save(os.path.join(self.output_dir, "videos", "reconstruction", f"{subject}" ,f"sample.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
#                 videos = [Image.fromarray(image) for image in ref.astype(np.uint8)]
#                 videos[0].save(os.path.join(self.output_dir, "videos", "reconstruction", f"{subject}" ,f"reference.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
            
#             df.to_csv(os.path.join(self.output_dir, "videos", "reconstruction" ,f"data.csv"))

#         res_dict = None 
#         if return_res_dict: 
#             res_dict = dict(
#                 gts = groundtruths,
#                 preds= samples
#             )

#         return df, res_dict

#     def sample_one_image(self, image, xt=None, style_emb=None, eta=0.0):
#         """Get the result of autoencoding a single image
#         """
#         self.model.eval()

#         x0 = image.unsqueeze(dim=0).to(self.device)
#         batch_size = x0.shape[0]

#         if len(x0.shape) == 4:
#             dim = 3
#         if len(x0.shape) == 5:
#             dim = 4

#         if xt is None:
#             xt = self.encode_stochastic(x0)
#         if style_emb is None:
#             style_emb = self.model.encoder(x0)

#         x0_preds = []
#         xt_preds = []
#         for _t in tqdm(reversed(range(self.num_timesteps)), desc='decoding...', total=self.num_timesteps):
#             t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
#             e = self.model.unet(xt, t, style_emb)
#             alphas_shape = self.alphas_cumprod[t].shape

#             # Equation 12 of Denoising Diffusion Implicit Models
#             x0_t = (
#                 torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt
#                 - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
#             ).clamp(-1, 1)
#             e = (
#                 (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
#                 / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
#             )

#             sigma = (
#                 eta
#                 * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
#                 / torch.sqrt((1 - self.alphas_cumprod[t]) / self.alphas_cumprod_prev[t])
#             )
#             xt = (
#                 torch.sqrt(self.alphas_cumprod_prev[t]).view(*alphas_shape + (1,) *dim) * x0_t
#                 + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
#             )
#             xt = xt + torch.randn_like(x0) * sigma if _t != 0 else xt

#             x0_preds.append(x0_t[0])
#             xt_preds.append(xt[0])

#         result = {
#             'x0_preds': x0_preds,
#             'xt_preds': xt_preds,
#             'input': x0[0],
#             'output': xt_preds[-1],
#         }
#         return result

#     def encode_stochastic(self, model, x0, disable_tqdm=False, lowres_cond_img=None, lowres_noise_times=None):
#         """
#         Get stochastic encoded tensor xT.
#         It is necessary to obtain stochastic subcode for high-quality reconstruction, but not when training.
#         See https://github.com/phizaz/diffae/issues/17 for more details.
#         """
#         batch_size = x0.shape[0]
#         if len(x0.shape) == 4:
#             dim = 3
#         if len(x0.shape) == 5:
#             dim = 4

#         eta = 0.0
#         xt = x0.detach().clone()
#         style_emb = model.encoder(x0)

#         for _t in tqdm(range(self.num_timesteps), disable=disable_tqdm, desc='stochastic encoding...'):
#             t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t

#             e = model.unet(xt, t, text_embeds=style_emb, lowres_cond_img=lowres_cond_img, lowres_noise_times=lowres_noise_times)
#             alphas_shape = self.alphas_cumprod[t].shape
#             x0_t = (
#                 torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt
#                 - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
#             ).clamp(-1, 1)

#             e = (
#                 (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
#                 / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
#             )
#             xt = (
#                 torch.sqrt(self.alphas_cumprod_next[t]).view(*alphas_shape + (1,) *dim) * x0_t
#                 + torch.sqrt(1 - self.alphas_cumprod_next[t]).view(*alphas_shape + (1,) *dim) * e
#             )
            

#         return xt

#     def only_interpolate(self, xt_1, xt_2, style_emb_1, style_emb_2, alpha, eta=0.0):
#         """Interpolation of 2 images.
#         """

#         if len(xt_1.shape) == 4 and len(xt_2.shape) == 4:
#             dim = 3
#         elif len(xt_1.shape) == 5 and len(xt_2.shape) == 5:
#             dim = 4
#         else:
#             print("given dimensions do not match")

#         def cos(a, b):
#             a = a.contiguous().view(-1)
#             b = b.contiguous().view(-1)
#             a = torch.nn.functional.normalize(a, dim=0)
#             b = torch.nn.functional.normalize(b, dim=0)
#             return (a * b).sum()
#         theta = torch.arccos(cos(xt_1, xt_2))

#         self.model.eval()
#         batch_size = xt_1.shape[0]

#         xt = (
#             torch.sin((1 - alpha) * theta) * xt_1.flatten() + torch.sin(alpha * theta) * xt_2.flatten()
#         ) / torch.sin(theta)
#         xt = xt.view(-1, *xt_1.shape[1:])

#         style_emb = (1 - alpha) * style_emb_1 + alpha * style_emb_2
#         return xt, style_emb

#     def interpolate(self, xt_1, xt_2, style_emb_1, style_emb_2, alpha, eta=0.0):
#         """Interpolation of 2 images.
#         """

#         if len(xt_1.shape) == 4 and len(xt_2.shape) == 4:
#             dim = 3
#         elif len(xt_1.shape) == 5 and len(xt_2.shape) == 5:
#             dim = 4
#         else:
#             print("given dimensions do not match")

#         def cos(a, b):
#             a = a.contiguous().view(-1)
#             b = b.contiguous().view(-1)
#             a = torch.nn.functional.normalize(a, dim=0)
#             b = torch.nn.functional.normalize(b, dim=0)
#             return (a * b).sum()
#         theta = torch.arccos(cos(xt_1, xt_2))

#         self.model.eval()
#         batch_size = xt_1.shape[0]

#         xt = (
#             torch.sin((1 - alpha) * theta) * xt_1.flatten() + torch.sin(alpha * theta) * xt_2.flatten()
#         ) / torch.sin(theta)
#         xt = xt.view(-1, *xt_1.shape[1:])

#         style_emb = (1 - alpha) * style_emb_1 + alpha * style_emb_2

#         x0_preds = []
#         xt_preds = []
#         for _t in tqdm(reversed(range(self.num_timesteps)), desc='decoding...', total=self.num_timesteps):
#             t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
#             e = self.model.unet(xt, t, text_embeds=style_emb)
#             alphas_shape = self.alphas_cumprod[t].shape

#             # Equation 12 of Denoising Diffusion Implicit Models
#             x0_t = (
#                 torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt
#                 - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
#             ).clamp(-1, 1)
#             e = (
#                 (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
#                 / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
#             )
#             sigma = (
#                 eta
#                 * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
#                 * torch.sqrt(1 - self.alphas_cumprod[t] / self.alphas_cumprod_prev[t])
#             )
#             xt = (
#                 torch.sqrt(self.alphas_cumprod_prev[t]).view(*alphas_shape + (1,) *dim) * x0_t
#                 + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
#             )
#             xt = xt + torch.randn_like(xt_1) * sigma if _t != 0 else xt

#             x0_preds.append(x0_t[0])
#             xt_preds.append(xt[0])

#         result = {
#             'x0_preds': x0_preds,
#             'xt_preds': xt_preds,
#             'output': xt_preds[-1],
#         }
#         return result


class Sampler:
    def __init__(self, model, config, device):
        """
        Args:
            model: Diffusion Autoencoder model.
            cfg (dict): A dict of config.
        """
        self.model = model
        self.config = config

        self.device = device
        self.model.to(self.device)

        self.output_dir = self.config.output_dir
        self.num_timesteps = config.trainer.timestep_sampler.num_sample_steps
        self.betas = get_betas(config)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.zeros(1, device=self.device)], dim=0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.lpips_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(self.device)
    
    def sample_interpolated_testdata_batch(self, batch, eta=0.0, metrics=["mse", "psnr", "ssim"], slice_nr='rand'):
        if slice_nr == "rand":
            slice_nr = np.random.randint(1 , batch[0].shape[2]-1)
        prev, next = slice_nr - 1, slice_nr + 1
        x0_prev, _, _, _, fnames = self.model.preprocess(batch, **self.config.dataset, slice_idx=prev)
        x0_inter, _, _, _, fnames = self.model.preprocess(batch, **self.config.dataset, slice_idx=slice_nr)
        x0_next, _, _, _, fnames = self.model.preprocess(batch, **self.config.dataset, slice_idx=next)

        x0_prev = x0_prev.to(self.device)
        x0_inter = x0_inter.to(self.device)
        x0_next = x0_next.to(self.device)

        xt_1 = self.encode_stochastic(x0_prev)
        xt_2 = self.encode_stochastic(x0_next)

        style_emb_1 = self.model.encoder(x0_prev)
        style_emb_2 = self.model.encoder(x0_next)

        results = self.interpolate(xt_1, xt_2, style_emb_1, style_emb_2, alpha=0.5, eta=eta)
        inter = results['output'].unsqueeze(0)

        print("fnames: ", fnames[0])
        subject = extract_seven_concurrent_numbers(fnames[0])[0]
        return subject, x0_inter, inter, slice_nr

    def sample_interpolated_testdata(self, test_dataset, metrics=["mse", "psnr", "ssim"], **kwargs):
        """Autoencode test data and calculate evaluation metrics.
        """
        assert self.config.bs == 1, f"can only interpolate using batch size 1"
        test_dataset = test_dataset
        test_loader = DataLoader(test_dataset, batch_size=self.config.bs)

        keys = ['subject', 'slice_nr'] + metrics
        df = pd.DataFrame(columns=keys)

        self.model.eval()
        for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            subject, x0_inter, inter, slice_nr = self.sample_interpolated_testdata_batch(batch, **kwargs)
            
            ref = x0_inter[0,0,...].cpu().numpy()
            sample = inter[0,0,...].cpu().numpy()

            mse, psnr, ssim = calculate_metrics(ref, sample, metrics)
            results = [subject, slice_nr, mse, psnr, ssim]
            df.loc[len(df.index)] = results

            # try to safe the gifs

            ref, sample = ref*255, sample*255
            os.makedirs(os.path.join(self.output_dir, "videos", "interpolation", subject), exist_ok=True)
            videos = [Image.fromarray(image) for image in sample.astype(np.uint8)]
            videos[0].save(os.path.join(self.output_dir, "videos", "interpolation", subject ,f"sample.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
            videos = [Image.fromarray(image) for image in ref.astype(np.uint8)]
            videos[0].save(os.path.join(self.output_dir, "videos", "interpolation", subject ,f"reference.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
            df.to_csv(os.path.join(self.output_dir, "videos", "interpolation" ,f"data.csv"))

        return df

    def sample_testdata_batch(self, batch, eta=0.0, slice_nr="rand"):
        if slice_nr == "rand":
            slice_nr = np.random.randint(0 , batch[0].shape[2])
        x0, _, _, _, fnames = self.model.preprocess(batch, **self.config.dataset, slice_idx=slice_nr)

        x0 = x0.to(self.device)
        batch_size = x0.shape[0]

        if len(x0.shape) == 4:
            dim = 3
        if len(x0.shape) == 5:
            dim = 4

        xt = self.encode_stochastic(x0, disable_tqdm=False)
        style_emb = self.model.encoder(x0)

        for _t in tqdm(reversed(range(self.num_timesteps)), desc="decoding ..."):
            t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
            e, _ = self.model.unet(xt, t, text_embeds=style_emb)
            alphas_shape = self.alphas_cumprod[_t].shape
            test = torch.sqrt(1.0 / self.alphas_cumprod[_t] - 1).view(*alphas_shape + (1,) *dim)
            # Equation 12 of Denoising Diffusion Implicit Models
            x0_t = (
                torch.sqrt(1.0 / self.alphas_cumprod[_t]).view(*alphas_shape + (1,) *dim) * xt
                - torch.sqrt(1.0 / self.alphas_cumprod[_t] - 1).view(*alphas_shape + (1,) *dim) * e
            ).clamp(-1, 1)
            e = (
                (torch.sqrt(1.0 / self.alphas_cumprod[_t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
                / (torch.sqrt(1.0 / self.alphas_cumprod[_t] - 1).view(*alphas_shape + (1,) *dim))
            )
            sigma = (
                eta
                * torch.sqrt((1 - self.alphas_cumprod_prev[_t]) / (1 - self.alphas_cumprod[_t]))
                / torch.sqrt((1 - self.alphas_cumprod[_t]) / self.alphas_cumprod_prev[_t])
            )
            xt = (
                torch.sqrt(self.alphas_cumprod_prev[_t]).view(*alphas_shape + (1,) *dim) * x0_t
                + torch.sqrt(1 - self.alphas_cumprod_prev[_t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
            )
            xt = xt + torch.randn_like(x0) * sigma.view(*alphas_shape + (1,) *dim) if _t != 0 else xt
        
        return fnames, x0, xt, slice_nr


    def sample_testdata(self, test_dataset, eta=0.0):
        """Autoencode test data and calculate evaluation metrics.
        """
        test_dataset = test_dataset
        test_loader = DataLoader(test_dataset, batch_size=self.config.bs)

        keys = ['subject', 'slice_nr', "mse", "psnr", "ssim"]
        df = pd.DataFrame(columns=keys)

        self.model.eval()
        for iter, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            fnames, x0, xt, slice_nr = self.sample_interpolated_testdata_batch(batch)

            batch_size = xt.shape[0]

            for i in range(batch_size):
                ref = x0[i,0,...].cpu().numpy()
                sample = xt[i,0,...].cpu().numpy()
                subject = extract_seven_concurrent_numbers(fnames['sa'][i])[0]
                mse, psnr, ssim = calculate_metrics(ref, sample, keys[2:])
                results = [subject, slice_nr, mse, psnr, ssim]
                print(results)
                df.loc[len(df.index)] = results

                # try to safe the gifs
                
                ref, sample = ref*255, sample*255
                os.makedirs(os.path.join(self.output_dir, "videos", "reconstruction", subject), exist_ok=True)
                videos = [Image.fromarray(image) for image in sample.astype(np.uint8)]
                videos[0].save(os.path.join(self.output_dir, "videos", "reconstruction", f"{subject}" ,f"sample.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
                videos = [Image.fromarray(image) for image in ref.astype(np.uint8)]
                videos[0].save(os.path.join(self.output_dir, "videos", "reconstruction", f"{subject}" ,f"reference.gif"), save_all=True, append_images=videos[1:], duration=1000/8, loop=0)
            
            df.to_csv(os.path.join(self.output_dir, "videos", "reconstruction" ,f"data.csv"))


        return df

    def sample_one_image(self, image, xt=None, style_emb=None, eta=0.0):
        """Get the result of autoencoding a single image
        """
        self.model.eval()

        x0 = image.unsqueeze(dim=0).to(self.device)
        batch_size = x0.shape[0]

        if len(x0.shape) == 4:
            dim = 3
        if len(x0.shape) == 5:
            dim = 4

        if xt is None:
            xt = self.encode_stochastic(x0)
        if style_emb is None:
            style_emb = self.model.encoder(x0)

        x0_preds = []
        xt_preds = []
        for _t in tqdm(reversed(range(self.num_timesteps)), desc='decoding...', total=self.num_timesteps):
            t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
            e = self.model.unet(xt, t, style_emb)
            alphas_shape = self.alphas_cumprod[t].shape

            # Equation 12 of Denoising Diffusion Implicit Models
            x0_t = (
                torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt
                - torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim) * e
            ).clamp(-1, 1)
            e = (
                (torch.sqrt(1.0 / self.alphas_cumprod[t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
                / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(*alphas_shape + (1,) *dim))
            )
            sigma = (
                eta
                * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
                / torch.sqrt((1 - self.alphas_cumprod[t]) / self.alphas_cumprod_prev[t])
            )
            xt = (
                torch.sqrt(self.alphas_cumprod_prev[t]).view(*alphas_shape + (1,) *dim) * x0_t
                + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
            )
            xt = xt + torch.randn_like(x0) * sigma if _t != 0 else xt

            x0_preds.append(x0_t[0])
            xt_preds.append(xt[0])

        result = {
            'x0_preds': x0_preds,
            'xt_preds': xt_preds,
            'input': x0[0],
            'output': xt_preds[-1],
        }
        return result

    def encode_stochastic(self, x0, disable_tqdm=False):
        """
        Get stochastic encoded tensor xT.
        It is necessary to obtain stochastic subcode for high-quality reconstruction, but not when training.
        See https://github.com/phizaz/diffae/issues/17 for more details.
        """
        batch_size = x0.shape[0]
        if len(x0.shape) == 4:
            dim = 3
        if len(x0.shape) == 5:
            dim = 4

        xt = x0.detach().clone()
        style_emb = self.model.encoder(x0)
        for _t in tqdm(range(self.num_timesteps), disable=disable_tqdm, desc='stochastic encoding...'):
            t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
            t = t.type(torch.long)

            e, _ = self.model.unet(xt, t, text_embeds=style_emb)
            alphas_shape = self.alphas_cumprod[_t].shape
            x0_t = (
                torch.sqrt(1.0 / self.alphas_cumprod[_t]).view(*alphas_shape + (1,) *dim) * xt
                - torch.sqrt(1.0 / self.alphas_cumprod[_t] - 1).view(*alphas_shape + (1,) *dim) * e
            ).clamp(-1, 1)

            e = (
                (torch.sqrt(1.0 / self.alphas_cumprod[_t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
                / (torch.sqrt(1.0 / self.alphas_cumprod[_t] - 1).view(*alphas_shape + (1,) *dim))
            )
            xt = (
                torch.sqrt(self.alphas_cumprod_next[_t]).view(*alphas_shape + (1,) *dim) * x0_t
                + torch.sqrt(1 - self.alphas_cumprod_next[_t]).view(*alphas_shape + (1,) *dim) * e
            )

        return xt

    def interpolate(self, xt_1, xt_2, style_emb_1, style_emb_2, alpha, eta=0.0):
        """Interpolation of 2 images.
        """

        if len(xt_1.shape) == 4 and len(xt_2.shape) == 4:
            dim = 3
        elif len(xt_1.shape) == 5 and len(xt_2.shape) == 5:
            dim = 4
        else:
            print("given dimensions do not match")

        def cos(a, b):
            a = a.contiguous().view(-1)
            b = b.contiguous().view(-1)
            a = torch.nn.functional.normalize(a, dim=0)
            b = torch.nn.functional.normalize(b, dim=0)
            return (a * b).sum()
        theta = torch.arccos(cos(xt_1, xt_2))

        self.model.eval()
        batch_size = xt_1.shape[0]

        xt = (
            torch.sin((1 - alpha) * theta) * xt_1.flatten() + torch.sin(alpha * theta) * xt_2.flatten()
        ) / torch.sin(theta)
        xt = xt.view(-1, *xt_1.shape[1:])
        style_emb = (1 - alpha) * style_emb_1 + alpha * style_emb_2

        x0_preds = []
        xt_preds = []
        for _t in tqdm(reversed(range(self.num_timesteps)), desc='decoding...', total=self.num_timesteps):
            t = torch.ones(batch_size, dtype=torch.long, device=self.device) * _t
            e, _ = self.model.unet(xt, t, text_embeds=style_emb)
            alphas_shape = self.alphas_cumprod[_t].shape

            # Equation 12 of Denoising Diffusion Implicit Models
            x0_t = (
                torch.sqrt(1.0 / self.alphas_cumprod[_t]).view(*alphas_shape + (1,) *dim) * xt
                - torch.sqrt(1.0 / self.alphas_cumprod[_t] - 1).view(*alphas_shape + (1,) *dim) * e
            ).clamp(-1, 1)
            e = (
                (torch.sqrt(1.0 / self.alphas_cumprod[_t]).view(*alphas_shape + (1,) *dim) * xt - x0_t)
                / (torch.sqrt(1.0 / self.alphas_cumprod[_t] - 1).view(*alphas_shape + (1,) *dim))
            )
            sigma = (
                eta
                * torch.sqrt((1 - self.alphas_cumprod_prev[_t]) / (1 - self.alphas_cumprod[_t]))
                * torch.sqrt(1 - self.alphas_cumprod[_t] / self.alphas_cumprod_prev[_t])
            )
            xt = (
                torch.sqrt(self.alphas_cumprod_prev[_t]).view(*alphas_shape + (1,) *dim) * x0_t
                + torch.sqrt(1 - self.alphas_cumprod_prev[_t] - sigma**2).view(*alphas_shape + (1,) *dim) * e
            )
            xt = xt + torch.randn_like(xt_1) * sigma if _t != 0 else xt

            x0_preds.append(x0_t[0])
            xt_preds.append(xt[0])

        result = {
            'x0_preds': x0_preds,
            'xt_preds': xt_preds,
            'output': xt_preds[-1],
        }
        return result