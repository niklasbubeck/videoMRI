from pathlib import Path
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
from einops import rearrange
import time
import numpy as np
import random as rand
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from omegaconf import OmegaConf

from .sampler import Sampler
from .models.network_helpers import resize_video_to
from .models.model import CascadedDiffusionModel
from .models.loss import SimpleLoss, DiceLoss, SSIMLoss
from .utils import Meter, TimestepSampler, get_betas, seed_everything, training_reproducibility_cudnn, vid_to_wandb, calculate_metrics
from .sampler import Sampler

class Trainer(torch.nn.Module):
    def __init__(self, model, config, output_dir, train_dataset, test_dataset, split_batches=True, **kwargs):
        super().__init__()
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, self.config.bs)
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(self.test_dataset, self.config.bs, shuffle=False)


        self.three_d = self.config.three_d
        self.start_time = kwargs['start_time']
        self.train_days = kwargs['train_days']
        try:
            self.steps = kwargs["steps"][self.config.stage]
        except TypeError:
            self.steps = kwargs['steps']

        self.optim_state = None
        try: 
            self.optim_state = kwargs[f'optim{self.config.stage}']
        except: 
            pass

        self.prepared = False
         # init accelerator 
        self.accelerator = Accelerator(**{
            'split_batches': split_batches,
            'mixed_precision': 'no',
            'kwargs_handlers': [DistributedDataParallelKwargs(find_unused_parameters = True)], 
            })

        self.ckpt_dir = os.path.join(self.output_dir, 'models')

        seed_everything(self.config.seed)
        training_reproducibility_cudnn()

        self.log_interval = self.config.checkpoint.log_every_x_it
        self.save_interval = self.config.checkpoint.save_every_x_it
        
        self.print(f'Output a log for every {self.log_interval} steps')
        self.print(f'Save checkpoint every {self.save_interval} steps')
        self.print(f'Checkpoints are saved in {self.ckpt_dir}')
        if self.is_main: self._init_wandb(config, self.train_days)

        self.optimizer = self.get_optimizer()
        if self.optim_state is not None:
            self.print(f"Load Optimizer state dict for stage {self.config.stage}")
            self.optimizer.load_state_dict(self.optim_state)

        self.criterion_recon = SimpleLoss() if self.config.trainer.loss.recon == "mse" else SSIMLoss()
        self.criterion_seg = SimpleLoss() if self.config.trainer.loss.seg == "mse" else DiceLoss()

        self.fp16 = config.trainer.fp16
        self.grad_accum_steps = config.trainer.grad_accum_steps

        self.clip_grad_norm = config.trainer.gradient_clipping.clip_grads

        self.num_timesteps = config.trainer.timestep_sampler.num_sample_steps
        self.timestep_sampler = TimestepSampler(config.trainer.timestep_sampler, device=self.device)

        self.sampler = Sampler(model=model, config=config, device=self.device)

        self.model.to(self.device)
        self.to(self.device)

        # define betas and alphas
        self.betas = get_betas(config)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
    
    @staticmethod
    def _images_to_wandb(image_array, caption):
        images = wandb.Image(image_array, caption=caption)
        wandb.log({f"{caption}": images})

    @staticmethod 
    def _videos_to_wandb(vid, caption):
        vid = torch.repeat_interleave(vid, 3, dim=1)
        vid = rearrange(vid, 'b c t h w -> t c h (b w)').clamp(0,1).cpu().detach().numpy()
        vid = vid *255
        vid = wandb.Video(vid.astype(np.uint8), caption=caption)
        wandb.log({f"{caption}": vid})


    @staticmethod
    def _delay2str(t):
        t = int(t)
        secs = t%60
        mins = (t//60)%60
        hours = (t//3600)%24
        days = t//86400
        string = f"{secs}s"
        if mins:
            string = f"{mins}m {string}"
        if hours:
            string = f"{hours}h {string}"
        if days:
            string = f"{days}d {string}"
        return string

    @staticmethod
    def _init_wandb(config, train_days):
        wandb.init(
            name=f"{config.checkpoint.exp_name}_[{train_days}d]",
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=OmegaConf.to_container(config, resolve=True) # type: ignore
        )

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def unwrapped_unet(self):
        return self.accelerator.unwrap_model(self.unet_being_trained)

    @property
    def disable_tqdm(self):
        if self.is_main:
            return False 
        else:
            return True

    def print(self, msg):
        if not self.is_main:
            return
        return self.accelerator.print(msg)

    def _sample_noise(self, input, view):
        # sample the noise 
        batch_size = input.shape[0]
        t_img = self.timestep_sampler.sample(batch_size)
        noise_img = torch.randn_like(input, device=self.device)
        alpha_t = self.alphas_cumprod[t_img].view(*view)
        xt = torch.sqrt(alpha_t) * input + torch.sqrt(1.0 - alpha_t) * noise_img
        return xt , t_img, noise_img

    def _one_line_log(self, cur_step, loss_recon, loss_seg, batch_per_epoch, start_time, validation=False):
        s_step = f'Step: {cur_step:<6}'
        s_loss = f'Loss: {loss_recon + loss_seg:<6.4f}' if not validation else f'Val loss: {loss_recon + loss_seg:<6.4f}'
        s_loss_recon = f'Loss_recon: {loss_recon:<6.4f}' if not validation else f'Val loss_recon: {loss_recon:<6.4f}'
        s_loss_seg = f'Loss_seg: {loss_seg:<6.4f}' if not validation else f'Val loss_seg: {loss_seg:<6.4f}'
        s_epoch = f'Epoch: {(cur_step//batch_per_epoch):<4.0f}'
        s_mvid = f'Mvid: {(cur_step*self.config.bs/1e6):<6.4f}'
        s_delay = f'Elapsed time: {self._delay2str(time.time() - start_time):<10}'
        print(f'{s_step} | {s_loss} {s_epoch} {s_mvid} | {s_delay}', end='\r') # type: ignore
        if cur_step % 1000 == 0:
            print() # Start new line every 1000 steps
        
        wandb.log({
            "loss" if not validation else "val_loss" : loss_recon + loss_seg, 
            "loss_recon" if not validation else "val_loss_recon" : loss_recon,
            "loss_seg" if not validation else "val_loss_seg" : loss_seg, 
            "step": cur_step, 
            "epoch": cur_step//batch_per_epoch, 
            "mvid": cur_step*self.config.bs/1e6
        })

    def get_optimizer(self):
        optimizer_cls = getattr(torch.optim, self.config.trainer.optimizer.name)
        optimizer = optimizer_cls(self.model.parameters(), **self.config.trainer.optimizer.params)
        return optimizer
    
    def prepare(self, unet_number): 
        if self.prepared: 
            print("Trainer alredy prepared,...moving on")
            return 
        else: 
            self.unet_being_trained, self.train_loader, self.optimizer, self.test_loader = self.accelerator.prepare(self.model, self.train_loader, self.optimizer, self.test_loader)
            self.prepared = True
            return 
        

    def train(self, unet_number=0):
        # state = dict(
        #     model = self.model.state_dict(),
        # )
        # torch.save(state, os.path.join(self.ckpt_dir, "cascaded_model.pt"))
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.prepare(unet_number)

        self.print('Training start ...')
        steps = self.steps
        for self.epoch in range(self.config.trainer.epoch):
            self.model.train()
            self.epoch_start_time = time.time()
            self.print(f'---> Start train epoch {self.epoch + 1}')

            for self.iter, batch in enumerate(tqdm(self.train_loader, disable=self.disable_tqdm)):
                steps += 1
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    random = np.random.randint(0 , batch[0].shape[2])
                    sa, _, sa_seg, _, fnames = self.model.preprocess(batch, **self.config.dataset, slice_idx=random, unet_number=unet_number)
                    # handling 3d and 2d view
                    view = [1 for i in range(len(sa.shape))]
                    view[0] = -1

                    xt, t, noise_img = self._sample_noise(sa, view)
                    # xt_seg, t_seg, noise_img_seg = self._sample_noise(sa_seg, view)

                    # t_low = None
                    # if lowres is not None: 
                    #     # augment the lowres image too 
                    #     t_low= self.timestep_sampler.sample(batch_size)
                    #     noise_low = torch.randn_like(lowres, device=self.device)
                    #     alpha_t = self.alphas_cumprod[t_low].view(*view)
                    #     lowres = torch.sqrt(alpha_t) * lowres + torch.sqrt(1.0 - alpha_t) * noise_low

                    out_recon, out_seg = self.unet_being_trained(sa, xt, t.float())

                    pred_recon = xt - out_recon
                    pred_seg = xt - out_seg

                    if self.is_main and self.iter % 100 == 0:
                        vid_recon = torch.cat([sa, pred_recon], dim=-2)
                        vid_seg = torch.cat([sa_seg, pred_seg], dim=-2)
                        self._videos_to_wandb(vid_recon, 'recon')
                        self._videos_to_wandb(vid_seg, "seg")

                    loss_recon = self.criterion_recon(pred_recon, sa)
                    loss_seg = self.criterion_seg(pred_seg, sa_seg)
                    loss = loss_recon + loss_seg

                if self.is_main: self._one_line_log(steps, loss_recon, loss_seg, len(self.train_loader), self.start_time)
                self.accelerator.backward(loss)
                self.optimizer.step()
        

                # Udpate the logging stuff 
                metrics=["mse", "psnr", "ssim"]
                if (steps + 1) % self.log_interval == 0:
                    with torch.no_grad():
                        mses=[]
                        psnrs=[]
                        ssims=[]
                        gts = []
                        samples = []
                        # get interpolation results
                        for iter, batch in tqdm(enumerate(self.test_loader), disable=self.disable_tqdm):
                            if iter >=1:
                                break
                            for n in [5]:
                                subject, gt, sample, slice_nr1 = self.sampler.sample_interpolated_testdata_batch(batch, metrics=metrics, slice_nr=n)
                                gts.append(gt)
                                samples.append(sample)

                                ref = gt[0,0,...].cpu().numpy()
                                sample = sample[0,0,...].cpu().numpy()

                                mse, psnr, ssim = calculate_metrics(ref, sample, metrics)
                                mses.append(mse)
                                psnrs.append(psnr)
                                ssims.append(ssim)
                        gts = torch.stack(gts, dim=0).squeeze(0)
                        samples = torch.stack(samples, dim=0).squeeze(0)
                        vid = torch.cat([gts, samples], dim=-2)
                        if self.is_main:
                            self._videos_to_wandb(vid, "eval_inter")


                        mse = sum(mses) / len(mses)
                        psnr = sum(psnrs) / len(psnrs)
                        ssim = sum(ssims) / len(ssims)
                        if self.is_main:
                            wandb.log({
                                "val_mse_inter" : mse, 
                                "val_psnr_inter" : psnr,
                                "val_ssim_inter" : ssim, 
                                "step": steps, 
                                "epoch": steps//len(self.train_loader), 
                                "mvid": steps*self.config.bs/1e6
                            })


                        mses=[]
                        psnrs=[]
                        ssims=[]
                        gts=[]
                        samples=[]
                        # get reconstruction results
                        for iter, batch in tqdm(enumerate(self.test_loader), disable=self.disable_tqdm):
                            if iter >=1:
                                break 
                            for n in [5]:
                                fnames, gt, sample, slice_nr = self.sampler.sample_testdata_batch(batch, slice_nr=n)
                                gts.append(gt)
                                samples.append(sample)
                                ref = gt[0,0,...].cpu().numpy()
                                sample = sample[0,0,...].cpu().numpy()

                                mse, psnr, ssim = calculate_metrics(ref, sample, metrics)
                                mses.append(mse)
                                psnrs.append(psnr)
                                ssims.append(ssim)

                        mse = sum(mses) / len(mses)
                        psnr = sum(psnrs) / len(psnrs)
                        ssim = sum(ssims) / len(ssims)
                        if self.is_main:
                            wandb.log({
                                "val_mse_recon" : mse, 
                                "val_psnr_recon" : psnr,
                                "val_ssim_recon" : ssim, 
                                "step": steps, 
                                "epoch": steps//len(self.train_loader), 
                                "mvid": steps*self.config.bs/1e6
                            })

                        gts = torch.stack(gts, dim=0).squeeze(0)
                        samples = torch.stack(samples, dim=0).squeeze(0)
                        vid = torch.cat([gts, samples], dim=-2)
                        if self.is_main:
                            self._videos_to_wandb(vid, "eval_recon")

                if self.is_main:
                    if (steps + 1) % self.save_interval == 0:
                        additional_data = {
                            "steps": steps,
                            "time_elapsed": time.time() - self.start_time
                        }
                        self.save_ckpt(name=f'ckpt.{self.train_days}.pt', **additional_data)

            

            epoch_elapsed_time = time.time() - self.epoch_start_time
            self.print(f'Epoch {self.epoch + 1} done. ({epoch_elapsed_time:.1f} sec)')

        self.print('Training done')

    def save_ckpt(self, name, **kwargs):
        state = dict(
            epoch = self.epoch + 1,
            model = self.unet_being_trained.state_dict(),
            optimizer = self.optimizer.state_dict(),
            **kwargs
        )
        torch.save(state, os.path.join(self.ckpt_dir, name))
        print(f'Saving checkpoint to {os.path.join(self.ckpt_dir, name)}')

