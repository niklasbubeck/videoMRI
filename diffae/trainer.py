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
from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora

import torch.nn.functional as F
from monai.networks.utils import one_hot
from generative.networks.nets import AutoencoderKL
from scipy.ndimage.morphology import distance_transform_edt
from ema_pytorch import EMA


from .sampler import Sampler
from .models.network_helpers import resize_video_to, normalize_neg_one_to_one, unnormalize_zero_to_one
from .models.model import CascadedDiffusionModel
from .models.loss import SimpleLoss, DiceLoss, SSIMLoss, VideoSSIMLoss, SegmentationCriterion, TemporalGradLoss
from .utils import Meter, TimestepSampler, get_betas, seed_everything, training_reproducibility_cudnn, vid_to_wandb, calculate_metrics, find_bounding_box3D, center_crop_bbox
from .sampler import Sampler
from .models.network_helpers import unnormalize_zero_to_one, normalize_neg_one_to_one

class Trainer(torch.nn.Module):
    def __init__(self, ema_model, model, aekl_model, config, output_dir, train_dataset, test_dataset, split_batches=True, **kwargs):
        super().__init__()
        self.model = model
        self.ema_model = ema_model
        self.aekl_model = aekl_model
        self.config = config
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, self.config.bs, num_workers=min(self.config.bs, 16), shuffle=True)
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(self.test_dataset, self.config.bs, num_workers=min(self.config.bs, 16), shuffle=False)
        self.clip_grad_norm = self.config.trainer.clip_grad_norm
        print(f"Training with Clip grad norm: {self.clip_grad_norm}")

        self.use_latent = False
        if self.aekl_model is not None: 
            self.use_latent = True

        self.three_d = self.config.three_d
        self.start_time = kwargs['start_time']
        self.train_days = kwargs['train_days']
        try:
            self.steps = kwargs["steps"][self.config.stage]
        except TypeError:
            self.steps = kwargs['steps']

        self.optim_state = None
        try: 
            self.optim_state = kwargs[f'optimizer']
        except Exception as e: 
            print("Error when loading optimizer state dict. If this is the begin of the training you can ignore this error: \n ",e)

        self.schedule_state = None 
        try: 
            self.schedule_state = kwargs[f'scheduler']
        except Exception as e : 
            print("Error when loading optimizer state dict. If this is the begin of the training you can ignore this error: \n ",e)

        self.prepared = False
         # init accelerator 
        self.accelerator = Accelerator(**{
            'split_batches': self.config.trainer.accelerator.split_batches,
            'mixed_precision': self.config.trainer.accelerator.mixed_precision,
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
        if self.is_main: wandb.watch(self.model)


        self.optimizer = self.get_optimizer()
        if self.optim_state is not None:
            self.print(f"Load Optimizer state dict for stage {self.config.stage}")
            self.optimizer.load_state_dict(self.optim_state)

        self.scheduler = self.get_scheduler(self.optimizer)
        if self.schedule_state is not None:
            self.print(f"Load Scheduler state dict for stage {self.config.stage}")
            self.scheduler.load_state_dict(self.schedule_state)

        self.criterion_recon = SimpleLoss() if self.config.trainer.loss.recon == "mse" else VideoSSIMLoss()
        self.criterion_seg = SimpleLoss() if self.config.trainer.loss.seg == "mse" else SegmentationCriterion(type="dice")
        self.temp_grad_loss = TemporalGradLoss()

        self.num_timesteps = config.trainer.timestep_sampler.num_sample_steps
        self.timestep_sampler = TimestepSampler(config.trainer.timestep_sampler, device=self.device)

        if self.aekl_model is not None: 
            self.aekl_model.eval()

        # put everything to correct device 
        self.model.to(self.device)
        if self.aekl_model is not None:
            self.aekl_model.to(self.device)
        self.to(self.device)

        # define betas and alphas
        self.betas = get_betas(config)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
    
    def _images_to_wandb(self, image_array, caption):
        images = wandb.Image(image_array.clamp(0,1), caption=caption)
        wandb.log({f"{caption}": images})

    def _videos_to_wandb(self, vid, caption):
        vid = torch.repeat_interleave(vid, 3, dim=1)
        vid = rearrange(vid, 'b c t h w -> t c h (b w)').clamp(0,1).cpu().detach().numpy()
        vid = vid *255
        vid = wandb.Video(vid.astype(np.uint8), caption=caption)
        wandb.log({f"{caption}": vid})

    def _vid_or_image_to_wandb(self, vid_img, caption):
        print("vid_img shape: ", vid_img.shape)
        if self.config.dataset.normalize:
            vid_img = unnormalize_zero_to_one(vid_img)
        if len(vid_img.shape) == 3:
            vid_img = vid_img.unsqueeze(-3)
        if len(vid_img.shape) == 4:
            vid_img = vid_img.unsqueeze(-3)
            # self._images_to_wandb(vid_img, caption)
        if len(vid_img.shape) == 5:
            self._videos_to_wandb(vid_img, caption)
        else: 
            raise Exception("given shape of data cannot be handled")


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

    def print(self, msg, **kwargs):
        if not self.is_main:
            return
        return self.accelerator.print(msg, **kwargs)
    
    def wait(self):
        return self.accelerator.wait_for_everyone()

    def _create_tgrad_mask(self, gt_seg, crop_size):
        image_size=(gt_seg.shape[-1], gt_seg.shape[-2])
        box = torch.ones(image_size, dtype=torch.uint8)
        masks = []
        for seg in gt_seg:
            top, bottom, left, right = center_crop_bbox(image_size, crop_size=crop_size)

            box[bottom:top, left:right] = 0 # for the cardiac center region, maybe you can use segmentation groundtruth to localize it. and don't forget to add offset
            box = torch.from_numpy(distance_transform_edt(box) / 4)  # laplace weighting decay
            box = torch.exp(-0.2*box).type(torch.float32)
            weighting_mask = 1 - box
            masks.append(weighting_mask[None,None,None,...])
        return torch.cat(masks, dim=0)

    def _sample_noise(self, input, view, noise=None, time=None):
        # sample the noise 
        batch_size = input.shape[0]
        if time is not None:
            t_img = time
        else:
            t_img = self.timestep_sampler.sample(batch_size)
        if noise is not None: 
            noise_img = noise
        else:
            noise_img = torch.randn_like(input, device=self.device)
        alpha_t = self.alphas_cumprod[t_img].view(*view)
        xt = torch.sqrt(alpha_t) * input + torch.sqrt(1.0 - alpha_t) * noise_img
        return xt , t_img, noise_img


    def _one_line_log(self, cur_step, loss_recon, loss_seg, batch_per_epoch, start_time, validation=False, **additional):
        s_step = f'Step: {cur_step:<6}'
        s_loss = f'Loss: {loss_recon + loss_seg:<6.4f}' if not validation else f'Val loss: {loss_recon + loss_seg:<6.4f}'
        s_loss_recon = f'Loss_recon: {loss_recon:<6.4f}' if not validation else f'Val loss_recon: {loss_recon:<6.4f}'
        s_loss_seg = f'Loss_seg: {loss_seg:<6.4f}' if not validation else f'Val loss_seg: {loss_seg:<6.4f}'
        s_epoch = f'Epoch: {(cur_step//batch_per_epoch):<4.0f}'
        s_mvid = f'Mvid: {(cur_step*self.config.bs/1e6):<6.4f}'
        s_delay = f'Elapsed time: {self._delay2str(time.time() - start_time):<10}'
        self.print(f'{s_step} | {s_loss} {s_epoch} {s_mvid} | {s_delay}', end='\r') # type: ignore
        if cur_step % 1000 == 0:
            self.print("\n") # Start new line every 1000 steps
        
        wandb.log({
            "loss" if not validation else "val_loss" : loss_recon + loss_seg, 
            "loss_recon" if not validation else "val_loss_recon" : loss_recon,
            "loss_seg" if not validation else "val_loss_seg" : loss_seg, 
            "step": cur_step, 
            "epoch": cur_step//batch_per_epoch, 
            "mvid": cur_step*self.config.bs/1e6,
            **additional
        })

    def get_optimizer(self, params=None):
        params = params if params is not None else self.model.parameters()
        optimizer_cls = getattr(torch.optim, self.config.trainer.optimizer.name)
        optimizer = optimizer_cls(params, **self.config.trainer.optimizer.params)
        return optimizer
    
    def get_scheduler(self, optimizer):
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.config.trainer.scheduler.name)
        scheduler = scheduler_cls(optimizer, **self.config.trainer.scheduler.params)
        return scheduler

    def prepare(self, model): 
        if self.prepared: 
            self.print("Trainer alredy prepared,...moving on")
            return 
        else: 
            self.unet_being_trained, self.aekl_model, self.train_loader, self.optimizer, self.test_loader = self.accelerator.prepare(model, self.aekl_model, self.train_loader, self.optimizer, self.test_loader)
            self.prepared = True
            return 

    @torch.no_grad()
    def evalutate_train(self, model, mode="interpolation", metrics=["mse", "psnr", "ssim"], num_timesteps=1000, noise=False):
        assert mode in ["interpolation", "reconstruction"], f"Given mode is not known: {mode}"
        key_prefix = f"val/{mode}/{num_timesteps}/{noise}"
        sampler = Sampler(model=model, aekl_model=self.aekl_model, config=self.config, device=self.device)
        sampler.update_betas_and_alphas(num_timesteps)
        fcn = sampler.sample_testdata_batch if mode == "reconstruction" else sampler.sample_interpolated_testdata_batch

        mses, psnrs, ssims, gts, dices, dices_1, dices_2, dices_3, samples, gts_seg, samples_seg = [], [], [], [], [], [], [], [], [], [], []
        # get interpolation results
        for iter, batch in tqdm(enumerate(self.test_loader), disable=self.disable_tqdm):
            if iter >= 1:
                break
            for slice_nr in [5]:
                subjects, gt_recons, sample_recons, gt_segs, sample_segs, _ = fcn(batch, slice_nr=slice_nr, noise=noise)

                gt_segs = [None] * gt_recons.shape[0] if gt_segs is None else gt_segs
                sample_segs = [None] * gt_recons.shape[0] if sample_segs is None else sample_segs
                for gt, sample, gt_seg, seg in zip(gt_recons, sample_recons, gt_segs, sample_segs):
                    mses.append(calculate_metrics(gt[0].cpu().numpy(), sample[0].cpu().numpy(), metrics=["mse"])[0])
                    psnrs.append(calculate_metrics(gt[0].cpu().numpy(), sample[0].cpu().numpy(), metrics=["psnr"])[0])
                    ssims.append(calculate_metrics(gt[0].cpu().numpy(), sample[0].cpu().numpy(), metrics=["ssim"])[0])

                    if seg is not None:
                        seg = seg[0].cpu().numpy()
                        dice_loss, dice = self.criterion_seg(one_hot(gt_seg.squeeze(0), num_classes=4), seg)
                        dices.append(1 - dice_loss.item())
                        dices_1.append(dice[0].item())
                        dices_2.append(dice[1].item())
                        dices_3.append(dice[2].item())

                    gts.append(gt)
                    samples.append(sample)

                    if seg is not None:
                        gts_seg.append(gt_seg)
                        samples_seg.append(seg)

        vid = torch.cat([torch.stack(gts, dim=0), torch.stack(samples, dim=0)], dim=-2)

        if seg is not None:
            gts_seg = torch.stack(gts_seg, dim=0).squeeze(0)
            samples_seg = torch.stack(samples_seg, dim=0).squeeze(0)
            vid_seg = torch.cat([gts_seg, torch.argmax(samples_seg, dim=1).unsqueeze(1)], dim=1)

        if self.is_main:
            self._vid_or_image_to_wandb(vid, key_prefix)
            if seg is not None:
                self._vid_or_image_to_wandb(vid_seg / 4, key_prefix)

        metrics = {f"{key_prefix}/mse": sum(mses) / len(mses),
                   f"{key_prefix}/psnr": sum(psnrs) / len(psnrs),
                   f"{key_prefix}/ssim": sum(ssims) / len(ssims)}
        if seg is not None:
            metrics.update({
                f"{key_prefix}/dice_inter": sum(dices) / len(dices),
                f"{key_prefix}/dice_1": sum(dices_1) / len(dices_1),
                f"{key_prefix}/dice_2": sum(dices_2) / len(dices_2),
                f"{key_prefix}/dice_3": sum(dices_3) / len(dices_3)})
        if self.is_main:
            wandb.log({**metrics, "step": self.steps, "epoch": self.steps // len(self.train_loader), "mvid": self.steps * self.config.bs / 1e6})

    def train(self, unet_number=0):
        self.wait()
        torch.autograd.set_detect_anomaly(True)
        self.prepare(self.model)

        self.print('Training start ...')
        steps = self.steps
        for self.epoch in range(self.config.trainer.epoch):
            self.model.train()
            self.epoch_start_time = time.time()
            self.print(f'---> Start train epoch {self.epoch + 1}')

            for self.iter, batch in enumerate(tqdm(self.train_loader, disable=self.disable_tqdm)):
                steps += 1
                self.optimizer.zero_grad()
                with self.accelerator.autocast():
                    random = None #np.random.randint(0 , batch[0].shape[2])
                    sa, _, sa_seg, _, fnames = self.model.preprocess(batch, **self.config.dataset, slice_idx=random, unet_number=unet_number)
                    sa , idx = batch['img'], batch['index']
                    gt_sa = sa.detach().clone()
                    # handling 3d and 2d view
                    view = [1 for i in range(len(sa.shape))]
                    view[0] = -1
                    if self.use_latent:
                        sa = self.aekl_model.encode_stage_2_inputs(sa)
                        print("after latent: ", sa.shape)


                    xt, t, noise_img = self._sample_noise(sa, view)

                    out_recon, out_seg = self.unet_being_trained(sa, xt, t.float())
                    

                    pred_recon = (xt - out_recon * torch.sqrt(1-self.alphas_cumprod[t].view(*view))) / torch.sqrt(self.alphas_cumprod[t].view(*view))
                    pred_recon = pred_recon.clamp(-1, 1)

                    if out_seg is not None:
                        pred_seg = (xt - out_seg * torch.sqrt(1-self.alphas_cumprod[t].view(*view))) / torch.sqrt(self.alphas_cumprod[t].view(*view))
                        pred_seg = F.softmax(pred_seg, dim=1)

                    if self.use_latent:
                        pred_recon = self.aekl_model.decode_stage_2_outputs(pred_recon)
                        print(out_recon.shape)
                        if out_seg is not None:
                            pred_seg = self.aekl_model.decode_stage_2_outputs(pred_recon)

                    if out_seg is not None: 
                        pred_seg = F.softmax(pred_seg, dim=1)

                    if self.is_main and self.iter % 1000 == 0:
                        vid_recon = torch.cat([gt_sa, pred_recon], dim=-2)
                        self._vid_or_image_to_wandb(vid_recon, f'train_recon')
                        vid_noise = torch.cat([sa, xt, noise_img, out_recon])
                        self._vid_or_image_to_wandb(vid_noise, f"train_noise")
                        if out_seg is not None:
                            vid_seg = torch.cat([sa_seg, torch.argmax(pred_seg, dim=1).unsqueeze(1)], dim=-2)
                            self._vid_or_image_to_wandb(vid_seg/4, "train_seg")

                    loss_recon = self.criterion_recon(pred_recon, gt_sa)
                    print("loss_recon: ", loss_recon)
                    loss_seg = 0
                    if out_seg is not None:
                        loss_seg, dice_seg = self.criterion_seg(pred_seg, one_hot(sa_seg, num_classes=4), one_hot_enc=False)
                        loss_seg *= self.config.trainer.loss.weight
                    loss = loss_recon + loss_seg if out_seg is not None else loss_recon

                    grad_t_loss = 0
                    if self.config.trainer.loss.tgrad:
                        mask = self._create_tgrad_mask(sa_seg, crop_size=64)
                        grad_t_loss = self.temp_grad_loss(pred_recon, mask=mask.to(self.device)) 
                        loss = loss + grad_t_loss * self.config.trainer.loss.tweight
                    

                if self.is_main: self._one_line_log(steps, loss_recon, loss_seg, len(self.train_loader), self.start_time, additional={"tgrad": grad_t_loss, "learning_rate": self.optimizer.param_groups[0]['lr'], "lr_sched": self.scheduler.get_last_lr()})
                self.accelerator.backward(loss)
                if self.clip_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.unet_being_trained.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.ema_model.update()
        

                
                if (steps + 1) % self.log_interval == 0 and self.is_main:
                    self.evalutate_train(self.ema_model.ema_model, mode="reconstruction", num_timesteps=20, noise=False)
                    self.evalutate_train(self.ema_model.ema_model, mode="reconstruction", num_timesteps=50, noise=False)
                    self.evalutate_train(self.ema_model.ema_model, mode="reconstruction", num_timesteps=100, noise=False)
                    self.evalutate_train(self.ema_model.ema_model, mode="reconstruction", num_timesteps=20, noise=True)
                    self.evalutate_train(self.ema_model.ema_model, mode="reconstruction", num_timesteps=50, noise=True)
                    self.evalutate_train(self.ema_model.ema_model, mode="reconstruction", num_timesteps=100, noise=True)

                if self.is_main:
                    if (steps + 1) % self.save_interval == 0:
                        additional_data = {
                            "steps": steps,
                            "time_elapsed": time.time() - self.start_time
                        }
                        self.save_ckpt(name=f'ckpt.{self.train_days}.pt', **additional_data)

            
            self.scheduler.step()
            epoch_elapsed_time = time.time() - self.epoch_start_time
            self.print(f'Epoch {self.epoch + 1} done. ({epoch_elapsed_time:.1f} sec)')

        self.print('Training done')



    def train_finetune(self, unet_number=0, fine_tune=True):
        torch.autograd.set_detect_anomaly(True)
        self.prepare(unet_number)
        if fine_tune:
            # Make parameters in the unet not require gradients
            for param in self.model.unet.parameters():
                param.requires_grad = False

        self.print('Finetune training start ...')
        steps = self.steps
        for self.epoch in range(self.config.trainer.epoch):
            self.model.train()
            self.epoch_start_time = time.time()
            self.print(f'---> Start train epoch {self.epoch + 1}')

            for self.iter, batch in enumerate(tqdm(self.train_loader, disable=self.disable_tqdm)):
                steps += 1
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    random = np.random.randint(0+1 , batch[0].shape[2]-1)
                    sa_prev, _, _, _, fnames = self.model.preprocess(batch, **self.config.dataset, slice_idx=random - 1, unet_number=unet_number)
                    sa_inter, _, _, _, fnames = self.model.preprocess(batch, **self.config.dataset, slice_idx=random, unet_number=unet_number)
                    sa_next, _, _, _, fnames = self.model.preprocess(batch, **self.config.dataset, slice_idx=random + 1, unet_number=unet_number)
                    gt_sa_prev = sa_prev.detach().clone()
                    gt_sa_inter = sa_inter.detach().clone()
                    gt_sa_next = sa_next.detach().clone()

                    # handling 3d and 2d view
                    view = [1 for i in range(len(sa_prev.shape))]
                    view[0] = -1

                    if self.use_latent:
                        sa_prev = self.aekl_model.encode_stage_2_inputs(sa_prev)
                        sa_next = self.aekl_model.encode_stage_2_inputs(sa_next)

                    # interpolate xt from prev and next
                    with torch.no_grad():
                        sem_prev = self.model.encoder(sa_prev)
                        sem_next = self.model.encoder(sa_next)

                    sem_inter = (sem_prev+sem_next) // 2


                    if self.use_latent:
                        pred_recon = self.aekl_model.decode_stage_2_outputs(pred_recon)

                    loss = self.criterion_recon(gt_inter, sem_inter)
                    
                    
                if self.is_main: self._one_line_log(steps, loss, 0, len(self.train_loader), self.start_time, additional={"learning_rate": self.optimizer.param_groups[0]['lr'], "lr_sched": self.scheduler.get_last_lr()})
                self.accelerator.backward(loss)
                if self.clip_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.unet_being_trained.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.ema_model.update()
        

                # Udpate the logging stuff 
                if self.is_main:
                    if (steps + 1) % self.save_interval == 0:
                        self.evalutate_train(self.ema_model.ema_model, mode="interpolation", num_timesteps=5, noise=False)
                if self.is_main:
                    if (steps + 1) % self.save_interval == 0:
                        additional_data = {
                            "steps": steps,
                            "time_elapsed": time.time() - self.start_time
                        }
                        self.save_ckpt(name=f'ckpt.{self.train_days}.pt', **additional_data)

            
            self.scheduler.step()
            epoch_elapsed_time = time.time() - self.epoch_start_time
            self.print(f'Epoch {self.epoch + 1} done. ({epoch_elapsed_time:.1f} sec)')

        self.print('Training done')


    def save_ckpt(self, name, **kwargs):
        state = dict(
            epoch = self.epoch + 1,
            ema_model = self.ema_model.state_dict(),
            model = self.unet_being_trained.state_dict(),
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            **kwargs
        )
        torch.save(state, os.path.join(self.ckpt_dir, name))
        self.print(f'Saving checkpoint to {os.path.join(self.ckpt_dir, name)}')

