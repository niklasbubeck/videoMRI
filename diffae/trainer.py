from pathlib import Path
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
from einops import rearrange
import time
import numpy as np
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from omegaconf import OmegaConf

from .models.network_helpers import resize_video_to
from .models.model import CascadedDiffusionModel
from .models.loss import SimpleLoss
from .utils import Meter, TimestepSampler, get_betas, seed_everything, training_reproducibility_cudnn, vid_to_wandb
from .sampler import Sampler

class Trainer(torch.nn.Module):
    def __init__(self, model, config, output_dir, train_dataset, three_d=False, split_batches=True, **kwargs):
        super().__init__()
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, self.config.bs)

        self.three_d = three_d
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

        self.sampler = Sampler(self.model, self.config, self.device)


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

        self.criterion = SimpleLoss()

        self.fp16 = config.trainer.fp16
        self.grad_accum_steps = config.trainer.grad_accum_steps

        self.clip_grad_norm = config.trainer.gradient_clipping.clip_grads

        self.num_timesteps = config.trainer.timestep_sampler.num_sample_steps
        self.timestep_sampler = TimestepSampler(config.trainer.timestep_sampler, device=self.device)
 
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


    def _one_line_log(self, cur_step, loss, batch_per_epoch, start_time, validation=False):
        s_step = f'Step: {cur_step:<6}'
        s_loss = f'Loss: {loss:<6.4f}' if not validation else f'Val loss: {loss:<6.4f}'
        s_epoch = f'Epoch: {(cur_step//batch_per_epoch):<4.0f}'
        s_mvid = f'Mvid: {(cur_step*self.config.bs/1e6):<6.4f}'
        s_delay = f'Elapsed time: {self._delay2str(time.time() - start_time):<10}'
        print(f'{s_step} | {s_loss} {s_epoch} {s_mvid} | {s_delay}', end='\r') # type: ignore
        if cur_step % 1000 == 0:
            print() # Start new line every 1000 steps
        
        wandb.log({
            "loss" if not validation else "val_loss" : loss, 
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
            self.unet_being_trained, self.train_loader, self.optimizer = self.accelerator.prepare(self.model.get_unet(unet_number), self.train_loader, self.optimizer)
            self.prepared = True
            return 

    def train(self, unet_number=0):
        state = dict(
            model = self.model.state_dict(),
        )
        torch.save(state, os.path.join(self.ckpt_dir, "cascaded_model.pt"))
        self.train_loss_meter = Meter()
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.prepare(unet_number)


        print('Training start ...')
        steps = self.steps
        for self.epoch in range(self.config.trainer.epoch):
            self.model.train()
            self.epoch_start_time = time.time()
            print(f'---> Start train epoch {self.epoch + 1}')

            for self.iter, batch in enumerate(tqdm(self.train_loader, disable=self.disable_tqdm)):
                steps += 1
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    random = np.random.randint(0 , batch[0].shape[2])
                    (x0, lowres, seg), fnames, reference = self.model.preprocess(batch, **self.config.dataset, slice_idx=random, unet_number=unet_number)
                    x0 = x0.to(self.device)
                    if lowres is not None:
                        lowres = lowres.to(self.device)

                    # handling three d
                    view = [1 for i in range(len(x0.shape))]
                    view[0] = -1

                    # sample the noise 
                    batch_size = x0.shape[0]
                    t_img = self.timestep_sampler.sample(batch_size)
                    noise_img = torch.randn_like(x0, device=self.device)
                    alpha_t = self.alphas_cumprod[t_img].view(*view)
                    xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * noise_img

                    t_low = None
                    if lowres is not None: 
                        # augment the lowres image too 
                        t_low= self.timestep_sampler.sample(batch_size)
                        noise_low = torch.randn_like(lowres, device=self.device)
                        alpha_t = self.alphas_cumprod[t_low].view(*view)
                        lowres = torch.sqrt(alpha_t) * lowres + torch.sqrt(1.0 - alpha_t) * noise_low

                    outputs = self.unet_being_trained(x0, xt, t_img.float(), lowres, lowres_noise_times=t_low)
                    loss = self.criterion(outputs, noise_img)
                    loss /= self.grad_accum_steps

                if self.is_main: self._one_line_log(steps, loss, len(self.train_loader), self.start_time)
                # self.scaler.scale(loss).backward()
                self.accelerator.backward(loss)
                self.train_loss_meter.update(loss.item())

                if (self.iter + 1) % self.grad_accum_steps == 0:
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    # self.scaler.update()
                    # self.optimizer.zero_grad()

                
                # Udpate the logging stuff 
                # if self.is_main:
                    # if (steps + 1) % self.log_interval == 0:
                    #     with torch.no_grad():
                    #         results = self.infer_interpolate(x0[0], x0[1], [0.0, 0.5, 1.0])
                    #         results = torch.stack([torch.from_numpy(res['output']) for res in results])
                    #     if not self.three_d:
                    #         print(results.shape)
                    #         self._images_to_wandb(rearrange(results, 'b c h w -> (c h) (b w)'), caption="interpolation")
                    #         print(
                    #             'epoch: {}/{}, iter: {}/{}, steps: {} loss {:.3f}'.format(
                    #                 self.epoch + 1, self.config.trainer.epoch,
                    #                 self.iter + 1, len(self.train_loader),
                    #                 steps,
                    #                 self.train_loss_meter.avg,
                    #             )
                    #         )
                    #     else: 
                    #         vid_to_wandb(results)
                        

                self.train_loss_meter.reset()
                if self.is_main:
                    if (steps + 1) % self.save_interval == 0:
                        additional_data = {
                            "steps": steps,
                            "time_elapsed": time.time() - self.start_time
                        }
                        self.save_ckpt(name=f'stage_{unet_number}/ckpt.{self.train_days}.pt', **additional_data)


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

    def infer_interpolate(self, image_1, image_2, alphas):
        """Interpolation of 2 images.

        Args:
            image_1, image_2 (PIL Image): A single PIL Image.
            alphas (float or List[float]): Interpolation parameter(s).

        Returns:
            result (List[dict]): A List of result which has the following keys,
                output (numpy.ndarray): A output (autoencoded) image array.
                x0_preds (List[numpy.ndarray]): A list of predicted x0 per timestep.
                xt_preds (List[numpy.ndarray]): A list of predicted xt per timestep.
        """
        device = 'cuda'
        if isinstance(alphas, (int, float)):
            alphas = [alphas]

        x0_1 = image_1.unsqueeze(dim=0).to(device)
        x0_2 = image_2.unsqueeze(dim=0).to(device)
        xt_1 = self.sampler.encode_stochastic(x0_1)
        xt_2 = self.sampler.encode_stochastic(x0_2)

        style_emb_1 = self.model.encoder(x0_1)
        style_emb_2 = self.model.encoder(x0_2)
        

        results = []
        for i, alpha in enumerate(alphas):
            print(f" \n Interpolate {i+1}/{len(alphas)} \n")
            result = self.sampler.interpolate(xt_1, xt_2, style_emb_1, style_emb_2, alpha)

            # Unnormalize and to numpy.ndarray
            for k, v in result.items():
                if isinstance(v, list):
                    for i, x in enumerate(result[k]):
                        assert torch.is_tensor(x)
                        result[k][i] = x.cpu().detach().numpy()
                elif torch.is_tensor(v):
                    result[k] = x.cpu().detach().numpy()

            results.append(result)

        return results

