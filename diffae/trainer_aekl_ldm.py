""" Training functions for the different models. """
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.losses.adversarial_loss import PatchAdversarialLoss
from pynvml.smi import nvidia_smi
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from einops import rearrange
import numpy as np
from torch import Tensor
import wandb
import time
# from util import log_ldm_sample_unconditioned, log_reconstructions

class PSNR(torch.nn.Module):
    def __init__(self, max_value=1.0, magnitude_psnr=True):
        super(PSNR, self).__init__()
        self.max_value = max_value
        self.magnitude_psnr = magnitude_psnr

    def forward(self, u, g):
        """

        :param u: noised image
        :param g: ground-truth image
        :param max_value:
        :return:
        """
        if self.magnitude_psnr:
            u, g = torch.abs(u), torch.abs(g)
        batch_size = u.shape[0]
        diff = (u.reshape(batch_size, -1) - g.reshape(batch_size, -1))
        square = torch.conj(diff) * diff
        max_value = g.abs().max() if self.max_value == 'on_fly' else self.max_value
        if square.is_complex():
            square = square.real
        v = torch.mean(20 * torch.log10(max_value / torch.sqrt(torch.mean(square, -1))))
        return v

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")

# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion Model
# ----------------------------------------------------------------------------------------------------------------------
def train_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    # text_encoder,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    # writer_train: SummaryWriter,
    # writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    scale_factor: float = 1.0,
) -> float:
    scaler = GradScaler()

    val_loss = eval_ldm(
        model=model,
        stage1=stage1,
        scheduler=scheduler,
        # text_encoder=text_encoder,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        # writer=writer_val,
        sample=True,
        scale_factor=scale_factor,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_ldm(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            # text_encoder=text_encoder,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            # writer=writer_train,
            scaler=scaler,
            scale_factor=scale_factor,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_ldm(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                # text_encoder=text_encoder,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                # writer=writer_val,
                sample=True,
                scale_factor=scale_factor,
                epoch=epoch,
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(model.state_dict(), str(run_dir / "best_model.pth"))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    # text_encoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    # writer: SummaryWriter,
    scaler: GradScaler,
    scale_factor: float = 1.0,
) -> None:
    model.train()

    epoch_l2_loss = 0.0
    tic = time.time()
    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, batch in pbar:
        slc_random = np.random.randint(0 , batch[0].shape[1])
        t_random = np.random.randint(0, 17)
        sa, _, _, fnames = batch
        images = sa[..., slc_random, ::5, :, :].to(device)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            # with torch.no_grad():
            #     e = stage1.encode_stage_2_inputs(images) * scale_factor
            e = images      
            # prompt_embeds = text_encoder(reports.squeeze(1))
            # prompt_embeds = prompt_embeds[0]

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
            noise_pred, _ = model(x0=e, xt=noisy_e, t=timesteps)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_l2_loss += losses["loss"].item()/len(loader) * images.shape[0]
        # writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        # for k, v in losses.items():
            # writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix({"epoch": epoch, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})
    wandb.log({f"train/loss": epoch_l2_loss}, step=epoch, commit=False)
    wandb.log({f"train/lr": get_lr(optimizer)}, step=epoch, commit=False)
    wandb.log({f"train/epoch": epoch}, step=epoch, commit=False)
    wandb.log({f"train/time (min)": (time.time()-tic)/60}, step=epoch)


@torch.no_grad()
def eval_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    # text_encoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    # writer: SummaryWriter,
    sample: bool = False,
    scale_factor: float = 1.0,
    epoch: int = 0,
) -> float:
    model.eval()
    total_losses = OrderedDict()

    for batch in loader:
        slc_random = batch[0].shape[1]//2
        sa, _, _, fnames = batch
        images = sa[..., slc_random, ::5, :, :].to(device)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        with autocast(enabled=True):
            # e = stage1.encode_stage_2_inputs(images) * scale_factor
            e = images

            # prompt_embeds = text_encoder(reports.squeeze(1))
            # prompt_embeds = prompt_embeds[0]

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
            noise_pred, _ = model(x0=e, xt=noisy_e, t=timesteps)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    # for k, v in total_losses.items():
    #     writer.add_scalar(f"{k}", v, step)
    for k, v in total_losses.items():
        wandb.log({f"val/{k}": v}, step=epoch, commit=False)

    if sample:
        log_ldm_sample_unconditioned(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            # text_encoder=text_encoder,
            spatial_shape=tuple(e.shape[1:]),
            # writer=writer,
            step=step,
            device=device,
            scale_factor=scale_factor,
            cond=e,
            epoch=epoch,
        )

    return total_losses["loss"]

@torch.no_grad()
def log_ldm_sample_unconditioned(
    model: nn.Module,
    stage1: nn.Module,
    # text_encoder,
    scheduler: nn.Module,
    spatial_shape: Tuple,
    # writer: SummaryWriter,
    step: int,
    device: torch.device,
    scale_factor: float = 1.0,
    cond: Tensor = None,
    epoch: int = 0,
) -> None:
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    # prompt_embeds = torch.cat((49406 * torch.ones(1, 1), 49407 * torch.ones(1, 76)), 1).long()
    # prompt_embeds = text_encoder(prompt_embeds.squeeze(1).to(device))
    # prompt_embeds = prompt_embeds[0]

    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred, _ = model(x0=cond, xt=latent, t=torch.asarray((t,)).to(device))
        latent, _ = scheduler.step(noise_pred, t, latent)

    # x_hat = stage1.decode(latent / scale_factor)
    x_hat = latent
    video_log = (x_hat+1)/2
    videos_to_wandb(video_log, "vis/recon", epoch)
    # img_0 = np.clip(a=x_hat[0, 0, :, :, 60].cpu().numpy(), a_min=0, a_max=1)
    # fig = plt.figure(dpi=300)
    # plt.imshow(img_0, cmap="gray")
    # plt.axis("off")
    # writer.add_figure("SAMPLE", fig, step)

def videos_to_wandb(vid, caption, step):
    vid = torch.repeat_interleave(vid, 3, dim=1)
    vid = rearrange(vid, 'b c t h w -> t c h (b w)').clamp(0,1).cpu().detach().numpy()
    vid = vid *255
    vid = wandb.Video(vid.astype(np.uint8), caption=caption)
    wandb.log({f"{caption}": vid})