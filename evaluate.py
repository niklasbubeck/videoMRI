import argparse
from diffae.interface import DiffusionAutoEncodersInterface
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import nibabel as nib 
from einops import rearrange
import torch 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--resume", type=str, default="auto")
    parser.add_argument("--bs", type=int, default="-1")
    parser.add_argument("--lr", type=float, default="-1")
    parser.add_argument("--steps", type=int, default=-1, help="diffusion steps")
    parser.add_argument("--debug", type=bool,default=False, help="if to log with wandb")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, vars(args))

    # Overwrite config values with args
    exp_name = args.config.split("/")[-1].split(".")[0] # get config file name
    config.checkpoint.exp_name = exp_name
    config.dataset.num_frames = int(config.dataset.fps * config.dataset.duration)
    if args.bs != -1:
        config.dataloader.batch_size = args.bs
        config.dataloader.num_workers = args.bs
        config.checkpoint.batch_size = min(args.bs, config.checkpoint.batch_size)
    if args.lr != -1:
        config.trainer.lr = args.lr
    if args.steps != -1:
        if config.imagen.get("elucidated", True) == True:
            config.imagen.num_sample_steps = args.steps
        else:
            config.imagen.timesteps = args.steps


    diffae = DiffusionAutoEncodersInterface(config, mode='infer', ckpt_path="/vol/aimspace/users/bubeckn/diffusion-autoencoders/outputs/diffusion/1diffae/train/models/ckpt.0.pt")

    path = '/vol/aimspace/projects/ukbb/data/cardiac/cardiac_segmentations/subjects/1000071/sa_cropped.nii.gz'
    nii = nib.load(path).get_fdata()
    nii1 = rearrange(nii, "h w d t -> d t h w")[5, 0, ...]
    nii1 = torch.from_numpy(nii1).type(torch.float)

    nii_mid = rearrange(nii, "h w d t -> d t h w")[6, 0, ...]
    nii1_mid = torch.from_numpy(nii_mid).type(torch.float)

    nii2 = rearrange(nii, "h w d t -> d t h w")[7, 0, ...]
    nii2 = torch.from_numpy(nii2).type(torch.float)
    
    fig, ax = plt.subplots(1, 3, tight_layout=True)
    ax[0].imshow(nii1, cmap='gray')
    ax[1].imshow(nii_mid, cmap='gray')
    ax[2].imshow(nii2, cmap='gray')
    fig.savefig("gt.png")

    alphas = [0.0, 0.5, 1.0]
    result = diffae.infer_interpolate(nii1.unsqueeze(0), nii2.unsqueeze(0), alphas=alphas)
    print(len(result))

    fig, ax = plt.subplots(1, len(alphas), tight_layout=True)
    for i, r in enumerate(result):
        ax[i].imshow(r['output'], cmap='gray')
        # ax[i].set_title(alphas[i])
        # ax[i].axis('off')

    fig.savefig("figure.png")


if __name__ == '__main__':
    main()