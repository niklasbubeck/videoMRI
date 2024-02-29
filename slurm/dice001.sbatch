#!/bin/bash 
#SBATCH -J dice10
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --mem=40GB
#SBATCH --gres=gpu1
#SBATCH --time=72:00:00

source ~/.bashrc 
source ~/miniconda3/bin/activate videomri

cd ~/Projects/videoMRI

srun python train.py --config=./configs/1recon_mse_seg_dice_001.yaml --bs=2 --stage=0 --three_d --resume=overwrite
