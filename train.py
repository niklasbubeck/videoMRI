import argparse
from diffae.interface import DiffusionAutoEncodersInterface
from diffae.interface_aekl import AEKLInterface
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    parser.add_argument("--stage", type=int, required=True, help="stage of the cascading scheme to train")
    parser.add_argument("--three_d", action='store_true', help="if to use 3d data or not")
    parser.add_argument("--resume", type=str, default="auto")
    parser.add_argument("--bs", type=int, default="-1")
    parser.add_argument("--lr", type=float, default="-1")
    parser.add_argument("--steps", type=int, default=-1, help="diffusion steps")
    parser.add_argument("--debug", action="store_true", help="if to log with wandb")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, vars(args))

    # Overwrite config values with args
    # exp_name = args.config.split("/")[-1].split(".")[0] # get config file name
    exp_name = config.general.exp_name 
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

    if config.general.network == "AEKL":
        interface = AEKLInterface(config, mode='train')
    else:
        # TODO: instead of giving three d or not, let user choose the network to use
        interface = DiffusionAutoEncodersInterface(config, mode='train')

    interface.train()


if __name__ == '__main__':
    main()
