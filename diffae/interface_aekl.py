import json
import math
import time
import os 
import shutil
from pathlib import Path
from collections import OrderedDict
import torch.optim as optim
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

import wandb
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from .classifier import ClassifierTrainer, LinearClassifier, evaluate_classifier
from .dataset import UKBB, get_torchvision_transforms
from .models.model import DiffusionAutoEncoders, DiffusionAutoEncoders3D, CascadedDiffusionModel
from .sampler import Sampler
from .trainer import Trainer
from .utils import get_torchvision_unnormalize
from diffae.interface import DiffusionAutoEncodersInterface
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from diffae.trainer_aekl import train_aekl

class AEKLInterface(DiffusionAutoEncodersInterface):
    def __init__(self, config, mode, ckpt_path=None, split_batches=True):
        """Setting up config, output directory, model, and dataset.

        Args:
            args (dict): A dict of arguments with the following keys,
                mode == 'train':
                    data_name (str): Dataset name.
                    size (int): Image size.
                    expn (str): Experiment name.
                mode in {'test', 'clf_train', 'clf_test', 'infer'}:
                    output (str): Path to output directory.
        """
        # Check for correct modes
        assert mode in {'train', 'test', 'clf_train', 'clf_test', 'infer'}
        
        # Init arguments
        self.mode = mode
        self.config = self._init_config(config)
        self.epoch = config.general.training_epochs
        self.eval_freq = config.general.eval_freq
        self.adv_weight = config.stage1.adv_weight
        self.kl_weight = config.stage1.kl_weight
        self.perceptual_weight = config.stage1.perceptual_weight
        self.adv_start = config.stage1.adv_start
        self.device = torch.device("cuda")
        # Setup logging and output dirs
        self.output_dir = self._init_output_dir()
        self.config.output_dir = self.output_dir

        self.model, self.train_days, self.start_time, self.steps = self._init_model(ckpt_path=ckpt_path)

        # clf_ckpt_path = self.output_dir / 'ckpt' / args['clf_ckpt'] if 'clf_ckpt' in args else None
        # self.clf_model = self._init_clf_model(clf_ckpt_path)

        if self.mode in {'test', 'infer'}:
            self.sampler = Sampler(self.model, self.config, device='cuda')
            # self.unnormalize = get_torchvision_unnormalize(
            #     self.cfg['test']['dataset'][-1]['params']['mean'],
            #     self.cfg['test']['dataset'][-1]['params']['std'],
            # )

        self._init_dataset()

        self.discriminator = PatchDiscriminator(**config.discriminator.params).cuda()
        self.perceptual_loss = PerceptualLoss(**config.perceptual_network.params).cuda()

            # Optimizers
        self.optimizer_g = optim.Adam(self.model.parameters(), lr=config.stage1.base_lr)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.stage1.disc_lr)



    def _init_config(self, args):
        """
        Making last adaptions for setting up config dict if necessary.
        """
        return args

    @staticmethod
    def _delete_folder(path):
        print(f"deleting folders: {path}")
        try:
            shutil.rmtree(path)
        except:
            pass

    @staticmethod
    def _create_save_folder(path, config):
        os.makedirs(path, exist_ok = True)
        OmegaConf.save(config, os.path.join(path, "model_config.yaml"))
        os.makedirs(os.path.join(path, "videos"), exist_ok = True)
        os.makedirs(os.path.join(path, "models"), exist_ok = True)

    def _init_output_dir(self):
        output_dir = Path(os.path.join(self.config.checkpoint.path, self.config.checkpoint.exp_name, self.mode))

        # Delete previous experiment if requested
        if self.config.resume == 'overwrite':
            self._delete_folder(os.path.join(output_dir, "models"))

        # create save folder and save the config
        self._create_save_folder(output_dir, self.config)

        return output_dir

    def _merge_model(self, checkpoints=None, only_stage=[0]):
        if checkpoints is None: 
            checkpoints = []
            basepath = (os.path.join(os.path.dirname(self.output_dir), "train", "models"))
            cascaded_state_dict = torch.load(os.path.join(basepath, "cascaded_model.pt"))
            stagedirs = sorted(os.listdir(basepath))
            for stagedir in stagedirs:
                try:
                    # get latest model 
                    stageckpts = sorted(os.listdir(os.path.join(basepath, stagedir)))
                    checkpoints.append(os.path.join(basepath, stagedir, stageckpts[-1]))
                except NotADirectoryError:
                    pass

        loaded_ckpts = []
        for ckpt in checkpoints:
            loaded_ckpts.append(torch.load(f"{ckpt}"))
            print(f"Loaded {ckpt} successfully")
        
        for stage, ckpt in enumerate(loaded_ckpts):
            if stage in only_stage:
                unet_parameters = {key: value for key, value in ckpt['model'].items()}
                for key, value in unet_parameters.items():
                    full_key = f'unets.{stage}.{key}'  # Adjust this key based on the structure of your larger model
                    cascaded_state_dict["model"][full_key] = value

        # optimizers 
        for stage ,ckpt in enumerate(loaded_ckpts):
            cascaded_state_dict[f'optim{stage}'] = ckpt['optimizer']

        # steps
        cascaded_state_dict['steps'] = {}
        for stage, ckpt in enumerate(loaded_ckpts):
            cascaded_state_dict['steps'][stage] = ckpt['steps']
        
        # time_elapsed
        cascaded_state_dict['time_elapsed'] = sum([ ckpt['time_elapsed'] for ckpt in loaded_ckpts ])


        return cascaded_state_dict    


    def _init_model(self, ckpt_path=None):
        """
        Setting up DiffusionAutoEncoders module and handling the resume 
        """
        print('Initializing Diffusion Autoencoders...')
        # if self.three_d:
            # build the unets
        

        print("three_d", self.config.three_d)
        if self.config.three_d:
            model = AutoencoderKL(**self.config.stage1.params).cuda()
        else:
            raise NotImplementedError("Only 3D models are supported at the moment")

        # if specifically known as for inference
        if ckpt_path is not None: 
            state = torch.load(ckpt_path)
            model.load_state_dict(state['model'])
            train_days = int(ckpt_path.split(".")[1])+1 # format is ckpt.X.pt
            start_time = time.time() - state['time_elapsed']
            return model, train_days, start_time, state["steps"]

        if self.config.resume == 'auto' and len(os.listdir(os.path.join(self.output_dir, "models"))) > 0:
            checkpoints = sorted(os.listdir(os.path.join(self.output_dir, "models")))
            print(checkpoints)
            train_days = int(checkpoints[-1].split(".")[1])+1 # format is ckpt.X.pt
            weight_path = os.path.join(self.output_dir, "models", checkpoints[-1])
            print(f'Loading Diff-AE checkpoint from: {weight_path}')
            try:
                state = torch.load(weight_path)
                model.load_state_dict(state['model'])
            except: # remove "module." from keys (DDP)
                new_state = {}
                new_state['model'] = {}
                for k, v in state['model'].items():
                    name = k[7:] # remove `module.`
                    new_state['model'][name] = v
                model.load_state_dict(new_state['model'])
            start_time = time.time() - state['time_elapsed']
            steps = state['steps']
        else:
            train_days = 0
            steps = 0
            start_time = time.time()
            print('Training from scratch')
        return model, train_days, start_time, steps


    def _init_clf_model(self, ckpt_path=None):
        """Setting up LinearClassifier module.
        """
        print('Initializing classifier...')
        clf_model = LinearClassifier(self.cfg)
        if ckpt_path is not None:
            print('Loading classifier checkpoint...')
            state = torch.load(ckpt_path)
            clf_model.load_state_dict(state['model'])
        return clf_model

    def _init_dataset(self):
        """Setting up transforms and dataset.
        """
        assert self.mode in {'train', 'test', 'clf_train', 'clf_test', 'infer'}
        print('Initializing dataset...')

        if self.mode in {'train', 'clf_train'}:
            self.transforms = get_torchvision_transforms(self.config, 'train')
        else:
            self.transforms = get_torchvision_transforms(self.config, 'test')

        # TODO: outsource sbj file
        ds = UKBB(self.config, sbj_file=self.config.dataset.sbj_file, transforms=self.transforms)
        idxs = list(range(0, len(ds)))
        split_idx = int(len(idxs) * self.config.dataset.train_pct)

        self.train_dataset = Subset(ds, idxs[:split_idx])
        self.test_dataset = Subset(ds, idxs[split_idx:])

        # elif self.mode == 'clf_train':
        #     image_dataset = get_dataset(name=data_name, split='train', transform=self.transforms)
        #     image_loader = DataLoader(image_dataset, **self.cfg['classifier']['train']['dataloader'])
        #     self.dataset = EmbeddingDataset(image_loader, self.model.encoder, self.cfg)
        # elif self.mode == 'clf_test':
        #     image_dataset = get_dataset(name=data_name, split='test', transform=self.transforms)
        #     image_loader = DataLoader(image_dataset, **self.cfg['classifier']['test']['dataloader'])
        #     self.dataset = EmbeddingDataset(image_loader, self.model.encoder, self.cfg)

    def train(self):
        """DiffusionAutoEncoders training.
        """
        assert self.mode == 'train'
        add_data = {
            "train_days": self.train_days,
            "start_time": self.start_time,
            "steps": self.steps
        }
        wandb.init(
            name=f"{self.config.checkpoint.exp_name}",
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            config=OmegaConf.to_container(self.config, resolve=True) # type: ignore
        )

        # Train model
        print(f"Starting Training")
        val_loss = train_aekl(
            model=self.model,
            discriminator=self.discriminator,
            perceptual_loss=self.perceptual_loss,
            start_epoch=0,
            best_loss=float("inf"),
            train_loader=self.train_dataset,
            val_loader=self.test_dataset,
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            n_epochs=self.epoch,
            eval_freq=self.eval_freq,
            # writer_train=writer_train,
            # writer_val=writer_val,
            device=self.device,
            run_dir=self.output_dir,
            kl_weight=self.kl_weight,
            adv_weight=self.adv_weight,
            perceptual_weight=self.perceptual_weight,
            adv_start=self.adv_start,
        )

        # trainer = Trainer(self.model, self.config, self.output_dir, self.train_dataset, self.test_dataset, **add_data)
        # trainer.train(self.config.stage)



    def clf_train(self):
        """LinearClassifier training.
        """
        self.clf_model.set_norm_params(self.dataset)
        self.clf_trainer = ClassifierTrainer(self.clf_model, self.cfg, self.output_dir, self.dataset)
        self.clf_trainer.train()

    @torch.inference_mode()
    def clf_test(self):
        """LinearClassifier evaluation.
        """
        print('Classifier evaluation start...')
        result = evaluate_classifier(self.clf_model, self.cfg, self.dataset)
        with (self.output_dir / 'clf_test.json').open('w') as fp:
            json.dump(result, fp, indent=4, sort_keys=False)

    @torch.inference_mode()
    def infer(self, image, xt=None, style_emb=None):
        """Autoencode a single image.

        Args:
            image: (PIL Image): A single PIL Image.
            style_emb (torch.tensor, optional): A tensor of SemanticEncoder embedding.
                You can perform conditional generation with arbitary embedding by passing this argument.

        Returns:
            result (dict): A result of autoencoding which has the following keys,
                input (numpy.ndarray): A input image array.
                output (numpy.ndarray): A output (autoencoded) image array.
                x0_preds (List[numpy.ndarray]): A list of predicted x0 per timestep.
                xt_preds (List[numpy.ndarray]): A list of predicted xt per timestep.
        """
        assert self.mode == 'infer'
        if not torch.is_tensor(image):
            image = self.transforms(image)
        result = self.sampler.sample_one_image(image, xt=xt, style_emb=style_emb)

        # Unnormalize and to numpy.ndarray
        for k, v in result.items():
            if isinstance(v, list):
                for i, x in enumerate(result[k]):
                    assert torch.is_tensor(x)
                    result[k][i] = x.permute(1, 2, 0).cpu().detach().numpy()
            elif torch.is_tensor(v):
                result[k] = v.permute(1, 2, 0).cpu().detach().numpy()

        return result

    @torch.inference_mode()
    def infer_manipulate(self, image, target_id, s=0.3):
        """Attribute manipulation using classifier.

        Args:
            image (PIL Image): A single PIL Image.
            target_id (int): Target attribute id.
            s (float or List[float]): Attribute manipulation parameter(s).

        Returns:
            result (dict): A result of autoencoding which has the following keys,
                input (numpy.ndarray): A input image array.
                output (numpy.ndarray): A output (autoencoded) image array.
                x0_preds (List[numpy.ndarray]): A list of predicted x0 per timestep.
                xt_preds (List[numpy.ndarray]): A list of predicted xt per timestep.
        """
        assert self.mode == 'infer'
        device = self.cfg['general']['device']
        if isinstance(s, (int, float)):
            s = [s]

        clf_ckpt_path = self.output_dir / 'ckpt/clf_last_ckpt.pth'
        self.clf_model = self._init_clf_model(clf_ckpt_path)
        self.clf_model.to(device)
        direction = torch.nn.functional.normalize(self.clf_model.classifier.weight[target_id][None, :], dim=1)

        x0 = self.transforms(image).unsqueeze(dim=0).to(device)
        xt = self.sampler.encode_stochastic(x0)
        style_emb = self.model.encoder(x0)
        style_emb = self.clf_model.normalize(style_emb)

        results = []
        for i in range(len(s)):
            transferred_style_emb = style_emb + s[i] * math.sqrt(style_emb.shape[1]) * direction
            transferred_style_emb = self.clf_model.unnormalize(transferred_style_emb)

            result = self.infer(image, xt=xt, style_emb=transferred_style_emb)
            results.append(result)

        return results

    @torch.inference_mode()
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
        assert self.mode == 'infer'
        device = 'cuda'
        if isinstance(alphas, (int, float)):
            alphas = [alphas]

        if not torch.is_tensor(image_1):
            image_1 = self.transforms(image_1)
        if not torch.is_tensor(image_2):
            image_2 = self.transforms(image_2)

        x0_1 = image_1.unsqueeze(dim=0).to(device)
        x0_2 = image_2.unsqueeze(dim=0).to(device)

        xt_1 = self.sampler.encode_stochastic(x0_1)
        xt_2 = self.sampler.encode_stochastic(x0_2)

        style_emb_1 = self.model.encoder(x0_1)
        style_emb_2 = self.model.encoder(x0_2)

        results = []
        for alpha in alphas:
            result = self.sampler.interpolate(xt_1, xt_2, style_emb_1, style_emb_2, alpha)

            # Unnormalize and to numpy.ndarray
            for k, v in result.items():
                if isinstance(v, list):
                    for i, x in enumerate(result[k]):
                        assert torch.is_tensor(x)
                        result[k][i] = x.permute(1, 2, 0).cpu().detach().numpy()
                elif torch.is_tensor(v):
                    result[k] = x.permute(1, 2, 0).cpu().detach().numpy()

            results.append(result)

        return results
