from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import Meter, seed_everything, training_reproducibility_cudnn


@torch.inference_mode()
def evaluate_classifier(clf_model, cfg, test_dataset):
    """Evaluation of multi-label classifications

    Args:
        clf_model (torch.nn.Module): Multi-label classifier module.
        cfg (dict): A dict of config.
        test_dataset (torch.utils.data.Dataset): Test dataset class.

    Returns:
        result (dict): A dict of accuracy and AUROC values per label.
    """
    test_loader = DataLoader(test_dataset, **cfg['classifier']['test']['dataloader'])

    device = cfg['general']['device']
    clf_model.to(device)
    clf_model.eval()

    gt = []
    score = []
    for batch in tqdm(test_loader):
        inputs, targets = batch
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = clf_model(inputs).sigmoid()
        gt.append(targets.to('cpu').numpy())
        score.append(outputs.to('cpu').numpy())
    gt = np.concatenate(gt)
    score = np.concatenate(score)

    accuracy_list = []
    auroc_list = []
    result = {}
    for i in range(cfg['classifier']['num_class']):
        accuracy = accuracy_score(gt[:, i], np.where(score[:, i] < 0.5, 0, 1))
        auroc = roc_auc_score(gt[:, i], score[:, i])
        accuracy_list.append(accuracy)
        auroc_list.append(auroc)
        result[i] = {
            'accuracy': accuracy,
            'auroc': auroc,
        }
    result['macro_average'] = {
        'accuracy': sum(accuracy_list) / cfg['classifier']['num_class'],
        'auroc': sum(auroc_list) / cfg['classifier']['num_class'],
    }

    return result


class LinearClassifier(nn.Module):
    """Linear classifier for multi-label classification.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        in_chans = self.cfg['model']['network']['encoder']['emb_channels']
        out_chans = self.cfg['classifier']['num_class']
        self.classifier = nn.Linear(in_chans, out_chans)

        self.device = self.cfg['general']['device']

        self.register_buffer('std', torch.ones(self.cfg['model']['network']['encoder']['emb_channels']))
        self.register_buffer('mean', torch.zeros(self.cfg['model']['network']['encoder']['emb_channels']))

    def set_norm_params(self, emb_dataset):
        """Calculate normalization parameters
        """
        embs = []
        for i in range(len(emb_dataset)):
            emb, _ = emb_dataset[i]
            embs.append(emb)
        embs = torch.stack(embs)

        std, mean = torch.std_mean(embs, dim=0, unbiased=False)
        std = std.to(self.device)
        mean = mean.to(self.device)

        self.register_buffer('std', std)
        self.register_buffer('mean', mean)

    def normalize(self, emb):
        emb -= self.mean[None, :]
        emb /= self.std[None, :]
        return emb

    def unnormalize(self, emb):
        emb *= self.std[None, :]
        emb += self.mean[None, :]
        return emb

    def forward(self, emb, is_normalized=False):
        if not is_normalized:
            emb = self.normalize(emb)

        out = self.classifier(emb)
        return out


class ClassifierTrainer:
    def __init__(self, model, cfg, output_dir, train_dataset):
        self.model = model
        self.cfg = cfg
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, **self.cfg['classifier']['train']['dataloader'])

        self.device = self.cfg['general']['device']
        self.model.to(self.device)

        self.ckpt_dir = Path(self.output_dir / 'ckpt')
        self.ckpt_dir.mkdir(exist_ok=True)
        print(f'Checkpoints are saved in {self.ckpt_dir}')

        seed_everything(cfg['general']['seed'])
        training_reproducibility_cudnn()

        self.log_interval = cfg['classifier']['train']['log_interval']
        self.save_interval = cfg['classifier']['train']['save_interval']
        print(f'Output a log for every {self.log_interval} iteration')
        print(f'Save checkpoint every {self.save_interval} epoch')

        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

        self.fp16 = cfg['classifier']['train']['fp16']
        self.grad_accum_steps = cfg['classifier']['train']['grad_accum_steps']

        self.clip_grad_norm = cfg['classifier']['train']['clip_grad_norm']

    def get_criterion(self):
        criterion_cfg = self.cfg['classifier']['train']['loss']
        criterion_cls = getattr(torch.nn, criterion_cfg['name'])
        criterion = criterion_cls()
        print(f'Use {criterion_cfg["name"]} as criterion')
        return criterion

    def get_optimizer(self):
        optimizer_cfg = self.cfg['classifier']['train']['optimizer']
        optimizer_cls = getattr(torch.optim, optimizer_cfg['name'])
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_cfg['params'])
        print(f'Use {optimizer_cfg["name"]} optimizer')
        return optimizer

    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.cfg['classifier']['train']['epoch']):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter, batch in enumerate(self.train_loader):
            self.before_iter()
            self.train_one_iter(batch)
            self.after_iter()

    def train_one_iter(self, batch):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs, is_normalized=False)
            loss = self.criterion(outputs, targets)
            loss /= self.grad_accum_steps

        self.scaler.scale(loss).backward()
        self.train_loss_meter.update(loss.item())

        if (self.iter + 1) % self.grad_accum_steps == 0:
            if self.clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    def before_train(self):
        self.train_loss_meter = Meter()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        print('Training start ...')

    def after_train(self):
        print('Training done')

    def before_epoch(self):
        self.model.train()
        self.epoch_start_time = time()
        print(f'---> Start train epoch {self.epoch + 1}')

    def after_epoch(self):
        self.save_ckpt('clf_last_ckpt.pth')
        epoch_elapsed_time = time() - self.epoch_start_time
        print(f'Epoch {self.epoch + 1} done. ({epoch_elapsed_time:.1f} sec)')

        if (self.epoch + 1) % self.save_interval == 0:
            self.save_ckpt(name=f'clf_epoch_{self.epoch + 1}_ckpt.pth')

    def before_iter(self):
        pass

    def after_iter(self):
        if (self.iter + 1) % self.log_interval == 0:
            print(
                'epoch: {}/{}, iter: {}/{}, loss{:.3f}'.format(
                    self.epoch + 1, self.cfg['classifier']['train']['epoch'],
                    self.iter + 1, len(self.train_loader),
                    self.train_loss_meter.latest,
                )
            )
            self.tblogger.add_scalar('clf_train_loss', self.train_loss_meter.latest, self.iter + 1)
            self.train_loss_meter.reset()

    def save_ckpt(self, name):
        print(f'Saving checkpoint to {self.ckpt_dir / name}')
        state = {
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, self.ckpt_dir / name)
