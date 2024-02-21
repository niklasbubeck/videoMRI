import math

import numpy as np
import torch
import torch.nn.functional as F


class TimestepSampler:
    def __init__(self, config, device):
        """
        Args:
            cfg: A dict of config.
        """
        self.num_timesteps = config.num_sample_steps
        self.mode = config.sample
        self.device = device

    def sample(self, size):
        """Sample the time steps to be used in training.

        Args:
            size (int): Number of time steps.

        Returns:
            timesteps (torch.tensor): Sampled time steps.
                shape = (size, )
                dtype = torch.int64
        """
        if self.mode == 'uniform':
            timesteps = self._uniform_sample(size)
        elif self.mode == 'antithetic':
            timesteps = self._antithetic_sample(size)
        else:
            raise NotImplementedError(self.mode)

        return timesteps

    def _uniform_sample(self, size):
        """Uniform sampling of time steps.

        Args:
            size (int): Number of time steps.

        Returns:
            timesteps (torch.tensor): Sampled time steps.
                shape = (size, )
                dtype = torch.int64
        """
        timesteps = torch.randint(low=0, high=self.num_timesteps, size=(size, ), device=self.device)
        return timesteps

    def _antithetic_sample(self, size):
        """Antithetical sampling of time steps.

        Args:
            size (int): Number of time steps.

        Returns:
            timesteps (torch.tensor): Sampled time steps.
                shape = (size, )
                dtype = torch.int64
        """
        timesteps = torch.randint(low=0, high=self.num_timesteps, size=(size // 2 + 1, ), device=self.device)
        timesteps = torch.cat([timesteps, self.num_timesteps - timesteps - 1], dim=0)[:size]
        return timesteps


def get_betas(config):
    """Get the value of beta.

    Args:
        cfg (dict): A dict of config.

    Returns:
        betas (torch.tensor): Beta values for each time step.
            shape = (num_timesteps, )
            dtype = torch.float32
    """
    num_timesteps = config.trainer.timestep_sampler.num_sample_steps
    schedule = config.trainer.beta.schedule

    if schedule == 'linear':
        start = config.trainer.beta.linear.start
        end = config.trainer.beta.linear.end
        betas = np.linspace(start, end, num_timesteps, dtype=np.float64)
    # elif schedule == 'cosine':
        # s = cfg['model']['beta'][schedule]['s']
        # max_beta = cfg['model']['beta'][schedule]['max_beta']

        # def function(t, num_timesteps=num_timesteps, s=s):
        #     numer = (t / num_timesteps + s) * math.pi
        #     denom = (1 + s) * 2
        #     return math.cos(numer / denom) ** 2

        # betas = []
        # for t in range(num_timesteps):
        #     beta_t = min(1 - function(t + 1) / function(t), max_beta)
        #     betas.append(beta_t)
        # betas = np.array(betas, dtype=np.float64)

    betas = torch.from_numpy(betas).float()
    return betas