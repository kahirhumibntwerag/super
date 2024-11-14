import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from inspect import isfunction
from schedules import extract_into_tensor, exists, default, LinearSchedule, BetaSchedule

to_torch = partial(torch.tensor, dtype=torch.float32)

class Diffusion(nn.Module):
    def __init__(self,
                schedule: BetaSchedule,
                parameterization: str ='eps',
                v_posterior: float = 0.0
                ):

        super().__init__()
        self.parameterization = parameterization
        self.v_posterior = v_posterior
        self.schedule = schedule
        self.register_schedule()

    def register_schedule(self):

        betas =  self.schedule.betas().numpy()
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod.shape[0] == self.schedule.timesteps

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas

        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")

        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def add_noise(self, x0, t):
      noise = torch.randn_like(x0)

      sqrt_alphas_cumprod = extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape)
      sqrt_one_minus_alphas_cumprod = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

      xt = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
      return xt, noise

