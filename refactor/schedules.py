import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from inspect import isfunction

to_torch = partial(torch.tensor, dtype=torch.float32)

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class BetaSchedule(ABC):
    @abstractmethod
    def betas(self):
        pass

class LinearSchedule(BetaSchedule):
    def __init__(self, timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        self.timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end

    def betas(self):
        return (
            torch.linspace(self.linear_start ** 0.5, self.linear_end ** 0.5, self.timesteps, dtype=torch.float64) ** 2
        )

class CosineSchedule(BetaSchedule):
    def __init__(self, timesteps=1000, cosine_s=8e-3):
        self.timesteps = timesteps
        self.cosine_s = cosine_s

    def betas(self):
        timesteps = (
            torch.arange(self.timesteps + 1, dtype=torch.float64) / self.timesteps + self.cosine_s
        )
        alphas = timesteps / (1 + self.cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        return torch.clamp(betas, min=0, max=0.999)

class SqrtLinearSchedule(BetaSchedule):
    def __init__(self, timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        self.timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end

    def betas(self):
        return torch.linspace(self.linear_start, self.linear_end, self.timesteps, dtype=torch.float64)

class SqrtSchedule(BetaSchedule):
    def __init__(self, timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        self.timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end

    def betas(self):
        return torch.linspace(self.linear_start, self.linear_end, self.timesteps, dtype=torch.float64) ** 0.5

class KinearSchedule(BetaSchedule):
    def __init__(self, timesteps=1000, linear_start=1e-4, linear_end=2e-2):
        self.timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end

    def betas(self):
        return torch.linspace(self.linear_start, self.linear_end, self.timesteps, dtype=torch.float64) ** 1.5

class SqrtCosineSchedule(BetaSchedule):
    def __init__(self, timesteps=1000, cosine_s=8e-3):
        self.timesteps = timesteps
        self.cosine_s = cosine_s

    def betas(self):
        timesteps = (
            torch.arange(self.timesteps + 1, dtype=torch.float64) / self.timesteps + self.cosine_s
        )
        alphas = timesteps / (1 + self.cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = torch.sqrt(1 - alphas[1:] / alphas[:-1])
        return torch.clamp(betas, min=0, max=0.999)
