import sys
import math
import torch as th
import torch.nn as nn
import numpy as np



def get_schedule():
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)


    betas = th.from_numpy(betas).float()
    alphas = 1.0 - betas
    alphas_cump = alphas.cumprod(dim=0)

    return betas, alphas_cump


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class Schedule(object):
    def __init__(self):
        device = th.device('cuda')
        betas, alphas_cump = get_schedule()

        self.betas, self.alphas_cump = betas.to(device), alphas_cump.to(device)
        self.alphas_cump_pre = th.cat([th.ones(1).to(device), self.alphas_cump[:-1]], dim=0)
        self.total_step = 1000

        self.method = choose_method('F-PNDM')  # add pflow
        self.ets = None

    def diffusion(self, img, t_end, t_start=0, noise=None):
        if noise is None:
            noise = th.randn_like(img)
        alpha = self.alphas_cump.index_select(0, t_end).view(-1, 1, 1, 1)
        img_n = img * alpha.sqrt() + noise * (1 - alpha).sqrt()

        return img_n, noise

    def denoising(self, lr, img_n, t_end, t_start, model, first_step=False, pflow=False):
      if first_step:
        self.ets = []
      img_next = self.method(lr, img_n, t_start, t_end, model, self.alphas_cump, self.ets)

      return img_next

#
#
#
#
#
#

import sys
import copy
import torch as th
import torch

def choose_method(name):
    if name == 'DDIM':
        return gen_order_1
    elif name == 'S-PNDM':
        return gen_order_2
    elif name == 'F-PNDM':
        return gen_order_4
    elif name == 'FON':
        return gen_fon
    elif name == 'PF':
        return gen_pflow
    else:
        return None


def gen_pflow(img, t, t_next, model, betas, total_step):
    n = img.shape[0]
    beta_0, beta_1 = betas[0], betas[-1]

    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # drift, diffusion -> f(x,t), g(t)
    drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    score = - model(img, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
    drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

    return drift


def gen_fon(img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t + t_next) / 2.0, t_next]
    if len(ets) > 2:
        noise = model(img, t)
        img_next = transfer(img, t, t-1, noise, alphas_cump)
        delta1 = img_next - img
        ets.append(delta1)
        delta = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = model(img, t_list[0])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_1 = img_ - img
        ets.append(delta_1)

        img_2 = img + delta_1 * (t - t_next).view(-1, 1, 1, 1) / 2.0
        noise = model(img_2, t_list[1])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_2 = img_ - img

        img_3 = img + delta_2 * (t - t_next).view(-1, 1, 1, 1) / 2.0
        noise = model(img_3, t_list[1])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_3 = img_ - img

        img_4 = img + delta_3 * (t - t_next).view(-1, 1, 1, 1)
        noise = model(img_4, t_list[2])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_4 = img_ - img
        delta = (1 / 6.0) * (delta_1 + 2*delta_2 + 2*delta_3 + delta_4)

    img_next = img + delta * (t - t_next).view(-1, 1, 1, 1)
    return img_next


def gen_order_4(lr, img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t+t_next)/2, t_next]
    if len(ets) > 2:
        noise_ = model(torch.cat([img, lr], dim=1), t)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = runge_kutta(lr, img, t_list, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def runge_kutta(lr, x, t_list, model, alphas_cump, ets):
    e_1 = model(torch.cat([x, lr], dim=1), t_list[0])
    ets.append(e_1)
    x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

    e_2 = model(torch.cat([x_2, lr], dim=1), t_list[1])
    x_3 = transfer(x, t_list[0], t_list[1], e_2, alphas_cump)

    e_3 = model(torch.cat([x_3, lr], dim=1), t_list[1])
    x_4 = transfer(x, t_list[0], t_list[2], e_3, alphas_cump)

    e_4 = model(torch.cat([x_4, lr], dim=1), t_list[2])
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et


def gen_order_2(img, t, t_next, model, alphas_cump, ets):
    if len(ets) > 0:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = 0.5 * (3 * ets[-1] - ets[-2])
    else:
        noise = improved_eular(img, t, t_next, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def improved_eular(x, t, t_next, model, alphas_cump, ets):
    e_1 = model(x, t)
    ets.append(e_1)
    x_2 = transfer(x, t, t_next, e_1, alphas_cump)

    e_2 = model(x_2, t_next)
    et = (e_1 + e_2) / 2
    # x_next = transfer(x, t, t_next, et, alphas_cump)

    return et


def gen_order_1(img, t, t_next, model, alphas_cump, ets):
    noise = model(img, t)
    ets.append(noise)
    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def transfer(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

    x_next = x + x_delta
    return x_next


def transfer_dev(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long()+1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long()+1].view(-1, 1, 1, 1)

    x_start = th.sqrt(1.0 / at) * x - th.sqrt(1.0 / at - 1) * et
    x_start = x_start.clamp(-1.0, 1.0)

    x_next = x_start * th.sqrt(at_next) + th.sqrt(1 - at_next) * et

    return x_next



#
#
#
#
#
#
class Runner(object):
    def __init__(self, schedule, sample_speed, device):
        self.diffusion_step = 1000
        self.sample_speed = sample_speed
        self.device = device

        self.schedule = schedule


    def sample_image(self, lr, noise, seq, model, pflow=False):
        with th.no_grad():
          imgs = [noise]
          seq_next = [-1] + list(seq[:-1])

          start = True
          n = noise.shape[0]

          for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (th.ones(n) * i).to(self.device)
            t_next = (th.ones(n) * j).to(self.device)

            img_t = imgs[-1].to(self.device)
            img_next = self.schedule.denoising(lr, img_t, t_next, t, model, start, pflow)
            start = False

            imgs.append(img_next.to(self.device))

          img = imgs[-1]

          return img