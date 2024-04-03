import glob
import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision.utils as tvu
import tqdm


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """

    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        # #在model中，我们需要更新其中的参数，训练结束将参数保存下来。
        # 但在某些时候，我们可能希望模型中的某些参数不更新（从开始到结束均保持不变），但又希望参数保存下来（model.state_dict() ），这时就会用到 register_buffer() 。
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, flow):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t, flow), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                extract(self.coeff1, t, x_t.shape) * x_t -
                extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, flow):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t, flow)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T, flow, last=True):
        """
        Algorithm 2.
        """
        skip = self.T // 1  # self.args.timesteps
        seq = range(0, self.T, skip)
        x = self.ddpm_steps(x_T, seq, self.betas, flow)
        if last:
            # x = x[0][-1]
            x = x[0][-1]
            return x

    def ddpm_steps(self, x, seq, b, flow, **kwargs):
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [x]
            x0_preds = []
            self.betas = b
            for i, j in zip(seq, seq_next):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(self.betas, t.long())
                atm1 = compute_alpha(self.betas, next_t.long())
                beta_t = 1 - at / atm1
                x = xs[-1].to('cuda')
                output = self.model(x.float(), t.long(), flow)
                e = output

                next_original_sample = (x - beta_t ** 0.5 * e ) / at ** 0.5
                next_sample_direction = (1 - atm1) ** 0.5 * e
                next_sample = atm1 ** 0.5 * next_original_sample + next_sample_direction

                xs.append(next_sample.to('cuda'))
            # for i, j in zip(reversed(seq), reversed(seq_next)):
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(self.betas, t.long())
                atm1 = compute_alpha(self.betas, next_t.long())
                beta_t = 1 - at / atm1
                x = xs[-1].to('cuda')
                    # t = t.long()
                output = self.model(x.float(), t.long(), flow)
                e = output

                x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
                x0_from_e = torch.clamp(x0_from_e, -1, 1)
                x0_preds.append(x0_from_e.to('cuda'))
                mean_eps = (
                                       (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
                               ) / (1.0 - at)

                mean = mean_eps
                noise = torch.randn_like(x)
                # noise = 0
                mask = 1 - (t == 0).float()
                mask = mask.view(-1, 1, 1, 1)
                logvar = beta_t.log()
                sample = mean + mask * torch.exp(0.5 * logvar) * noise
                xs.append(sample.to('cuda'))

            return xs, x0_preds


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
