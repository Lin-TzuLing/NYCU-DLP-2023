import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# self-defined import
from utils.schedules import ddpm_schedules


class DDPM(nn.Module):
    def __init__(self, unet_model, betas, noise_steps, device):
        super(DDPM, self).__init__()

        self.n_T = noise_steps
        self.device = device
        self.unet_model = unet_model

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], noise_steps).items():
            self.register_buffer(k, v)

        # loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, x, cond):
        """training ddpm, sample time and noise randomly (return loss)"""
        # t ~ Uniform(0, n_T)
        timestep = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  
        # eps ~ N(0, 1)
        noise = torch.randn_like(x)  

        x_t = (
            self.sqrtab[timestep, None, None, None] * x
            + self.sqrtmab[timestep, None, None, None] * noise
        ) 

        predict_noise = self.unet_model(x_t, cond, timestep/self.n_T)

        # return MSE loss between real added noise and predicted noise
        loss = self.mse_loss(noise, predict_noise)
        return loss

    def sample(self, cond, size, device):
        """sample initial noise and generate images based on conditions"""
        n_sample = len(cond)
        # x_T ~ N(0, 1), sample initial noise
        x_i = torch.randn(n_sample, *size).to(device)  
        for idx in tqdm(range(self.n_T, 0, -1), leave=False):
            timestep = torch.tensor([idx / self.n_T]).to(device)
            z = torch.randn(n_sample, *size).to(device) if idx > 1 else 0
            eps = self.unet_model(x_i, cond, timestep)
            x_i = (
                self.oneover_sqrta[idx] * (x_i - eps * self.mab_over_sqrtmab[idx])
                + self.sqrt_beta_t[idx] * z
            )
        return x_i