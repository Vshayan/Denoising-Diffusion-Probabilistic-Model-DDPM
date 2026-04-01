import torch
import numpy as np

def diffusion_coeff(t, sigma, device='cuda'):
    return torch.tensor(sigma**t, device=device)

def marginal_prob_std(t, sigma, device='cuda'):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=device)
    else:
        t = t.to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
