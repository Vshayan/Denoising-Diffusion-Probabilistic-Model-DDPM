import torch
from tqdm import tqdm

def Sampler(score_model, marginal_prob_std, diffusion_coeff, num_steps, 
            batch_size=64, x_shape=(1, 28, 28), device='cuda', eps=1e-3, y=None):
    
    t = torch.ones(batch_size, device=device)

    # Initialize x as pure Gaussian noise scaled by the maximum standard deviation
    init_x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]

    # Define a sequence of time steps from 1.0 (total noise) down to 0.001 (clean image):
    time_steps = torch.linspace(1., eps, num_steps, device=device)

    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
    return mean_x

def ODE_Sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                num_steps,
                batch_size=64,
                x_shape=(1, 28, 28),
                device='cuda',
                eps=1e-3, y=None):
    """
    A deterministic sampler based on the Probability Flow ODE.
    This version does NOT add noise back in during the steps.
    """
    t = torch.ones(batch_size, device=device)
    # Start with noise
    x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
    
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    with torch.no_grad():
        for time_step in tqdm(time_steps, desc="ODE Sampling"):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            
            # g(t) is the diffusion coefficient
            g = diffusion_coeff(batch_time_step)
            
            # The ODE update rule: 
            # dx = -0.5 * g(t)^2 * score * dt
            # We use 0.5 here because this is the 'Probability Flow' variant 
            # that stays deterministic.
            drift = 0.5 * (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y)
            x = x + drift * step_size
            
    return x

def DDIM_Sampler(score_model,
                 marginal_prob_std,
                 diffusion_coeff,
                 num_steps,
                 batch_size=64,
                 x_shape=(1, 28, 28),
                 device='cuda',
                 eps=1e-3, y=None):
    
    t = torch.ones(batch_size, device=device)
    # Start at t=1 with full noise scale
    x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
    
    # Sequence from 1.0 down to eps
    time_steps = torch.linspace(1., eps, num_steps, device=device)

    with torch.no_grad():
        for i in tqdm(range(len(time_steps) - 1), desc="DDIM Sampling"):
            t_curr = torch.ones(batch_size, device=device) * time_steps[i]
            t_next = torch.ones(batch_size, device=device) * time_steps[i+1]
            
            std_curr = marginal_prob_std(t_curr)[:, None, None, None]
            std_next = marginal_prob_std(t_next)[:, None, None, None]
            
            # 1. Get the score from the model
            score = score_model(x, t_curr, y=y)
            
            # 2. Key realization: in our framework, Noise = -score * std_curr
            # And Predicted x0 = x + (std_curr^2 * score)
            predicted_x0 = x + (std_curr**2) * score
            
            # 3. DDIM Step: Combine the predicted x0 with the current direction
            # This follows the formula: x_next = x0 + (direction_to_xt * std_next)
            # Since our 'noise direction' is -score * std_curr:
            direction_xt = -score * std_curr
            
            x = predicted_x0 + direction_xt * std_next
            
    return x