import matplotlib.pyplot as plt

def show_samples(samples, title="Generated Samples"):
    samples = samples.cpu().detach()
    grid_size = int(len(samples)**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def get_config(device='cuda'):
    from utils.diffusion import marginal_prob_std, diffusion_coeff
    import functools
    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)
    return marginal_prob_std_fn, diffusion_coeff_fn