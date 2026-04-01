import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.unet import UNet_Tranformer
from utils.diffusion import marginal_prob_std
import functools

def loss_fn_cond(model, x, y, marginal_prob_std_func, eps=1e-5):
    # 1. Sample a random time step 't' between 0 and 1
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

    # 2. Generate pure Gaussian noise (z)
    z = torch.randn_like(x)

    # 3. FIX: Use the argument 'marginal_prob_std_func' here!
    # This is the 'partial' version that already has sigma inside it.
    std = marginal_prob_std_func(random_t)

    # 4. Perturb the image
    perturbed_x = x + z * std[:, None, None, None]
    
    score = model(perturbed_x, random_t, y=y)
    loss = F.mse_loss(score * std[:, None, None, None], -z, reduction='mean')
    return loss

# --- Everything below this is only for direct training ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sigma = 25.0
    
    # This is where we "bake" sigma into the function
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)

    model = UNet_Tranformer(marginal_prob_std=marginal_prob_std_fn).to(device)
    dataset = datasets.MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # Pass the WRAPPED function here
            loss = loss_fn_cond(model, x, y, marginal_prob_std_fn)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")
        torch.save(model.state_dict(), 'ckpt.pth')