import torch
import argparse
from models.unet import UNet_Tranformer
from train import loss_fn_cond
from sample import Sampler, ODE_Sampler, DDIM_Sampler
from utils.helpers import show_samples, get_config
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'])
    parser.add_argument('--sampler', type=str, default='sde', choices=['sde', 'ode', 'ddim'], 
                        help='Choose sde (Langevin) or ode (Probability Flow)')
    parser.add_argument('--digit', type=int, default=3, help='Digit to generate')
    parser.add_argument('--steps', type=int, default=500, help='Number of denoising steps')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    marginal_fn, diff_fn = get_config(device)
    model = UNet_Tranformer(marginal_prob_std=marginal_fn).to(device)

    if args.mode == 'train':
        dataset = datasets.MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = Adam(model.parameters(), lr=1e-4)
        
        for epoch in range(10): # Adjust epochs as needed
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn_cond(model, x, y, marginal_fn)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"Epoch {epoch} Batch {i} Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), 'ckpt.pth')

    elif args.mode == 'sample':
        print(f"Loading checkpoint and sampling with {args.sampler.upper()}...")
        model.load_state_dict(torch.load('ckpt.pth', map_location=device))
        model.eval()
        
        # Create a batch of the same digit for testing
        y = torch.ones(64, dtype=torch.long, device=device) * args.digit
        
        if args.sampler == 'sde':
            # Original Sampler (Predictor-Corrector / Langevin)
            samples = Sampler(model, marginal_fn, diff_fn, num_steps=args.steps, y=y, device=device)
        elif args.sampler == 'ode':
            samples = ODE_Sampler(model, marginal_fn, diff_fn, num_steps=args.steps, y=y, device=device)

        else:
            samples = DDIM_Sampler(model, marginal_fn, diff_fn, num_steps=args.steps, y=y, device=device)
        
        show_samples(samples, title=f"Generated Digit: {args.digit} ({args.sampler.upper()})")

if __name__ == "__main__":
    main()