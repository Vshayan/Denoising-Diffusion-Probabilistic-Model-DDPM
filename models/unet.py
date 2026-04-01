import torch.nn as nn
from .embeddings import GaussianFourierProjection, Dense
from .attention import SpatialCrossAttention

class UNet_Tranformer(nn.Module):
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, text_dim=256, nClass=10):
        super().__init__()
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cond_embed = nn.Embedding(nClass, text_dim)
        self.act = nn.SiLU()
        self.marginal_prob_std = marginal_prob_std
        
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialCrossAttention(channels[2], text_dim)

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.attn4 = SpatialCrossAttention(channels[3], text_dim)

        # Decoding layers
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1, bias=False) # Removed output_padding
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1, padding=1)

    def forward(self, x, t, y=None):
        embed = self.act(self.time_embed(t))
        y_embed = self.cond_embed(y).unsqueeze(1)
        
        # Encoding
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))        # 28x28
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))       # 14x14
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))       # 7x7
        h3 = self.attn3(h3, y_embed)
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))       # 4x4
        h4 = self.attn4(h4, y_embed)

        # Decoding
        # h4 (4x4) -> tconv4 -> 7x7 (matches h3)
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        # h (7x7) -> tconv3 -> 14x14 (matches h2)
        h = self.act(self.tgnorm3(self.tconv3(h + h3) + self.dense6(embed)))
        # h (14x14) -> tconv2 -> 28x28 (matches h1)
        h = self.act(self.tgnorm2(self.tconv2(h + h2) + self.dense7(embed)))
        h = self.tconv1(h + h1)

        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h