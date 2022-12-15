import torch
import torch.nn as nn


def normalize(dim):
    return nn.GroupNorm(32, dim, eps=1e-6)

def activation():
    return nn.SiLU()


class LegacyResBlock(nn.Module):
    def __init__(self, dim, out_dim=None, k=3, s=1, p=1, padmode='replicate'):
        super().__init__()
        out_dim = out_dim or dim
        self.block1 = nn.Sequential(
            nn.Conv3d(dim, out_dim, k, s, p, padding_mode=padmode),
            normalize(out_dim),
            activation(),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(out_dim, out_dim, k, s, p, padding_mode=padmode)
            normalize(out_dim),
            activation(),
        )
        self.skip = nn.Identity() if dim == out_dim else nn.Conv3d(dim, out_dim, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)

class ResBlock(nn.Module):
    def __init__(self, dim, out_dim=None, k=3, s=1, p=1, padmode='replicate'):
        super().__init__()
        out_dim = out_dim or dim
        self.block1 = nn.Sequential(
            normalize(dim),
            activation(),
            nn.Conv3d(dim, out_dim, k, s, p, padding_mode=padmode),
        )
        self.block2 = nn.Sequential(
            normalize(out_dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, k, s, p, padding_mode=padmode)
        )
        self.skip = nn.Identity() if dim == out_dim else nn.Conv3d(dim, out_dim, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)


class PseudoBlcok(nn.Module):
    def __init__(self, dim, out_dim=None, k=3, s=1, p=1, padmode='replicate'):
        super().__init__()
        out_dim = out_dim or dim
        self.block1 = nn.Sequential(
            normalize(dim),
            activation(),
            nn.Conv3d(dim, out_dim, (1, k, k), s, (0, p, p), padding_mode=padmode),
            normalize(dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, (k, 1, 1), s, (p, 0, 0), padding_mode=padmode),
        )
        self.block2 = nn.Sequential(
            normalize(dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, (1, k, k), s, (0, p, p), padding_mode=padmode),
            normalize(dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, (k, 1, 1), s, (p, 0, 0), padding_mode=padmode),
        )
        self.skip = nn.Identity() if dim == out_dim else nn.Conv3d(dim, out_dim, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)


class ConvBlock(nn.Module):
    def __init__(self, dim, out_dim=None, k=3, s=1, padmode='replicate'):
        super().__init__()
        out_dim = out_dim or dim
        self.norm = normalize(dim)
        self.silu = activation()
        self.conv = nn.Conv3d(dim, out_dim, k, s, k//2, padding_mode=padmode)

    def forward(self, x):
        return self.conv(self.silu(self.norm(x)))


class STDCBlock(nn.Module):
    def __init__(self, dim, out_dim, n_blocks=4, s=1):
        self.stride = s
        blocks = []
        for i in range(n_blocks):
            if i == 0:
                blocks.append(ConvBlock(dim, out_dim//2, 1))
            elif i == 1 and n_blocks == 2:
                blocks.append(ConvBlock(out_dim//2, out_dim//2, stride=s))
            elif i == 1 and n_blocks > 2:
                blocks.append(ConvBlock(out_dim//2, out_dim//4, stride=s))
            elif i < n_blocks - 1:
                blocks.append(ConvBlock(out_dim//(2**i), out_dim//(2**(i+1))))
            else:
                blocks.append(ConvBlock(out_dim//(2**i), out_dim//(2**i)))

        self.blocks = nn.ModuleList(*blocks)

        if s == 2:
            self.

    def forward(self, x):
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            outs.append(x)

        return torch.cat(outs, dim=1)


class ActionClassifier(nn.Module):
    def __init__(self, block_type='res', ch=3, dims=(64, 128, 256, 512), classes=10):
        super().__init__()
        if block_type == 'res':
            block = ResBlock
        elif block_type == 'pseudo':
            block = PseudoBlcok

        
        self.init_conv = nn.Conv3d(ch, dims[0], (1, 4, 4), (1, 2, 2), (0, 1, 1), padding_mode='replicate')
        in_dim = dims[0]
        layers = []
        for i, dim in enumerate(dims):
            layers.append(block(in_dim, dim))
            layers.append(block(dim, dim))
            if i < len(dims) - 1:
                layers.append(nn.AvgPool3d(2, 2))
            else:
                layers.append(nn.AdaptiveAvgPool3d(1))
            in_dim = dim
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_dim, classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        return self.layers(x)
