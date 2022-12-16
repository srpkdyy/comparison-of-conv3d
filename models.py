import torch
import torch.nn as nn


def normalize(dim, groups=32):
    return nn.GroupNorm(min(groups, max(1, dim//8)), dim, eps=1e-6)

def activation():
    return nn.SiLU()

def spatial_conv(dim, out_dim=None, k=3, s=1, mode='replicate'):
    return nn.Conv3d(dim, out_dim or dim, (1,k,k), s, (0,k//2,k//2), padding_mode=mode)

def temporal_conv(dim, out_dim=None, k=3, s=1, mode='replicate'):
    return nn.Conv3d(dim, out_dim or dim, (k,1,1), s, (k//2,0,0), padding_mode=mode)


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
            nn.Conv3d(out_dim, out_dim, k, s, p, padding_mode=padmode),
            normalize(out_dim),
            activation(),
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
            normalize(out_dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, (k, 1, 1), s, (p, 0, 0), padding_mode=padmode),
        )
        self.block2 = nn.Sequential(
            normalize(out_dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, (1, k, k), s, (0, p, p), padding_mode=padmode),
            normalize(out_dim),
            activation(),
            nn.Conv3d(out_dim, out_dim, (k, 1, 1), s, (p, 0, 0), padding_mode=padmode),
        )
        self.skip = nn.Identity() if dim == out_dim else nn.Conv3d(dim, out_dim, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)


class ConvBlock(nn.Module):
    def __init__(self, dim, out_dim=None, k=3, s=1, conv='3d', padmode='replicate'):
        super().__init__()
        convs = ('spatial', 'temporal', '3d')
        assert conv in convs, f'Supported convs are {convs}'

        out_dim = out_dim or dim
        self.norm = normalize(dim)
        self.silu = activation()

        if conv == 'spatial':
            self.conv = spatial_conv(dim, out_dim, k, s, mode=padmode)
        elif conv == 'temporal':
            self.conv = temporal_conv(dim, out_dim, k, s, mode=padmode)
        else:
            self.conv = nn.Conv3d(dim, out_dim, k, s, k//2, padding_mode=padmode)

    def forward(self, x):
        return self.conv(self.silu(self.norm(x)))


class STDCBlock(nn.Module):
    def __init__(self, dim, out_dim, n_blocks=4, s=1):
        super().__init__()
        self.stride = s
        blocks = []
        for i in range(n_blocks):
            if i == 0:
                blocks.append(ConvBlock(dim, out_dim//2, k=1))
            elif i == 1 and n_blocks == 2:
                blocks.append(ConvBlock(out_dim//2, out_dim//2, s=s))
            elif i == 1 and n_blocks > 2:
                blocks.append(ConvBlock(out_dim//2, out_dim//4, s=s))
            elif i < n_blocks - 1:
                blocks.append(ConvBlock(out_dim//(2**i), out_dim//(2**(i+1))))
            else:
                blocks.append(ConvBlock(out_dim//(2**i), out_dim//(2**i)))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            outs.append(x)

        return torch.cat(outs, dim=1)


class TwistedBlock(nn.Module):
    def __init__(self, dim, out_dim, n_blocks=4):
        super().__init__()
        assert out_dim%2 == 0, 'out_dim must be divisible by 2'
        out_dim //= 2
        blocks1 = [ConvBlock(dim, out_dim//2, k=1, conv='spatial')]
        blocks2 = [ConvBlock(dim, out_dim//2, k=1, conv='temporal')]
        conv_types = ['spatial', 'temporal']

        for i in range(1, n_blocks):
            type1, type2 = conv_types[i%2], conv_types[(i+1)%2]
            if i < n_blocks - 1:
                in_d, out_d = out_dim//(2**i), out_dim//(2**(i+1))
                blocks1.append(ConvBlock(in_d, out_d, conv=type1))
                blocks2.append(ConvBlock(in_d, out_d, conv=type2))
            else:
                d = out_dim // (2**i)
                blocks1.append(ConvBlock(d, conv=type1))
                blocks2.append(ConvBlock(d, conv=type2))

        self.blocks1 = nn.ModuleList(blocks1)
        self.blocks2 = nn.ModuleList(blocks2)

    def forward(self, x):
        outs = []
        h1 = x
        h2 = x
        for block1, block2 in zip(self.blocks1, self.blocks2):
            h1 = block1(h1)
            h2 = block2(h2)
            outs.extend([h1, h2])
        return torch.cat(outs, dim=1)


class ActionClassifier(nn.Module):
    def __init__(self, block_type='res', ch=3, dims=(64, 128, 256, 512), classes=10):
        super().__init__()
        if block_type == 'res':
            block = ResBlock
        elif block_type == 'legacy':
            block = LegacyResBlock
        elif block_type == 'pseudo':
            block = PseudoBlcok
        elif block_type == 'stdc':
            block = STDCBlock
        elif block_type == 'twisted':
            block = TwistedBlock
        
        self.init_conv = nn.Conv3d(ch, dims[0], 3, 1, 1, padding_mode='replicate')
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


if __name__ == '__main__':
    a = torch.rand(2, 3, 32, 64, 64)
    types = ['res', 'legacy', 'pseudo', 'stdc', 'twisted']
    models = {t: ActionClassifier(t) for t in types}
    print(f'input shape, {a.shape}')
    for t, m in models.items():
        out = m(a)
        print(f'block_type: {t}, Params: {sum(p.numel() for p in m.parameters() if p.requires_grad) // 1000}K')

