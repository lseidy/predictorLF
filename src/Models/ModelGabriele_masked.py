import torch
import torch.nn as nn
#from einops.layers.torch import Reduce
import sys

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, _, h, w = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, h // 2, w // 2:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class MaskedModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, X):
        for enc in self.encoder:
            X = enc(X)
        for dec in self.decoder:
            X = dec(X)

        return X
