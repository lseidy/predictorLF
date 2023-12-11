import torch
import torch.nn as nn
from einops.layers.torch import Reduce
import sys

def preserving_dimensions(module_type, channels_in, channels_out):
    return module_type(channels_in, channels_out, 3, 1, 1)

class Repeat(Reduce):
    def __init__(self, pattern, **axes_lengths):
        super().__init__(pattern, 'repeat', **axes_lengths)

class UNetLike(nn.Module):
    def __init__(self, encoder, decoder, compose=lambda x,y: x): # torch.concat((x,y), axis=1)):
        super().__init__()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.compose = compose
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def save(self, path):
        torch.save(self.state_dict(), path)
    def forward(self, X):
        paths = []
        for enc in self.encoder:
            # print("enc", X.shape)
            paths.append(X)
            X = enc(X)
        # print("enc2", X.shape)
        # print('-----\n')
        paths = paths[::-1][1:]
        X = self.decoder[0](X)
        # print(f'dec {0}: {X.shape}')
        for i, (s, dec) in enumerate(zip(paths, self.decoder[1:-1])):
            # print(f'dec {i + 1}: {X.shape} {s.shape}')
            X = self.compose(X, s)
            X = dec(X)
            # print("shape", X.shape)
        return X
