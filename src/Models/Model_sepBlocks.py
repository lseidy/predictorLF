import torch
import torch.nn as nn
#from einops.layers.torch import Reduce
import sys

#MODEL WITH NO SKIP

class RegModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, in1):
        for i,enc in enumerate(self.encoder):
            in1 = enc(in1)

        for i, (dec) in enumerate(self.decoder):
            in1 = dec(in1)

        return in1
