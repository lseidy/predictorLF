from scipy.fftpack import dct
from scipy.linalg import hadamard
import numpy as np
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, loss):
        super(CustomLoss, self).__init__()
        self.loss_type = loss
    def hadamard_transform(self, block):
        hadamard_transform = torch.from_numpy(hadamard(block.shape[-1])).to(block.device, dtype=torch.float32)
        return torch.matmul(hadamard_transform, block)

    def dct2(self, x):

    # Apply 1D DCT along each dimension separately
        x = torch.fft.fftn(x, dim=(-2, -1))
        x_cos = torch.cos(torch.arange(x.shape[-2]).to(x.device).unsqueeze(-1) * torch.arange(x.shape[-1]).to(x.device).unsqueeze(0) * 2 * np.pi / x.shape[-2])
        x_cos = x_cos.unsqueeze(0).unsqueeze(0)  # Reshape to broadcast
        x *= x_cos
        return x

    def custom_loss(self, original, pred):
        if self.loss_type == 'satd':
            return torch.sum(torch.abs(self.hadamard_transform(original - pred)))
        
        if self.loss_type == 'dct':
            res = original - pred
            return torch.sum(torch.abs(self.dct2(res)))
        else:
            print("LOSS NOT FOUND!")

    def forward(self, original, pred):
        return self.custom_loss(original, pred)