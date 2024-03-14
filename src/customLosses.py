from scipy.linalg import hadamard
import numpy as np
import torch
import torch.nn as nn
import LightField as LF
from quantizer import lowPass
class CustomLoss(nn.Module):
    def __init__(self, loss, quantization, denormalize, blockSize):
        super(CustomLoss, self).__init__()
        self.loss_type = loss
        
       
        self.quantization = quantization
        self.quantizer = lowPass(blockSize)
        self.denormalize = denormalize

        if loss == 'satd':
            self.transform = self.hadamard_transform
        elif loss == 'dct':
            self.transform = self.dct2
        else:
            print("LOSS NOT FOUND!")


    def hadamard_transform(self, block):
        hadamard_transform = torch.from_numpy(hadamard(block.shape[-1])).to(block.device, dtype=torch.float32)
        return torch.matmul(hadamard_transform, block)

    def dct2(self, x):
        x = torch.fft.fftn(x, dim=(-2, -1))
        x_cos = torch.cos(torch.arange(x.shape[-2]).to(x.device).unsqueeze(-1) * torch.arange(x.shape[-1]).to(x.device).unsqueeze(0) * 2 * np.pi / x.shape[-2])
        x_cos = x_cos.unsqueeze(0).unsqueeze(0)  # Reshape to broadcast
        x *= x_cos
        return x

    def custom_loss(self, original, pred):
        if self.denormalize:
            res = LF.denormalize_image(original,8) - LF.denormalize_image(pred,8)
        else:
            res = original - pred

        if self.quantization:
            return torch.sum(torch.abs(self.quantizer.quantize(self.transform(res))))
        else:
            return torch.sum(torch.abs((self.transform(res))))

    def forward(self, original, pred):
        return self.custom_loss(original, pred)