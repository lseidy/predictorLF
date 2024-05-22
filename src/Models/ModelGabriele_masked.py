import torch
import torch.nn as nn
import torch.nn.functional as F

#from einops.layers.torch import Reduce
import sys

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, nFilters, kernel_size=4, stride = 2, pad = 1):
        super().__init__()
        self.conv = PartialConv2d(in_channels, nFilters, kernel_size=kernel_size, stride = stride, padding = pad)
        self.relu = nn.PReLU()
    
    def forward(self, x, mask):
        x, update_mask = self.conv(x, mask)
        #x = self.bn(x)
        x = self.relu(x)
        return x, update_mask

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, nFilters, kernel_size=3, stride = 1, pad = 1):
        super().__init__()
        self.upscale = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv = PartialConv2d(in_channels, nFilters, kernel_size=kernel_size, stride=stride, padding=pad)
        self.relu = nn.PReLU()
    def forward(self, x, mask):
        x = self.upscale(x)
        mask = self.upscale(mask)
        x, update_mask = self.conv(x, mask)
        #x = self.bn(x)
        x = self.relu(x)
        return x, update_mask

class PartialConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        # Inherit the parent class (Conv2d)
        super(PartialConv2d, self).__init__(in_channels, out_channels,
                                            kernel_size, stride=stride,
                                            padding=padding, dilation=dilation,
                                            groups=groups, bias=bias,
                                            padding_mode=padding_mode)

        # Define the kernel for updating mask
        self.mask_kernel = torch.ones(self.out_channels, self.in_channels,
                                      self.kernel_size[0], self.kernel_size[1])
        # Define sum1 for renormalization
        self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] \
                                              * self.mask_kernel.shape[3]
        # Define the updated mask
        self.update_mask = None
        # Define the mask ratio (sum(1) / sum(M))
        self.mask_ratio = None
        # Initialize the weights for image convolution
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, img, mask):
        with torch.no_grad():
            if self.mask_kernel.type() != img.type():
                self.mask_kernel = self.mask_kernel.to(img)
            # Create the updated mask
            # for calcurating mask ratio (sum(1) / sum(M))
            self.update_mask = F.conv2d(mask, self.mask_kernel,
                                        bias=None, stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=1)
            # calcurate mask ratio (sum(1) / sum(M))
            self.mask_ratio = self.sum1 / (self.update_mask + 1e-8)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
        # calcurate WT . (X * M)
        conved = torch.mul(img, mask)
        conved = F.conv2d(conved, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        if self.bias is not None:
            # Maltuply WT . (X * M) and sum(1) / sum(M) and Add the bias
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(conved - bias_view, self.mask_ratio) + bias_view
            # The masked part pixel is updated to 0
            output = torch.mul(output, self.mask_ratio)
        else:
            # Multiply WT . (X * M) and sum(1) / sum(M)
            output = torch.mul(conved, self.mask_ratio)

        return output, self.update_mask


class ModelPartialConv(nn.Module):
    def __init__(self, name, params):

        super().__init__()
        print('Using model Conv')

        
        
        in_channels = 1
        out_channels = 1
        nFilters = params.num_filters

        self.enc11 = EncoderLayer(in_channels, nFilters, kernel_size=3, stride=1)
        self.enc12 = EncoderLayer(nFilters, nFilters, kernel_size=3, stride=2)
        in_channels = nFilters
        nFilters = nFilters * 2


        self.enc21 = EncoderLayer(in_channels, nFilters, kernel_size=3, stride=1)
        self.enc22 = EncoderLayer(nFilters, nFilters, kernel_size=3, stride=2)
        in_channels = nFilters
        nFilters = nFilters * 2

        self.enc31 = EncoderLayer(in_channels, nFilters, kernel_size=3, stride=1)
        self.enc32 = EncoderLayer(nFilters, nFilters, kernel_size=3, stride=2)
        in_channels = nFilters
        nFilters = nFilters * 2

        self.enc41 = EncoderLayer(in_channels, nFilters, kernel_size=3, stride=1)
        self.enc42 = EncoderLayer(nFilters, nFilters, kernel_size=3, stride=2)
        in_channels = nFilters
        nFilters = nFilters * 2

        self.encoder = EncoderLayer(in_channels, 512, kernel_size=3, stride=1)


        #Decoder Definition
        nFilters = nFilters // 2
        self.d1 = DecoderLayer(512, nFilters)
        in_channels = nFilters
        nFilters = nFilters // 2
        self.d2 = DecoderLayer(in_channels, nFilters)
        in_channels = nFilters
        nFilters = nFilters // 2
        self.d3 = DecoderLayer(in_channels, nFilters)
        self.d4 = nn.ConvTranspose2d(nFilters, out_channels, kernel_size = 4, stride = 2, padding=1)
        self.sigmoid = nn.Sigmoid()




    def forward(self, x, mask):

        x, mask = self.enc11(x, mask)
        x, mask = self.enc12(x, mask)
        x, mask = self.enc21(x, mask)
        x, mask = self.enc22(x, mask)
        x, mask = self.enc31(x, mask)
        x, mask = self.enc32(x, mask)
        x, mask = self.enc41(x, mask)
        x, mask = self.enc42(x, mask)
        x, mask = self.encoder(x, mask)
        x, mask = self.d1(x, mask)
        x, mask = self.d2(x, mask)
        x, mask = self.d3(x, mask)
        x = self.d4(x)
        x = self.sigmoid(x)
        return x