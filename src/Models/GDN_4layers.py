import os.path
from argparse import Namespace

import torch
import torch.nn as nn
from Models.unetModelGabriele import UNetLike
from Models.ModelGabriele import RegModel
from Models.residualModel import residualCon
from torch.nn import Conv2d, ConvTranspose2d
from compressai.layers.gdn import GDN

#Arch from the paper from attilio
class GDN4l_NN(nn.Module):
    def __init__(self, name, params):
        super().__init__()
        # s, t, u, v = (params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size)
        n_filters = params.num_filters
        print("n_filters: ", n_filters)
        #print("kernels 3 no_skip ", params.no_skip)

        if params.skip_connections == "noSkip":
            type_mode = RegModel
            mul_fact = 1
            print("kernels 3 no-skip")
        elif params.skip_connections == "residual":
            type_mode = residualCon
            mul_fact = 1
            print("kernels 3 Residual")
        elif params.skip_connections == "skip":
            type_mode = UNetLike
            mul_fact = 2
            print("kernels 3 skip")


        flat_model = type_mode([  # 18, 64²
            nn.Sequential(

                Conv2d(1, n_filters, 3, stride=2, padding=1), GDN(n_filters),  # 10, 32²
            ),
            nn.Sequential(
                Conv2d((n_filters), (n_filters*2), 3, stride=2, padding=1), GDN(n_filters*2),  # 10, 16²
            ),
            nn.Sequential(
                Conv2d((n_filters*2), (n_filters*4), 3, stride=2, padding=1), GDN(n_filters*4),  # 10, 8²
            ),
            nn.Sequential(
                Conv2d((n_filters*4), 512, 3, stride=1, padding=1), GDN(512),  # 10, 4
            ),

        ], [
            nn.Sequential(  
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8
                nn.Conv2d(512, n_filters*4, kernel_size=3, stride=1, padding=1), GDN(n_filters*4, inverse=True)
            ),
            nn.Sequential(  
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16
                nn.Conv2d(mul_fact*(n_filters*4), n_filters * 2, kernel_size=3, stride=1, padding=1), GDN(n_filters*2, inverse=True)
            ),
            nn.Sequential(  
                nn.ConvTranspose2d(mul_fact *(n_filters*2), 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )

        ])
        self.network = flat_model
        self.name = name + '.data'
        try:
            if os.path.exists(self.name):
                self.load_state_dict(torch.load(self.name))
        except RuntimeError:
            pass
    
    def save(self):
        torch.save(self.state_dict(), self.name)
    def forward(self, X):
        #assert(tuple(X.shape[1:]) == (1,8*64,8*64))
        return self.network(X)


#
#params = Namespace()
#dims = (8,1,64,64)
#dims_out = (8,1,32,32)
#(params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size) = dims
#params.num_filters = 32
#params.skip_connections = "noSkip"
## print(params)
#model = UNetSpace("unet_space", params)
#model.eval()
#zeros = torch.zeros(1, 1, 64, 64)
#zeros_t = torch.zeros(8, 1, 32, 32)
#lossf = nn.MSELoss()
#from torchsummary import summary
#with torch.no_grad():
#    batch_size = model(zeros)
#    # print("batch_size: ", batch_size.shape)
#    # batch_size = batch_size[:,:,-32:, -32:]
#    summary(model, (1, 64, 64))
#    # print(batch_size.shape)
#    batch_size = batch_size[:, :, -32:, -32:]
#    # print(batch_size.shape)
#    # print(lossf(zeros_t, batch_size))