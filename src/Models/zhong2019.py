import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn

class zhongModel(nn.Module):

    def __init__(self, params):
        super(zhongModel, self).__init__()

        

        # MAINTAIN THE SAME ENCODER 1 CHANNEL 32X32
        # concatenating the last layer 1536x2x2
        self.cnn1 = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=5, stride=1, padding=2), nn.ReLU(),  #
               
            
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0), nn.ReLU(),  #
                

                nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),  #
                
        )
      
                
           
    
        
    
    def forward(self, input1):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output = self.cnn1(input1)
        return output

#from argparse import Namespace
#params = Namespace()
#dims = (8,1,64,64)
#dims_out = (8,1,32,32)
#(params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size) = dims
#params.num_filters = 16
#params.skip = False
#model = zhongModel(params)
#
#from torchsummary import summary
#with torch.no_grad():
#   summary(model, (3, 32, 32))

