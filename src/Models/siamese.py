import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        n_filters = 32

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3,stride=2),
            nn.PReLU(),
            
            nn.Conv2d(96, 256, kernel_size=3, stride=2),
            nn.PReLU(),

            nn.Conv2d(256, 384, kernel_size=3,stride=2),
            nn.PReLU()
        )

        # Setting up the Fully Connected Layers
        self.decoder = nn.Sequential(
            nn.Sequential(  # 10, 4
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8
                nn.Conv2d(384, n_filters*4, kernel_size=3, stride=1, padding=1), nn.PReLU()
            ),
            nn.Sequential(  # 10, 8
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8
                nn.Conv2d((n_filters*4), n_filters * 2, kernel_size=3, stride=1, padding=1), nn.PReLU()
            ),
            nn.Sequential(  # 10, 510²
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8
                nn.Conv2d( (n_filters * 2), n_filters, kernel_size=3, stride=1, padding=1), nn.PReLU()
            ),
            nn.Sequential(  # 10, 510²a
                #MOVE TO A NEW FILE
                nn.ConvTranspose2d((n_filters), 1, 4, stride=2, padding=1),
                
                #no conv trans to get rid of checker pattern on first block
                #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8
                #nn.Conv2d(mul_fact *(n_filters), 1, kernel_size=3, stride=1, padding=1), nn.PReLU(),
                nn.Sigmoid()
            )
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.decoder(output)
        return output

    def forward(self, input1, input2, input3):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3