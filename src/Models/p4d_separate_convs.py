import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms.functional as TF
import os


class AddDimension(nn.Module):
    def forward(self, x):
        return x.unsqueeze(0)  # Adiciona uma dimensão na posição 2
class RemDimension(nn.Module):
    def forward(self, x):
        return x.squeeze(0)  # Adiciona uma dimensão na posição 2

class P4D(nn.Module):

    def __init__(self, params):
        super(P4D, self).__init__()
        num_filters = params.num_filters
 
        self.spatial = nn.Sequential( #64x8x8
            nn.Conv3d(in_channels=1, out_channels=num_filters, kernel_size=(3, 1, 1), stride=(2,1,1), padding=(1,0,0)), nn.PReLU(),

            nn.Conv3d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0)), nn.PReLU(),
            nn.Conv3d(in_channels=num_filters*2, out_channels=num_filters*2, kernel_size=(3, 1, 1), stride=(2,1,1), padding=(1,0,0)), nn.PReLU(),

            nn.Conv3d(in_channels=num_filters*2, out_channels=num_filters*4, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0)), nn.PReLU(),
            nn.Conv3d(in_channels=num_filters*4, out_channels=num_filters*4, kernel_size=(3, 1, 1), stride=(2,1,1), padding=(1,0,0)), nn.PReLU(), 

            nn.Conv3d(in_channels=num_filters*4, out_channels=512, kernel_size=(3, 1, 1), stride= 1, padding=(1,0,0)), nn.PReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 1, 1), stride=(2,1,1), padding=(1,0,0)), nn.PReLU(), 
 
        )

        self.angular =nn.Sequential( #8²
            nn.Conv3d(in_channels=1, out_channels=num_filters, kernel_size=(1, 3, 3), stride=1, padding=(0,1,1)), nn.PReLU(),
            nn.Conv3d(in_channels=num_filters, out_channels=num_filters, kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0,1,1)), nn.PReLU(),

            nn.Conv3d(in_channels=num_filters, out_channels=512, kernel_size=(1, 3, 3), stride=1, padding=(0,1,1)), nn.PReLU(),
            
        )

        self.decoder =nn.Sequential( #64,4,4
            AddDimension(),
            
            nn.Upsample(scale_factor=(1,1,1), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1), nn.PReLU(),
            
            nn.Upsample(scale_factor=(1,1,1), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1), nn.PReLU(),
            
            nn.Upsample(scale_factor=(1,1,1), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), nn.PReLU(),
###
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), nn.PReLU(),
            
            nn.ConvTranspose3d(in_channels= 32, out_channels= 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid(),
            

        )
        self.decoder1 =nn.Sequential( #4,8,8
            AddDimension(),
            
            nn.Upsample(scale_factor=(2,1,1), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1), nn.PReLU(),
            
            nn.Upsample(scale_factor=(2,1,1), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1), nn.PReLU(),
            
            nn.Upsample(scale_factor=(2,1,1), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), nn.PReLU(),
    ##
            nn.Upsample(scale_factor=(2,1,1), mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), nn.PReLU(),
            
            nn.ConvTranspose3d(in_channels= 32, out_channels= 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid(),
            
        )
        self.feature =nn.Sequential( #1,64,8,8
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3), stride=1, padding=(0,1,1)), nn.PReLU(),
        )
        

    def forward(self, input1):

        output0 = self.angular(input1)
        #print("--------------------\n Output: ",output0.shape,"\n--------------------" )
        output1 = self.spatial(input1)
        #print("--------------------\n Output1: ",output1.shape,"\n--------------------" )
        output2 = self.decoder(output0)
        #print("--------------------\n Output2: ",output2.shape,"\n--------------------" )
        output3 = self.decoder1(output1)
        #print("--------------------\n Output3: ",output3.shape,"\n--------------------" )

        output = self.feature(output2+output3)

        return output
#from argparse import Namespace
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = P4D(num_filters=32).to(device)
#model.eval()
#file_path = "/mnt/c/Users/lucas/Documents/TCC/Pytorch_learn/saved_testBlock/blockTrainv3.pt"
#train = torch.load(file_path).to(device)
#file_path = "/mnt/c/Users/lucas/Documents/TCC/Pytorch_learn/saved_testBlock/blockTestv2.pt"
#test = torch.load(file_path).to(device)
#lossf = nn.MSELoss()
#num_filters = 32
#
#from torchsummary import summary
#with torch.no_grad():
#    batch_size = model(train)
#    rem = RemDimension()
#    batch_size= rem(batch_size)
#    print("batch_size: ", batch_size.shape)
#    #summary(model, train.shape, device=str(device))
#    print(batch_size.shape)
#
#    print("loss: ", lossf(test, batch_size))
#
#
#batch_size = torch.split(batch_size, 1,dim=1)
#
#predicted_block = []
#temp_block = []
#for i,mi in enumerate(batch_size):
#    #print(mi.shape)
#    temp_block.append(mi.squeeze(1))
#    if (i+1) % 8 == 0:
#        predicted_block.append(torch.cat(temp_block,dim=2))
#        temp_block = []
#predicted_block = torch.cat(predicted_block,dim=1)
##print(predicted_block.shape)
#         
#predicted_block = (predicted_block - predicted_block.min()) / (predicted_block.max() - predicted_block.min())
##imagem_pil = TF.to_pil_image(predicted_block)
#
## Converter a imagem PIL para o canal de luminância (escala de cinza)
##imagem_luminancia = imagem_pil.convert('L')
#
#save_dir="result_conv"
#os.makedirs(save_dir, exist_ok=True)
#
#file_path = os.path.join(save_dir,f'result_3.png')
#torchvision.utils.save_image(predicted_block,fp= file_path, format ="png" )
#print(f'Tensor {i} salvo em {file_path}')
#