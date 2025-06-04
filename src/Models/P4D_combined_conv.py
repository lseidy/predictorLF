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
import torch.nn.functional as F

from argparse import Namespace


class AddDimension(nn.Module):
    def forward(self, x):
        return x.unsqueeze(0)  # Adiciona uma dimensão na posição 2
class RemDimension(nn.Module):
    def forward(self, x):
        return x.squeeze(0)  # Adiciona uma dimensão na posição 2

def extract_epi(lf_tensor, direction='vertical', idx=0):
        """
        Extrai uma EPI e opcionalmente a visualiza.

        Args:
            lf_tensor (Tensor): Tensor com shape (B, U, V, H, W)
            direction (str): 'horizontal' ou 'vertical'
            idx (int): índice da amostra no batch

        Returns:
            Tensor: EPI concatenada com shape (1, U, V, L), onde L depende da direção
        """
        with torch.no_grad():
            print("Shape do tensor de entrada:", lf_tensor.shape)
            epis = []

            if direction == 'horizontal':
                for y in range(lf_tensor.shape[3]):  # H
                    epi = lf_tensor[idx, :, :, y, :]
                    epis.append(epi)
                epis = torch.cat(epis, dim=2)  # concatena no eixo W
            elif direction == 'vertical':
                for x in range(lf_tensor.shape[4]):  # W
                    epi = lf_tensor[idx, :, :, :, x]
                    epis.append(epi)
                epis = torch.cat(epis, dim=2)  # concatena no eixo H
            else:
                raise ValueError("Direção deve ser 'horizontal' ou 'vertical'")

            epi_tensor = epis.unsqueeze(0)

            return epi_tensor

def lenslet_to_lf(lenslet_img, V=16, H=16):
    """
    Converte imagem lenslet (ex: 256x256) em tensor Light Field com shape (B, V, H, Y, X)

    Args:
        lenslet_img (Tensor): imagem com shape (B, H_full, W_full)
        V (int): número de views verticais
        H (int): número de views horizontais

    Returns:
        Tensor: Light Field com shape (B, V, H, Y, X)
    """
    B, H_full, W_full = lenslet_img.shape
    subimg_Y = H_full // V
    subimg_X = W_full // H

    lf = torch.zeros((B, V, H, subimg_Y, subimg_X), dtype=lenslet_img.dtype)

    for v in range(V):
        for h in range(H):
            lf[:, v, h, :, :] = lenslet_img[:, v::V, h::H]

    return lf
class UpsampleLayer(nn.Module):
    def __init__(self, target_size):
        super(UpsampleLayer, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode='trilinear', align_corners=False)

class P4D(nn.Module):
    def __init__(self,params):
        super(P4D, self).__init__()
        n_filters = 32

        self.spatial = nn.Sequential(
            nn.Conv3d(1, n_filters, kernel_size=(3,1,1), padding=1), nn.PReLU(),
            nn.Conv3d(n_filters, n_filters*2, kernel_size=(3,1,1), stride=2, padding=1), nn.PReLU(),
            nn.Conv3d(n_filters*2, n_filters*4, kernel_size=(3,1,1), stride=2, padding=1), nn.PReLU(),
            nn.Conv3d(n_filters*4, n_filters*8, kernel_size=(3,1,1), stride=2, padding=1), nn.PReLU(),
        )
        self.angular = nn.Sequential(
            nn.Conv3d(1, n_filters, kernel_size=(1,3,3), padding=1), nn.PReLU(),
            nn.Conv3d(n_filters, n_filters*2, kernel_size=(1,3,3), stride=2, padding=1), nn.PReLU(),
            nn.Conv3d(n_filters*2, n_filters*4, kernel_size=(1,3,3), stride=2, padding=1), nn.PReLU(),
            nn.Conv3d(n_filters*4, n_filters*8, kernel_size=(1,3,3), stride=2, padding=1), nn.PReLU(),
        )
        self.epi_h = nn.Sequential(
            nn.Conv3d(1, n_filters, kernel_size=(1,1,3), padding=1), nn.PReLU(),
            nn.Conv3d(n_filters, n_filters*2, kernel_size=(1,1,3), stride=2, padding=1), nn.PReLU(),
            nn.Conv3d(n_filters*2, n_filters*4, kernel_size=(1,1,3), stride=2, padding=1), nn.PReLU(),
            nn.Conv3d(n_filters*4, n_filters*8, kernel_size=(1,1,3), stride=2, padding=1), nn.PReLU(),
        )
        self.epi_v = nn.Sequential(
            nn.Conv3d(1, n_filters, kernel_size=(1,1,3), padding=1), nn.PReLU(),
            nn.Conv3d(n_filters, n_filters*2, kernel_size=(1,1,3), stride=2, padding=1), nn.PReLU(),
            nn.Conv3d(n_filters*2, n_filters*4, kernel_size=(1,1,3), stride=2, padding=1), nn.PReLU(),
            nn.Conv3d(n_filters*4, n_filters*8, kernel_size=(1,1,3), stride=2, padding=1), nn.PReLU(),
        )

        # Upsample para spatial
        self.Upsample = nn.Sequential(
            #AddDimension(),
            UpsampleLayer(target_size=(4, 4, 2)),  # Aplica interpolação de trilinear
            nn.Conv3d(n_filters*8, n_filters*8, kernel_size=3, stride=1,padding=1), nn.PReLU(),
            #RemDimension()
        )


        self.all = nn.Sequential(
            nn.Conv3d(n_filters*8, n_filters*8, kernel_size=3, stride=1,padding=1), nn.PReLU()
        )

        self.deconv = nn.Sequential( # entrada 4x4x2
            nn.ConvTranspose3d(n_filters*8, n_filters*4, kernel_size=3, stride=1, padding=(1,1,0)), nn.PReLU(),

            nn.ConvTranspose3d(n_filters*4, n_filters*2, kernel_size=3, stride=2, padding=2), nn.PReLU(),
       #
            nn.ConvTranspose3d(n_filters*2, n_filters, kernel_size=3, stride=2, padding=2), nn.PReLU(),
            #
            nn.ConvTranspose3d(n_filters, 1, kernel_size=3, stride=2, padding=0, output_padding=(1)), nn.PReLU(),
            
        )

    def forward(self, input1, EPI_h, EPI_v):
        #print("--------------------\n INPUT: ",input1.shape,EPI_h.shape, EPI_v.shape,"\n--------------------")
        spatial = self.spatial(input1)
        #print("--------------------\n Spatial: ",spatial.shape,"\n--------------------")
        
        angular = self.angular(input1)
        #print("--------------------\n Angular: ",angular.shape,"\n--------------------")
        
        epi_h = self.epi_h(EPI_h)
        #print("--------------------\n EPI_H: ",epi_h.shape,"\n--------------------")
        
        epi_v = self.epi_v(EPI_v)
        #print("--------------------\n EPI_V: ",epi_v.shape,"\n--------------------")

        upsample_spatial = self.Upsample(spatial)
        #print("--------------------\n Upsample Spatial: ", upsample_spatial.shape,"\n--------------------")

        upsample_angular = self.Upsample(angular)
        #print("--------------------\n Upsample Angular: ", upsample_angular.shape,"\n--------------------")

        output = upsample_angular + upsample_spatial + epi_h + epi_v
        #print("--------------------\n output: ",output.shape,"\n--------------------")

        output = self.all(output)
        #print("--------------------\n output2: ",output.shape,"\n--------------------")
        output = self.deconv(output)
        #print("--------------------\n output3: ",output.shape,"\n--------------------")

        return output

"""  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = P4D().to(device)
model.eval()

#EPI training setup
file_path = "/mnt/c/Users/lucas/Documents/LF/Testes/train.pt"
train = torch.load(file_path).to(device)

train_epi = train.clone()
train_epi = lenslet_to_lf(train_epi)

EPI_h = extract_epi(train_epi, 'horizontal')
EPI_v = extract_epi(train_epi, 'vertical')

file_path = "/mnt/c/Users/lucas/Documents/LF/Testes/test.pt"
test = torch.load(file_path).to(device)

train = train.view(1, 4, 16, 4, 16).permute(0, 1, 3, 2, 4).reshape(1, 16, 16, 16)

lossf = nn.L1Loss()

with torch.no_grad():
    batch_size = model(train, EPI_h, EPI_v)
    rem = RemDimension()
    batch_size= rem(batch_size)
    print("batch_size: ", batch_size.shape)
    
    #summary(model, train.shape, device=str(device))

    print("loss: ", lossf(test, batch_size))
# Suponha que batch_size tenha shape (1, N, H, W)
batch_size = torch.split(batch_size, 1, dim=1)  # vira lista de (1, 1, H, W)

# Remove a dimensão do canal
batch = torch.stack([mi.squeeze(1) for mi in batch_size])  # (N, H, W)

# Divide em grupos de 8
chunks = torch.split(batch, 4)

# Junta cada grupo horizontalmente (dim=2), depois os blocos verticalmente (dim=1)
predicted_block = torch.cat(
    [torch.cat(list(chunk), dim=2) for chunk in chunks],
    dim=1
)

# Normalização [0,1]
predicted_block = (predicted_block - predicted_block.min()) / (predicted_block.max() - predicted_block.min())

print(predicted_block.shape)
# Salvar imagem
save_dir = "result_conv"
os.makedirs(save_dir, exist_ok=True)

file_path = os.path.join(save_dir, 'result_3.png')
torchvision.utils.save_image(predicted_block, fp=file_path, format="png")

print(f'Tensor salvo em {file_path}')
"""