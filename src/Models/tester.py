from argparse import Namespace

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the model class
from u3k_5L_S2_1view import UNetSpace


params = Namespace()
dims = (1,1,64,64)
dims_out = (1,1,32,32)
(params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size) = dims_out
params.num_filters = 32

# Load the saved model
model = UNetSpace("unet", params)
# model.load_state_dict(torch.load('/home/machado/saved_models/assimetricNet_average_UnetGabriele_12_0.0001.pth.tar'))
model.load_state_dict(torch.load('/home/idm/assimetricNet_average_UnetGabriele_12_0.0001.pth.tar', map_location=torch.device('cpu')))
model.eval()

# Load the input image using Pillow
# input_image_path = '/home/machado/Lenslet_8x8_Gscale/Light/Graffiti.png'
input_image_path = '/home/idm/New_Extracted_Dataset/Lenslet_8x8_Gscale/Light/Graffiti.png'
input_image = Image.open(input_image_path)

# Convert the image to a PyTorch tensor
preprocess = transforms.Compose([
    transforms.ToTensor(),
])

input_tensor = preprocess(input_image)
input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension
print(input_tensor.shape)

# Define block size
block_size = 64
stride = 32

# Get the image dimensions
width, height= input_image.size
print(height, width)
# exit()
# Initialize an empty array to store the output blocks
output_blocks = np.zeros((height, width))

# Process each block of the input image
for i in range(stride, height-stride, stride):
    for j in range(stride, width-stride, stride):
        # Extract a block of size 64x64
        # if i+block_size+1 < height and j+block_size+1 < width:
        block = input_tensor[:, :, i:i+block_size, j:j+block_size]
        # print(block.shape)
        # print(i+block_size)

        avgtop = input_tensor[:,:, :32, :].mean()
        avgleft = input_tensor[:,:, 32:, :32].mean()

        block[:, :, 32:, 32:] = torch.full((1, 1, 32, 32), ((avgtop+avgleft)/2))

        # Forward pass through the model
        with torch.no_grad():
            output_block = model(block)

        # Convert the output block to a NumPy array
        print("out", output_block.shape)
        output_block = output_block[:, :, 32:, 32:].squeeze().numpy()
        print("out2 ", output_block.shape)
        # Store the output block in the array
        output_blocks[i:i+stride, j:j+stride] = output_block
        print("outs ", output_blocks.shape)



print("outs2 ", output_blocks.shape)
# Create a new image using the output blocks
output_image = Image.fromarray(np.uint8(output_blocks))
print("outs2 ", output_image.size)

# Save the output image
output_image.save('output_image2.jpg')