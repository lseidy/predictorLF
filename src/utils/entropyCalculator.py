import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.measure import shannon_entropy
from skimage.filters.rank import entropy as entr
from skimage.morphology import square
import numpy as np
from scipy.stats import entropy as scip 

def image_to_tensor(image_path):
    image = Image.open(image_path)
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    print(image_tensor.shape)
    image_tensor = image_tensor[:,0,:,:]
    return image_tensor

def compute_residual_entropy(image1, image2):
    # Convert images to tensors
    tensor1 = image_to_tensor(image1)
    tensor2 = image_to_tensor(image2)
    print(tensor1)

    # Compute residuals
    residual = torch.abs(tensor1 - tensor2)

    # Calculate entropy of the residuals
    entropy = -torch.sum(residual * torch.log2(residual + 1e-10))

    squ= square(1)
    print(squ)
    print(residual.shape)
    return entropy.item(), shannon_entropy(residual), np.mean(entr(residual[0,:,:], square(32))), scip(residual[0,:,:])

# Example usage:
image1_path = "/home/machado/110_original.png"
image2_path = "/home/machado/110_predicted.png"

residual_entropy, shanon, squ, scipe = compute_residual_entropy(image1_path, image2_path)
print("Entropy of the residuals:", residual_entropy, shanon, squ, np.mean(scipe))
