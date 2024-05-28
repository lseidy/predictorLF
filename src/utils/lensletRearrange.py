import os

import einops as ein

import torch
import numpy as np
from PIL import Image

# normalizer_factor = 2/(2 ** 16 - 1)
#
# img = cv2.imread("/home/idm/Divided.png", cv2.IMREAD_UNCHANGED)
# # img_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
# # img_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB))
# # img_normalizada = img_ycbcr.astype(np.float32) * 255
#
# # img_normalizada = img_ycbcr.astype(np.float32) * normalizer_factor
#
# # funciona
# # img = np.load('/home/idm/vp_stack1.png.npy', allow_pickle=False)
# lenslet = ein.rearrange(img,' w h c ->  w h c')
#
# plt.imsave("/home/idm/savetest.png", lenslet)

# plt.figure()
# plt.imshow(img_ycbcr, interpolation='none')
# plt.grid(False)
# plt.title('lenslet')
# plt.show()


# img = Image.open("/home/idm/nonDivided.png")




def multiview2lenslet(path, path_rgb, path_gscale):
    print(path)
    img = (Image.open(path))
    lf_name = path.split("/")[-1]
    image_array = np.array(img)
    image_array = ein.rearrange(image_array, '(v h) (u w)  c -> (h v) (w u)  c', u=16, v=16)
    # Convert the NumPy array back to an image using Pillow
    reconstructed_image = Image.fromarray(image_array)

    grayscale_image = reconstructed_image.convert("L")
    grayscale_image.save(os.path.join(path_gscale, lf_name))

    reconstructed_image.save(os.path.join(path_rgb, lf_name))

    # reconstructed_image.show()
    # grayscale_image.show()
    # return reconstructed_image
    # To save the image to a new file
    # reconstructed_image.save("pillowTest.png")

def lesnlet2multiview(img, path_rgb, path_gscale, lf_name):
    image_array = np.array(img)
    image_array = ein.rearrange(image_array, '(h v) (w u)  c -> (v h) (u w)  c', u=8, v=8)
    # Convert the NumPy array back to an image using Pillow
    reconstructed_image = Image.fromarray(image_array)

    # grayscale_image = reconstructed_image.convert("L")
    reconstructed_image.save(os.path.join(path_gscale, lf_name))

    # reconstructed_image.save(os.path.join(path_rgb, lf_name))

def rotateBlocks(img, path_rgb, path_gscale, lf_name):
    b = 32
    image_array = np.array(img)
    image_array =torch.convert_to_tensor(image_array)
    outputImage = torch.zeros(image_array.shape)
    k, l = 0, 0
    print(image_array.shape)
    for i in range (int(image_array.shape[0]/32)-32):
        for j in range(int(image_array.shape[1]/32)-32):
            for c in range(int(image_array.shape[2]/32)-32):
                outputImage[i*32:(i+1)*32, j*32:(j+1)*32, :] = outputImage[l*32:(l+1)*32, k*32:(k+1)*32, :]

            l+= 32
            if l >= image_array.shape[0]:
                l=0
                k+=32






    # Convert the NumPy array back to an image using Pillow
    reconstructed_image = Image.fromarray(outputImage)

    # grayscale_image = reconstructed_image.convert("L")
    reconstructed_image.save(os.path.join(path_gscale, lf_name))
#
#path="/home/machado/New_Extracted_Dataset/Multiview_16x16/"
#pathOut="/home/machado/New_Extracted_Dataset/Lenslet_16x16_RGB/"
#pathOutg='/home/machado/New_Extracted_Dataset/Lenslet_16x16_Gscale/'


 #img = (Image.open("/home/idm/New_Extracted_Dataset/Lenslet_8x8_Gscale/Urban/Bikes.png"))
 #img = (Image.open(path))
 #rotateBlocks(img, "/home/idm/", "/home/idm/", "bikes_lens_predicted3.png")
#for classe in os.listdir(path):
#    os.makedirs(os.path.join(pathOut, classe), exist_ok=True)
#    os.makedirs(os.path.join(pathOutg, classe), exist_ok=True)
#    inner_path_rgb = os.path.join(pathOut, classe)
#    inner_path_g = os.path.join(pathOutg, classe)
#    for lf in os.listdir(os.path.join(path, classe)):
#        lf_path = os.path.join(path, classe)
#        #img = (Image.open(lf_path))
#        print(lf)
#        multiview2lenslet(lf_path, pathOut+classe, pathOutg+classe, lf)

