from skimage.metrics import peak_signal_noise_ratio
from skimage import io

def calculate_psnr(image_path1, image_path2):
    # Read the images
    imgOrig = io.imread(image_path1)
    imgPred = io.imread(image_path2)

    print(imgOrig.shape)
    print(imgPred.shape)


    #crop the black blocks and re-align, take out the 3 colors
    orig_croped= imgOrig[32:, 32:4976-16]
    pred_cropped= imgPred[:-32, :-32, 0]

    print(orig_croped.shape)
    print(pred_cropped.shape)


    io.imsave("orig_croped.png", orig_croped )
    io.imsave("pred_cropped.png", pred_cropped)
    # Calculate PSNR
    psnr_value = peak_signal_noise_ratio(orig_croped, pred_cropped)

    return psnr_value

# Example usage
image_path1 = "/home/idm/New_Extracted_Dataset/Lenslet_8x8_Gscale/Urban/Bikes.png"
image_path2 = '/home/idm/inverted_ij/00allBlocks_33.png'

psnr_result = calculate_psnr(image_path1, image_path2)
print(f"PSNR: {psnr_result} dB")
