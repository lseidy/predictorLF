from skimage.metrics import peak_signal_noise_ratio
from skimage import io

def calculate_psnr(image_path1, image_path2):
    # Read the images
    imgOrig = io.imread(image_path1)
    imgPred = io.imread(image_path2)


    #crop the black blocks and re-align, take out the 3 colors
    orig_croped= imgOrig[32:, 32:4976-16]
    pred_cropped= imgPred[:-32, :-32, 0]

    # print(orig_croped)
    # print(pred_cropped)


    # io.imsave("orig_croped.png", orig_croped )
    # io.imsave("pred_cropped.png", pred_cropped)
    # Calculate PSNR
    psnr_value = peak_signal_noise_ratio(orig_croped, pred_cropped)

    return psnr_value

# Example usage
image_path1 = "/home/idm/New_Extracted_Dataset/Lenslet_8x8_Gscale/Urban/Bikes.png"
image_path2 = ('/home/idm/reconstructions/newKeras_skip/allBlocks_1_15.png')

psnr_result = calculate_psnr(image_path1, image_path2)
print(f"PSNR: {psnr_result} dB")

#wrong connections:
#noSkip 31.806368798705066 dB
#withSKip 31.96135308885641 dB

# New implemented algorithm
# newKeras with Skip:   31.75680033586282 dB
# newKeras noSkip:      31.195165847555295 dB

