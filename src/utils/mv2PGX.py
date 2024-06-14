import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image, write_video
from torchvision.utils import save_image
import numpy as np
import cv2
from PIL import Image

def mv2pgx(images, output_file):
    #output_file = output_file.replace("_", "-")
    #with open(output_file, 'wb') as yuv_file:
        #for image_path in images:
       
    print(images, output_file)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    #img = read_image(images)
    img = Image.open(images)

    img = transform(img)
    _, height, width = img.shape[:]
    width=int(width/16)
    height=int(height/16)
    print(width, height)

    
    
    for i in range(16):
        for j in (reversed(range(16)) if i%2 != 0 else  range(16)): 
            view = img[:, j*height:j*height+height,i*width:i*width+width]
            #print(frame)
            #frame = frame * 255 # Convert to 0-255 scale
            # frame = frame.byte()
            #yuv_img = cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2YUV)
            
            
            # Write Y, U, and V planes to the file
            #y, u, v = cv2.split(yuv_img)
            #yuv_file.write(yuv_img)
            

            view = torch.permute(view, (1,2,0))
            outputname = f"{output_file}/0{j}_0{j}.ppm"
            print(outputname)
            save_image(view, outputname, format="ppm")


        
            
                #img = transform(img) * 255  # Convert to 0-255 scale
            #frame = frame.byte()
            
            # Convert the image to YUV format
            #yuv_img = frame.permute(1, 2, 0).numpy()
            #
            ## Append the YUV frame
            #yuv_frames.append(torch.from_numpy(yuv_img))
    #print(torch.stack(yuv_frames).shape)
    #write_video(output_file, torch.stack(yuv_frames), fps=30)
            #print(i, j)
            #
            #video[numFrame] = frame
            #numFrame +=1
            #print(numFrame)
            #video[numFrame, 0] =frame[0, :, :] * .299 + frame[1, :, :] * .587 + frame[ 2, :, :] * .114 
            #video[numFrame, 1] =frame[2, :, :]-video[numFrame, 0, :, :] * .564 + .5    
            #video[numFrame, 2] =frame[0, :, :]-video[numFrame, 0, :, :] * .713 + .5    


    #video = torch.permute(video, (0, 2,3,1))
    #print(video.shape)
    #write_video(output_file, video, fps=30)



    
    # Resize the image to the desired width and height


path = "/home/machado/New_Extracted_Dataset/Multiview_16x16/Mirrors_and_Transparency/Vespa.png"
out_path= "/home/machado/New_Extracted_Dataset/PPMs_16x16/Vespa/"
try:
     os.system(f"mkdir {out_path}")
except:
    pass
mv2pgx(path,out_path)
#having problem with the names so far
#for classe in os.listdir(path):
#    try:
#        os.system(f"mkdir {out_path}/{classe}")
#    except:
#        pass
#    class_path = os.path.join(path, classe)
#    for lf in os.listdir(class_path):
#        images_to_yuv(os.path.join(class_path,lf),os.path.join(out_path, classe,lf.split(".")[0]))