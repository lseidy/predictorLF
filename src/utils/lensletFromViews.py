import sys
import os
from pngsJoiner import read_images_in_order
from cut_center_LF import cut_center
from lensletRearrange import multiview2lenslet

dspath = "/home/machado/Downloads/lf_ds_8-64samples/images/"
for folder in os.listdir(dspath):
    lfpath = os.path.join(dspath, folder)
    read_images_in_order(lfpath, 15, 15)
    cut_center(lfpath, lfpath+"/"+folder+".png",os.path.join("/all.png"))
    multiview2lenslet(lfpath, dspath+"../Lenslet_RGB/", dspath+"../Lenslet_Gscale/", f"{folder}.png")