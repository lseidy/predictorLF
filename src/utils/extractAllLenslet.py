import os
import multiviewExtractor as extractor
import sys
import threading
import time


def extract_all(classe):

    lfr_path = "/home/machado/Lytro2.0_Inria_sirocco/LytroIllum_Dataset_INRIA_SIROCCO/Data"
    path = "/home/machado/Lytro2.0_Inria_sirocco/Lytro2.0_dataset_INRIA_SIROCCO/"
    # classe = "Buildings"

    save_lensletGscale_path = os.path.join(path, "Lenslet_Gscale", classe)
    save_mv_path = os.path.join(path, "MultiView_RGB", classe)
    save_lensletRGB_path = os.path.join(path, "Lenslet_RGB", classe)

    # for folder in os.listdir(lfr_path):
    inner_path = os.path.join(lfr_path)
    os.makedirs(save_lensletGscale_path, exist_ok=True)
    os.makedirs(save_mv_path, exist_ok=True)
    os.makedirs(save_lensletRGB_path, exist_ok=True)
    os.makedirs(save_lensletRGB_path, exist_ok=True)
    for lf in os.listdir(inner_path):
        print(lf)
        if lf.split(".")[0] + ".png" not in os.listdir(save_lensletRGB_path) and len(lf.split(".")) == 2:
            try:
                extractor.extract_lenslet(inner_path, save_lensletGscale_path, save_mv_path, save_lensletRGB_path, lf)
                os.replace(os.path.join(lfr_path,lf), os.path.join("/home/machado/Lytro2.0_Inria_sirocco/LytroIllum_Dataset_INRIA_SIROCCO/", 'Done/',lf))
            except IndexError:
                os.replace(os.path.join(lfr_path, lf), os.path.join("/home/machado/Lytro2.0_Inria_sirocco/LytroIllum_Dataset_INRIA_SIROCCO/", 'Broken/', lf))
    print("finished ", classe)


extract_all(sys.argv[1])

