import os
import numpy as np
import cv2
from process import process_image

def process_dataset(in_folder, out_folder):
    #uzimamo listu fajlova iz foldera u kome se nalazi dataset
    all_images = os.listdir(in_folder)
    #kreiranje izlaznog foldera
    os.mkdir(out_folder)
    for image_name in all_images:
        #otvaramo sliku pomocu njene putanje kao grayscale
        image_data = cv2.imread(in_folder + '/' + image_name, 0)

        # cuvamo sliku kao matricu nula i jedinica u txt fajl
        np.savetxt(out_folder + '/' + image_name.split('.')[0] + ".txt", process_image(image_data))
