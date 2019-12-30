import os
import numpy as np
import cv2
from process import process_image

def process_dataset(in_folder, out_folder):
    all_images = os.listdir(in_folder)
    os.mkdir(out_folder)
    for image_name in all_images:
        image_data = cv2.imread(in_folder + '/' + image_name, 0)

        np.savetxt(out_folder + '/' + image_name.split('.')[0] + ".txt", process_image(image_data))
