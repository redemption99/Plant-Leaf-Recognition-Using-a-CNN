import pickle
from labels import get_flavia_labels, get_labels
import os
import numpy as np
import cv2
from process import process_image
import os.path
from os import path
from cnn import create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
labels = get_flavia_labels()

model_name = 'model.pkl'

if not path.exists(model_name):
    create_model(model_name)

model = pickle.load(open(model_name, 'rb'))

def print_predictions(folder):
    all_images = os.listdir(folder)

    all_processed_images = []

    for image_name in all_images:
        image_data = cv2.imread(folder + '/' + image_name, 0)

        processed_image = process_image(image_data)

        all_processed_images.append(processed_image.reshape(processed_image.shape[0], processed_image.shape[1], 1))

    all_processed_images = np.asarray(all_processed_images)

    vec = model.predict_classes(all_processed_images)

    for image_name, idx in zip(all_images, vec):
        print(image_name, " -> ", labels[idx][1])


print_predictions('to_predict')
