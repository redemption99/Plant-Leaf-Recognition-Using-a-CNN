import numpy as np
import cv2

def process_image(image_data):
    # image dimensions
    width, height = image_data.shape

    # creating new squared image, filled with ones (white color), which represent the initial image exceeded by neutral pixels to the square
    padded_image = np.ones((max(width, height), max(width, height)))

    # finding upper left corner of centered image inside the padded image
    center_x = (max(width, height) - width) // 2
    center_y = (max(width, height) - height) // 2

    # writing the initial image inside the middle part of padded image
    padded_image[center_x: center_x + width, center_y: center_y + height] = image_data / 255

    # resizing image to 229x229 pixels
    scaled_image = cv2.resize(padded_image, (229, 229))

    # average pooling
    kernel = np.ones((3, 3), np.float32) / 9
    averaged_image = cv2.filter2D(scaled_image, -1, kernel)

    # setting leaf pixels to 1, non leaf to 0. ~0.95 is the average value of pixels
    rounded_image = (averaged_image < 0.95).astype(int) * 1.0

    # Laplacian filter for contour detection
    edges_image = cv2.Laplacian(rounded_image, cv2.CV_64F)

    # setting pixels on the contour of leaf to 1, other pixels to 0
    edges_image = (edges_image > 0.5).astype(int) * 1.0

    # returning processed image
    return edges_image
