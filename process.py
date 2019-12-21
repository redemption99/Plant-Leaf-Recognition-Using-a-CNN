import numpy as np
import cv2

def process_image(image_data):
    # dimenzije slike
    width, height = image_data.shape

    # kreiramo novu kvadratnu sliku, popunjenu jedinicama (bela boja), koja predstavlja pocetnu sliku prosirenu neutralnim pikselima do kvadrata
    padded_image = np.ones((max(width, height), max(width, height)))

    # nadjemo gornji levi cosak centrirane slike unutar prosirene slike
    center_x = (max(width, height) - width) // 2
    center_y = (max(width, height) - height) // 2

    # prepisemo pocetnu sliku u centralni deo nove slike
    padded_image[center_x: center_x + width, center_y: center_y + height] = image_data / 255

    # smanjimo sliku na velicinu od 200x200 piksela
    scaled_image = cv2.resize(padded_image, (229, 229))

    # svaki piksel postaje srednja vrednost njega i okolnih 8 piksela kako bi se eliminisao sum
    kernel = np.ones((3, 3), np.float32) / 9
    averaged_image = cv2.filter2D(scaled_image, -1, kernel)

    # svaki piksel postaje 0 ili 1, za granicu uzimamo 0.95 posto je to srednja vrednost piksela
    rounded_image = (averaged_image < 0.95).astype(int) * 1.0

    # primenjujemo Laplasov filter za detektovanje svih ivica
    edges_image = cv2.Laplacian(rounded_image, cv2.CV_64F)

    # opet zaokruzujemo sliku na nule i jedinice
    edges_image = (edges_image > 0.5).astype(int) * 1.0

    # cuvamo sliku kao matricu nula i jedinica u txt fajl
    return edges_image
