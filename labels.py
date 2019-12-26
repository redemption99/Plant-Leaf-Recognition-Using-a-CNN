import os
import csv

def get_flavia_labels():
    # get labels
    with open('labels.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        labels = []
        labels.append([0, "Undefined", 0, 0])

        for row in readCSV:
            labels.append([int(row[0]), row[1], int(row[2]), int(row[3])])
    return labels

def get_labels():
    # get labels

    all_images = os.listdir('dataset')

    labels = []
    labels.append([0, "Undefined"])

    for image_name in all_images:
        splitted_name = image_name.split('_')
        label_name = splitted_name[0]
        if not label_name == labels[-1][1]:
            labels.append([len(labels), label_name])

    return labels
