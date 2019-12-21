import csv

def get_labels():
    # get labels
    with open('labels.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        labels = []
        labels.append([0, "Undefined", 0, 0])

        for row in readCSV:
            labels.append([int(row[0]), row[1], int(row[2]), int(row[3])])
    return labels
