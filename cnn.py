import numpy as np
import os
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pickle
from preprocessing import process_dataset
from labels import get_flavia_labels, get_labels

labels = get_flavia_labels()
num_classes = len(labels)

#load data
def get_data():

    processed_folder = 'processed_flavia'
    if not os.path.exists(processed_folder):
        print("obrada dataseta")
        process_dataset('dataset_flavia', processed_folder)

    all_images = os.listdir(processed_folder)

    X = []
    Y = []

    #load data and labels

    for image_name in all_images:
        image_data = np.loadtxt(processed_folder + '/' + image_name)

        X.append(image_data.reshape((image_data.shape[0], image_data.shape[1], 1)))
        Y.append(np.zeros(num_classes))
        Y[-1][0] = 1

        imageid = int(image_name.split('.')[0])
        for j in labels:
            if j[2] <= imageid <= j[3]:
                Y[-1][0] = 0
                Y[-1][j[0]] = 1
                break
        ''' 
        splitted_name = image_name.split('_')
        label_name = splitted_name[0]

        if len(Y) > 1 and label_name == labels[np.argmax(Y[-2][1])]:
            Y[-1] = Y[-2]
        else:
            for j in labels:
                if label_name == j[1]:
                    Y[-1][0] = 0
                    Y[-1][j[0]] = 1
        '''

    return np.asarray(X), np.asarray(Y)

def create_model(model_name):

    print("ucitavanje dataseta")

    #ucitavanje podataka
    [X, Y] = get_data()

    print("zavrseno ucitavanje")

    batch_size = 32
    epochs = 50

    #80% dataseta koristimo za treniranje, 20% za testiranje
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    #kreiranje modela
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides=2, padding='valid', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=1))
    model.add(Conv2D(32, (3, 3), strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), strides=2, padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=1))

    model.add(Flatten())
    model.add(Dense(2048))  # Prvi potpuno povezan sloj
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(512))  # Drugi potpuno povezan sloj
    model.add(Activation('relu'))
    model.add(Dense(num_classes))  # Finalni potpuno povezan sloj
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Kompilacija modela
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #treniranje modela
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), shuffle=True)

    scores = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    pickle.dump(model, open(model_name, 'wb'))
