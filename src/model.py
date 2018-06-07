# -*- coding: utf-8 -*-

import numpy as np
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def load_data():
    f = open('cloth.pickle', 'rb')
    face_data, face_label = pickle.load(f)

    img_rows, img_cols = 64, 64
    img_size = img_rows * img_cols

    x_train = np.empty((810, img_size))
    y_train = np.empty(810, dtype=int)

    x_test = np.empty((90, img_size))
    y_test = np.empty(90, dtype=int)

    for i in range(3):
        for j in range(90):
            x_train[i * 90 + j] = np.ndarray.flatten(
                face_data[i * 100 + j] ^ 255)
            y_train[i * 90 + j] = face_label[i * 100 + j]

        for j in range(10):
            x_test[i * 10 + j] = np.ndarray.flatten(
                face_data[i * 100 + 90 + j] ^ 255)
            y_test[i * 10 + j] = face_label[i * 100 + 90 + j]

    for i in range(3):
        for j in range(180):
            x_train[270 + i * 180 + j] = np.ndarray.flatten(
                face_data[300 + i * 200 + j] ^ 255)
            y_train[270 + i * 180 + j] = face_label[300 + i * 200 + j]

        for j in range(20):
            x_test[30 + i * 20 + j] = np.ndarray.flatten(
                face_data[300 + i * 200 + 180 + j] ^ 255)
            y_test[30 + i * 20 + j] = face_label[300 + i * 200 + 180 + j]

    return (x_train, y_train), (x_test, y_test)


def save_to_json(model):
    with open('model.json', 'w') as fp:
        json_string = model.to_json()
        fp.write(json_string)
    print('Write model.json successed!')


def main():
    epochs = 12

    batch_size = 16
    num_classes = 6
    img_rows, img_cols = 64, 64
    (x_train, y_train), (x_test, y_test) = load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(120, (5, 5), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9,
                               nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    save_to_json(model)
    model.save_weights('weights.h5')
    print('Save weights successed!')


if __name__ == '__main__':
    main()
