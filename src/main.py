# -*- coding: utf-8 -*-

import numpy as np
import cPickle as pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def faces_load_data():
    f = open('faces.pkl', 'rb')
    face_data, face_label = pickle.load(f)

    img_rows, img_cols = 57, 47
    img_size = img_rows * img_cols

    x_train = np.empty((360, img_size))
    y_train = np.empty(360, dtype=int)

    x_test = np.empty((40, img_size))
    y_test = np.empty(40, dtype=int)

    for i in range(40):
        x_train[i * 9:(i + 1) * 9, :] = face_data[i * 10:i * 10 + 9, :]
        y_train[i * 9:(i + 1) * 9] = face_label[i * 10:i * 10 + 9]

        x_test[i] = face_data[i * 10 + 9, :]
        y_test[i] = face_label[i * 10 + 9]

    return (x_train, y_train), (x_test, y_test)

def mnist_load_data():
    def get_file_content(path):
        with open(path, 'rb') as f:
            content = f.read()
        return content


    def get_picture(content, size):
        data_set = []
        for index in range(size):
            start = index * 28 * 28 + 16
            picture = []
            for i in range(28 * 28):
                picture.append(ord(content[start + i]))
            data_set.append(picture)
        return np.array(data_set)

    def get_labels(content, size):
        labels = []
        for index in range(size):
            labels.append(ord(content[index + 8]))
        return np.array(labels)


    train_size = 60000
    test_size = 10000
    x_train = get_picture(
        get_file_content('../data/train-images-idx3-ubyte'),
        train_size
    )
    y_train = get_labels(
        get_file_content('../data/train-labels-idx1-ubyte'),
        train_size
    )
    x_test = get_picture(
        get_file_content('../data/t10k-images-idx3-ubyte'),
        test_size
    )
    y_test = get_labels(
        get_file_content('../data/t10k-labels-idx1-ubyte'),
        test_size
    )

    return (x_train, y_train), (x_test, y_test)


def main():
    epochs = 12

    # batch_size = 16
    # num_classes = 40
    # img_rows, img_cols = 57, 47
    # (x_train, y_train), (x_test, y_test) = faces_load_data()

    batch_size = 128
    num_classes = 10
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist_load_data()

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
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer= sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
