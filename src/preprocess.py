#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import pickle

img = Image.open('../faces.gif')

img_ndarray = np.asarray(img, dtype=int)
img_ndarray ^= 255

img_rows, img_cols = 57, 47
face_data = np.empty((400, img_rows * img_cols))

for row in range(20):
    for col in range(20):
        arr = img_ndarray[
            row * img_rows:(row + 1) * img_rows,
            col * img_cols:(col + 1) * img_cols,
        ]
        face_data[row * 20 + col] = np.ndarray.flatten(arr)

        # im = Image.new("L", (img_rows, img_cols))
        # for i in range(img_rows):
        #     for j in range(img_cols):
        #         im.putpixel((i, j), (arr[i, j]))
        # im.save('../face%d.gif' % (row * 20 + col))


face_label = np.empty(400, dtype=int)
for i in range(400):
    face_label[i] = i / 10


f = open('faces.pkl', 'wb')
pickle.dump((face_data, face_label), f)
f.close()
