# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:49:41 2019

@author: ATHARVA
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten, LeakyReLU
import matplotlib.pyplot as plt
from keras.utils import to_categorical

num_classes = 7
root = "path\\to\\folder\\"
folder_path = root + "model\\"

keras.backend.clear_session()

relu = LeakyReLU(alpha = 0.3)
model = Sequential([
        Dense(128, activation = 'relu', input_shape = [20]),
        Dense(256, activation = 'relu'),
        Dense(512, activation = 'relu'),
        Dense(256, activation = 'relu'),
        Dropout(0.3),
        Dense(512, activation = 'relu'),
        Dense(1024, activation ='relu'),
        Dropout(0.2),
        Dense(num_classes, activation = 'softmax'),
        ])

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model_json = model.to_json()
with open(folder_path + "network.json", "w") as json_file:
    json_file.write(model_json)
