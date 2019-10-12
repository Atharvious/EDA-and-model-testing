# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:02:12 2019

@author: ATHARVA
"""

import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten, LeakyReLU
import matplotlib.pyplot as plt
from keras.utils import to_categorical


#  To train on the data yourself, change data_path variable to the directory containing the train and test csv files.

data_path = "E:\\Academics\\Projects\\FMNIST\\"
df_train = pd.read_csv(data_path + "fashion-mnist_train.csv")

df_x = df_train.drop(labels = 'label', axis = 1)
df_y = df_train['label']

x_train = df_x.to_numpy()
x_train = x_train.reshape([60000,28,28,1])

y = df_y.to_numpy()
y_train = to_categorical(y,num_classes = 10)

""" Test code to make sure that the data has been properly reshaped:

sample_image = x_train[:1,:,:,:]
sample_image = sample_image.reshape([28,28])

plt.imshow(sample_image, cmap='gray')
plt.show()


"""
learning_rate = 0.05
num_epochs = 25
batch_size = 64
num_classes = 10

model = Sequential()

model.add(Conv2D(filters = 2,kernel_size = (5,5),strides =(1,1), padding = 'valid',
                 input_shape = (28,28,1)))
model.add(LeakyReLU(alpha = 0.05))


model.add(Conv2D(filters = 4, kernel_size = (4,4), strides = (1,1),
                 padding = 'valid'))

model.add(LeakyReLU(alpha = 0.05))

model.add(Conv2D(filters = 4, kernel_size = (3,3),strides = (1,1),
                 padding = 'valid'))

model.add(Dropout(0.5))

model.add(LeakyReLU(alpha = 0.05))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(100))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(32))
model.add(LeakyReLU(alpha = 0.05))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.layers


model.compile(optimizer = 'adagrad',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(x_train,y_train, epochs = num_epochs, batch_size = batch_size)


model_json = model.to_json()
with open("network.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("weights.h5")
print("Saved model to disk")
