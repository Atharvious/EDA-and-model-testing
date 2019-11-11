# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:07:15 2019

@author: ATHARVA
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten, LeakyReLU
import matplotlib.pyplot as plt

root = "path\\to\\folder\\"

class_data = pd.read_csv(root + "class.csv")
zoo_data = pd.read_csv(root + "zoo.csv")

labels = zoo_data['class_type']

features = zoo_data.drop(labels = 'class_type', axis = 1)

features.head()

legs = features['legs']

cla = legs.unique()

changes = {'0':0,
           '2':1,
           '4':2,
           '5':3,
           '6':4,
           '8':5}
legs_classed = legs.apply(lambda x: changes[str(x)])

legs_one_hot = to_categorical(legs_classed, num_classes = 6)

print(legs_one_hot.shape)

legs_one_hot = legs_one_hot.astype(int)
for x in range(5):
    features['legs_'+str(x+1)] = legs_one_hot[:,x]

features_processed = features.drop(labels = 'legs', axis = 1)

print(labels.unique())

labels = labels.apply(lambda x: x-1)

y_processed = to_categorical(labels, num_classes = 7)

features_processed = features_processed.drop(labels = 'animal_name', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(features_processed,y_processed,test_size = 0.2, random_state = 17)

json_file = open(root + "model\\network.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.fit(x = x_train,y = y_train,epochs = 100)
met = model.evaluate(x_test,y_test)

accuracy = met[1]
print("Accuracy = "+ str(accuracy*100))

os.chdir(root)
model.save_weights(root+"model\\weights.h5")
