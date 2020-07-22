# -*- coding: latin-1 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf #runing on tf 1.13.1
import glob
import numpy as np
from tensorflow.keras import layers, models
import time
import os
import shutil
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.utils import to_categorical

print (tf.__version__)

trainset = []
clases = []
testset = []
testclases = []

def getClassID(argument): 
    switcher = { 
        "caotico" : 0, 
        "complejo" : 1, 
        "fijo": 2,
        "periodico" : 3,
    } 
    return switcher.get(argument, "NOCLASS")

def parse_database():
  with open("dataset_82_SpecFlatness.csv") as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      clases.append(getClassID(className))
      floats = [float(i) for i in linea[0:-1]]
      trainset.append(floats)
  return trainset, clases

def parse_testset():
  with open("testset_82_SpecFlatness.csv") as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      testclases.append(getClassID(className))
      floats = [float(i) for i in linea[0:-1]]
      testset.append(floats)
  return testset, testclases

parse_database()
parse_testset()

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode = (np.arange(n_unique_labels) == labels[:,None]).astype(np.float32)
    return one_hot_encode

tr_features = np.array(trainset)
tr_labels = np.array(clases)
ts_features = np.array(testset)
ts_labels = np.array(testclases)

## reshape for CNN
print (tr_features.shape)
tr_features = tr_features.reshape(tr_features.shape[0], tr_features.shape[1], 1)
tr_features = tr_features.reshape(tr_features.shape[0], tr_features.shape[1], tr_features.shape[2], 1)
print (tr_features.shape)
ts_features = ts_features.reshape(ts_features.shape[0], ts_features.shape[1], 1)
ts_features = ts_features.reshape(ts_features.shape[0], ts_features.shape[1], ts_features.shape[2], 1)
print (tr_labels)
# Define a model
# Returns a short sequential model

ins = 82
outs = 4


tr_labels = to_categorical(tr_labels)
ts_labels = to_categorical(ts_labels)


def create_model():
  model = Sequential()
  model.add(Conv2D(ins, (1, 1), padding='same', input_shape=(ins,1,1)))
  model.add(Activation('relu'))
  # model.add(Conv2D(ins, (1, 1)))
  # model.add(Activation('relu'))
  # model.add(MaxPooling2D(pool_size=(1,1)))
  # model.add(Dropout(0.25))
  # model.add(Conv2D(ins, (1, 1), padding='same'))
  # model.add(Activation('relu'))
  # model.add(Conv2D(ins, (1, 1)))
  # model.add(Activation('relu'))
  # model.add(MaxPooling2D(pool_size=(1,1)))
  # model.add(Dropout(0.5))
  # model.add(Conv2D(ins, (1,1), padding='same'))
  # model.add(Activation('relu'))
  # model.add(Conv2D(ins*2, (1, 1)))
  # model.add(Activation('relu'))
  # model.add(MaxPooling2D(pool_size=(1,1)))
  # model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(ins))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(outs, activation='softmax'))
  model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
  return model

### As a `saved_model

### Caution: This method of saving a `tf.keras` model is experimental and may change in future versions.

### Build the model:

model = create_model()

model.fit(tr_features, tr_labels,  
          #validation_split = 0.20, 
          batch_size=12, 
          epochs = 1, 
          shuffle=True, 
          verbose=2)

"""Create a saved_model:"""

dir = '/home/atsintli/Desktop/Doctorado/Models/saved_models/v6CNN/1.h5'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

saved_model_path = dir

#saved_model_path = "/home/modelos/saved_models/v9/1/" #windows

#saved_model_path = "/home/atsintli/Desktop/Doctorado/Models/saved_models/v6/1" #linux

tf.keras.experimental.export_saved_model(model, saved_model_path)

#tf.contrib.saved_model.save_keras_model(model, saved_model_path)

#tf.keras.experimental.export_saved_model(model, saved_model_path)

# model.save(dir)


"""Reload a fresh keras model from the saved model."""

new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
model.summary()

"""Run the restored model."""

# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.

new_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(ts_features, ts_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))