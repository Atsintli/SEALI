# -*- coding: latin-1 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf #runing on tf 1.13.1
from tensorflow import keras
import glob
import numpy as np

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
        "periodico" : 3 
    } 
    return switcher.get(argument, "NOCLASS")

def parse_database():
  with open("trainset.csv") as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      clases.append(getClassID(className))
      floats = [float(i) for i in linea[0:-1]]
      trainset.append(floats)
  return trainset, clases

def parse_testset():
  with open("testset.csv") as archivo:
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

# Define a model
# Returns a short sequential model

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(380, activation=tf.keras.activations.relu, input_shape=(380, None)), #si le pongo None me manda error
    #keras.layers.Dropout(0.2),n_unique_labels,
    keras.layers.Dense(190, activation=tf.keras.activations.relu),
    keras.layers.Dense(3, activation=tf.keras.activations.softmax),
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

  return model

### As a `saved_model

### Caution: This method of saving a `tf.keras` model is experimental and may change in future versions.

### Build the model:

model = create_model()

model.fit(tr_features, tr_labels,  epochs = 10)

"""Create a saved_model:"""

import time

saved_model_path = "/home/atsintli/Desktop/Doctorado/Models/saved_models/v2/1"

#tf.contrib.saved_model.save_keras_model(model, saved_model_path)

tf.keras.experimental.export_saved_model(model, saved_model_path)

"""Reload a fresh keras model from the saved model."""

new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model.summary()

"""Run the restored model."""

# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.

new_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(ts_features, ts_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
