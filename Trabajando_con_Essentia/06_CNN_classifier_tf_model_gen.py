# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf #with this code have trouble in placeholder section
from tensorflow import keras
import glob
import numpy as np
import shutil

trainset = []
clases = []
testset = []
testclases = []

def parse_database():
  with open("anotatedMFCCsAsStrings.csv") as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      clases.append(className)
      floats = [float(i) for i in linea[0:-1]]
      trainset.append(floats)
  return trainset, clases 

def parse_testset():
  with open("anotatedMFCCsAsStrings.csv") as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      testclases.append(className)
      floats = [float(i) for i in linea[0:-1]]
      testset.append(floats)
  return testset, testclases

parse_database()
parse_testset()

tr_features = np.array(trainset)
tr_labels = np.array(clases)
ts_features = np.array(testset)
ts_labels = np.array(testclases)

### Define a model

ins=13
outs=14
ins2 = 120

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(ins2, activation=tf.nn.relu, input_shape=(ins,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(ins2, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(ins2, activation=tf.nn.relu),   
    keras.layers.Dropout(0.2),
    keras.layers.Dense(ins2, activation=tf.nn.relu),   
    keras.layers.Dropout(0.2),
    keras.layers.Dense(outs, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model

# Create a basic model instance
model = create_model()
model.summary()

#checkpoint_path = "training_1/cp.ckpt"
checkpoint_path = "/home/atsintli/Desktop/Doctorado/Models/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(tr_features, tr_labels,  
          batch_size=5,
          epochs = 500, 
          validation_data = (ts_features,ts_labels),
          callbacks = [cp_callback])  # pass callback to training

dir = '/home/atsintli/Desktop/Doctorado/Models/saved_models/Essentia_MFCCs_Loudness/1/'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

saved_model_path = dir
tf.contrib.saved_model.save_keras_model(model, saved_model_path)

"""Reload a fresh keras model from the saved model."""

new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model.summary()

"""Run the restored model."""

# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.

new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(ts_features, ts_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))