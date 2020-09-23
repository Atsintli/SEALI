# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf #with this code have trouble in placeholder section
from tensorflow import keras
import glob
import numpy as np
import shutil
from sklearn.utils import shuffle

trainset = []
clases = []
testset = []
testclases = []

def parse_database():
  with open("anotatedDataBase.csv") as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      clases.append(int(className))
      floats = [float(i) for i in linea[0:-1]]
      trainset.append(floats)
  return trainset, clases 

def parse_testset():
  with open("anotatedDataBase.csv") as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      testclases.append(int(className))
      floats = [float(i) for i in linea[0:-1]]
      testset.append(floats)
  return testset, testclases

parse_database()
parse_testset()

tr_features = np.array(trainset)
tr_labels = np.array(clases)
tr_features, tr_labels = shuffle(tr_features, tr_labels)

ts_features = (np.array(testset))
ts_labels = np.array(testclases)
ts_features, ts_labels = shuffle(ts_features, ts_labels)


### Define a model
ins=14
outs=6
ins2=64

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(ins2, activation=tf.nn.relu, input_shape=(ins,)),
    keras.layers.Dense(ins2, activation=tf.nn.relu),
    # keras.layers.Dropout(0.01),
    # keras.layers.Dense(ins2, activation=tf.nn.relu),
    keras.layers.Dense(outs, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model

# Create a basic model instance
model = create_model()
model.summary()

#checkpoint_path = "training_1/cp.ckpt"
checkpoint_path = "Models/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()
model.fit(tr_features, tr_labels,  
          batch_size=5,
          epochs = 10, 
          validation_data = (ts_features, ts_labels),
          callbacks = [cp_callback])  # pass callback to training

dir = 'Models/saved_models/Large_Essentia_MFCCs/1/'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

saved_model_path = dir
tf.saved_model.save(model, saved_model_path)

"""Reload a fresh keras model from the saved model."""

new_model = tf.keras.models.load_model(saved_model_path)
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