from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA   
from sklearn import datasets
import numpy as np
import tensorflow as tf #with this code have trouble in placeholder section
from tensorflow import keras
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets


#load the data
trainset = []
clases = []
testset = []
testclases = []

def getClassID(argument): 
    switcher = { 
        "caotico" : 0,
        "complejo" : 1,  
        "fijo" : 2,
        "periodico" : 3,
    } 
    return switcher.get(argument, "NOCLASS")

def parse_database(fileName):
  with open(fileName) as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      clases.append(getClassID(className))
      floats = [float(i) for i in linea[0:-1]]
      trainset.append(floats)
  return trainset, clases

def parse_testset(fileName):
  with open(fileName) as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      testclases.append(getClassID(className))
      floats = [float(i) for i in linea[0:-1]]
      testset.append(floats)
  return testset, testclases

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode = (np.arange(n_unique_labels) == labels[:,None]).astype(np.float32)
    return one_hot_encode

parse_database("dataset.csv")
parse_testset("testset.csv")

tr_features = np.array(trainset)
tr_labels = np.array(clases)
ts_features = np.array(testset)
ts_labels = np.array(testclases)

# Create and run an LDA, then use it to transform the features
lda = LinearDiscriminantAnalysis(n_components=19)
tr_features_pca = lda.fit(tr_features, tr_labels).transform(tr_features)

lda2 = LinearDiscriminantAnalysis(n_components=19)
ts_features_pca = lda.fit(ts_features, ts_labels).transform(ts_features)
# Print the number of features
print("Original number of features:", tr_features_pca.shape[1])
print("Reduced number of features:", ts_features_pca.shape[1])

file = open("dataset_LDA.csv", "w")
np.savetxt(file, (tr_features_pca))
file.close()
file = open("testset_LDA.csv", "w")
np.savetxt(file, (ts_features_pca))
file.close()

"""### Define a model
Let's build a simple model we'll use to demonstrate saving and loading weights.
"""
ins = tr_features_pca.shape[1]
outs = 4

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu, input_shape=(ins,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation=tf.nn.relu),   
    keras.layers.Dropout(0.2),
    keras.layers.Dense(outs, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model

# Create a basic model instance
model = create_model()
model.summary()

"""## Save checkpoints during training

The primary use case is to automatically save checkpoints *during* and at *the end* of training. This way you can use a trained model without having to retrain it, or pick-up training where you left of—in case the training process was interrupted.

`tf.keras.callbacks.ModelCheckpoint` is a callback that performs this task. The callback takes a couple of arguments to configure checkpointing.

### Checkpoint callback usage

Train the model and pass it the `ModelCheckpoint` callback:
"""

#checkpoint_path = "training_1/cp.ckpt"
checkpoint_path = "/home/atsintli/Desktop/Doctorado/Models/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()
epoch = 50

model.fit(tr_features_pca, tr_labels,  epochs=epoch, 
          validation_data = (ts_features_pca,ts_labels),
          callbacks = [cp_callback])  # pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

"""This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:"""

checkpoint_dir

"""Create a new, untrained model. When restoring a model from only weights, you must have a model with the same architecture as the original model. Since it's the same model architecture, we can share weights despite that it's a different *instance* of the model.

Now rebuild a fresh, untrained model, and evaluate it on the test set. An untrained model will perform at chance levels (~10% accuracy):
"""

model = create_model()

loss, acc = model.evaluate(ts_features_pca, ts_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

"""Then load the weights from the checkpoint, and re-evaluate:"""
model.load_weights(checkpoint_path)

loss,acc = model.evaluate(ts_features_pca, ts_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""### Checkpoint callback options

The callback provides several options to give the resulting checkpoints unique names, and adjust the checkpointing frequency.

Train a new model, and save uniquely named checkpoints once every 5-epochs:
"""

# include the epoch in the file name. (uses `str.format`)
#checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_path = "/home/atsintli/Desktop/Doctorado/Models/training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=500)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(tr_features_pca, tr_labels,
          epochs = epoch, callbacks = [cp_callback],
          validation_data = (ts_features_pca,ts_labels),
          verbose=1)

"""Now, look at the resulting checkpoints and choose the latest one:"""

#! ls {checkpoint_dir}

latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

"""Note: the default tensorflow format only saves the 5 most recent checkpoints.

To test, reset the model and load the latest checkpoint:
"""

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(ts_features_pca, ts_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""## What are these files?

The above code stores the weights to a collection of [checkpoint](https://www.tensorflow.org/guide/saved_model#save_and_restore_variables)-formatted files that contain only the trained weights in a binary format. Checkpoints contain:
* One or more shards that contain your model's weights.
* An index file that indicates which weights are stored in a which shard.

If you are only training a model on a single machine, you'll have one shard with the suffix: `.data-00000-of-00001`

## Manually save weights

Above you saw how to load the weights into a model.

Manually saving the weights is just as simple, use the `Model.save_weights` method.
"""

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(ts_features_pca, ts_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""## Save the entire model

The entire model can be saved to a file that contains the weight values, the model's configuration, and even the optimizer's configuration (depends on set up). This allows you to checkpoint a model and resume training later—from the exact same state—without access to the original code.

Saving a fully-functional model is very useful—you can load them in TensorFlow.js ([HDF5](https://js.tensorflow.org/tutorials/import-keras.html), [Saved Model](https://js.tensorflow.org/tutorials/import-saved-model.html)) and then train and run them in web browsers, or convert them to run on mobile devices using TensorFlow Lite ([HDF5](https://www.tensorflow.org/lite/convert/python_api#exporting_a_tfkeras_file_), [Saved Model](https://www.tensorflow.org/lite/convert/python_api#exporting_a_savedmodel_))

### As an HDF5 file

Keras provides a basic save format using the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) standard. For our purposes, the saved model can be treated as a single binary blob.
"""

model = create_model()

model.fit(tr_features_pca, tr_labels, epochs=epoch)

# Save entire model to a HDF5 file
model.save('my_model.h5')

"""Now recreate the model from that file:"""

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

"""Check its accuracy:"""

loss, acc = new_model.evaluate(ts_features_pca, ts_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""This technique saves everything:

* The weight values
* The model's configuration(architecture)
* The optimizer configuration

Keras saves models by inspecting the architecture. Currently, it is not able to save TensorFlow optimizers (from `tf.train`). When using those you will need to re-compile the model after loading, and you will lose the state of the optimizer.

### As a `saved_model`

Caution: This method of saving a `tf.keras` model is experimental and may change in future versions.

Build a fresh model:
"""

model = create_model()

model.fit(tr_features_pca, tr_labels, epochs=epoch)

"""Create a `saved_model`:"""

import shutil

dir = '/home/atsintli/Desktop/Doctorado/Models/saved_models/2/1/'
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

new_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(ts_features_pca, ts_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""## What's Next

That was a quick guide to saving and loading in with `tf.keras`.

* The [tf.keras guide](https://www.tensorflow.org/guide/keras) shows more about saving and loading models with `tf.keras`.

* See [Saving in eager](https://www.tensorflow.org/guide/eager#object_based_saving) for saving during eager execution.

* The [Save and Restore](https://www.tensorflow.org/guide/saved_model) guide has low-level details about TensorFlow saving.
"""



