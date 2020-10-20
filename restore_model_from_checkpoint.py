import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras

trainset = []
clases = []
testset = []
testclases = []

def parse_database():
  with open("database_as_matrix_mfcc.csv") as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(',')
      className = linea[-1]
      clases.append(int(className))
      floats = [float(i) for i in linea[0:-1]]
      trainset.append(floats)
  return trainset, clases 

def parse_testset():
  with open("testset_small.csv") as archivo:
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

ins = 13
ins2 = 16
outs = 7


# def create_model():
#   model = Sequential()
#   model.add(Dense(ins, (1, 1), padding='same', input_shape=(ins,1,1)))
#   model.add(Activation('relu'))
#   model.add(Flatten())
#   model.add(Dense(ins))
#   model.add(Activation('relu'))
#   model.add(Dropout(0.5))
#   model.add(Dense(outs, activation='softmax'))
#   model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
#   return model

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(ins2, activation=tf.nn.relu, input_shape=(ins,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(ins2, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(ins2, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(outs, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model

def train_step(net, example, optimizer):
  """Trains `net` on `example` using `optimizer`."""
  with tf.GradientTape() as tape:
    output = net(example['x'])
    loss = tf.reduce_mean(tf.abs(output - example['y']))
  variables = net.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss

def train_and_checkpoint(net, manager):
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  for _ in range(50):
    example = next(iterator)
    loss = train_step(net, example, opt)
    ckpt.step.assign_add(1)
    if int(ckpt.step) % 10 == 0:
      save_path = manager.save()
      print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
      print("loss {:1.2f}".format(loss.numpy()))

def toy_dataset():
  inputs = tr_features
  labels = tr_labels
  return tf.data.Dataset.from_tensor_slices(
    dict(x=inputs, y=labels)).repeat().batch(2)

opt = tf.keras.optimizers.Adam(0.1)
model = create_model()
dataset = toy_dataset()
iterator = iter(dataset)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model, iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, './Models/tf_ckpts', max_to_keep=3)

train_and_checkpoint(model, manager)

print(manager.checkpoints)  # List the three remaining checkpoints