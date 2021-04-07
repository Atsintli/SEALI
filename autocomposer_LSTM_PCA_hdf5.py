#%%
import numpy as np
import matplotlib.pyplot as plt
from essentia.streaming import *
from scipy.special import softmax
from essentia import INFO
from feature_extract_ import extract_all_mfccs
from functools import reduce
from toolz import assoc
import toolz as tz
import json
import soundfile
from sklearn.decomposition import PCA   
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa
import keras as K
from keras.models import *
from keras.layers.core import *
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.layers import BatchNormalization as BatchNorm

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features(input_data):
    features = list(map(lambda data: 
               list(tz.concat(getProps(
                ['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'],
                   data))),
    input_data))
    np_mfcc = np.array(features)
    #print(np_mfcc.shape)
    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(np_mfcc.data) 
    pca = PCA(n_components=10, whiten=True)
    pca_result = pca.fit_transform(x_train) #investigar si es posible que me de que cosa le hizo, cuales son las filas que tomó, o trasnforma los datos, qué es lo que da? cuáles y el peso?
    #print(pca_result.shape)
    #pca_result = pca_result.tolist()
    return pca_result

def create_network(x,y): #y has to be the number of diferent classes in the array
    neurons = 128
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        neurons,
        input_shape=(5, 2),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(neurons, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(neurons))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(neurons))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(12))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights('Models/weights-music18_5seq.hdf5')
    return model

def reducer( mfcc, acc, fileData):
  diff = np.linalg.norm(np.array(fileData['PCA']) - np.array(mfcc))
  if acc==None:
    return tz.assoc(fileData, 'diff', diff)
  else: 
    if acc['diff'] <= diff:
        return acc
    else:
        return tz.assoc(fileData, 'diff', diff)

def getClosestCandidate(mfcc):
  with open('pca_music18.json') as f:
      jsonData = json.load(f)
  return reduce(lambda acc, fileData: reducer(mfcc, acc, fileData), jsonData, None)

def dedupe(tracklist):
    acc = []
    for el in tracklist:
        if len(acc) == 0:
            acc.append(el)
            continue;
        if acc[-1]["file"] != el["file"]:
            acc.append(el)
    return acc

def concatenateFiles(fileName):
    folder = 'Autocomposer/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    files = []

    for i in range(1):
        audiototal = np.array([])
        for elements in fileName:
            num = ('segments_music18/' + elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                soundfile.write(folder + "music18_2" + ".wav", audiototal, sr)

#%%
#def generate_music():    
input_data = extract_all_mfccs(sorted(glob.glob('segments_music18/' + "*.wav"))[0:100])
#print("soy mfccs", input_data)
mfccs = concat_features(input_data)
print("soy pca", mfccs)
mfcc_results = mfccs
print("deben ser pca",mfcc_results)
prediction_input = np.reshape(mfcc_results, (1, len(mfcc_results), 2))
print(prediction_input)
#%%
i=0
model = create_network(prediction_input, 2)

while i < 50:
    prediction = model.predict(prediction_input, verbose=0)
    prediction_ = np.append(mfcc_results, prediction_input[-5:]) #add data(prediction) to prediction
    i = i+1

result = list(map(getClosestCandidate,prediction_))[5:]

print (list(map(lambda x: x['file'], result)))
print(len(result))

print("result", result)
lastResult = dedupe(result)
lista = list(map(lambda x: x['file'], lastResult))
print("last list", lista)
concatenateFiles(lista)

#generate_music()
# %%
