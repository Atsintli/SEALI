#%%
import json
import numpy as np
import glob
import csv
from numpy import loadtxt
import os
from utils import get_json, save_as_json, save_matrix_array
from utils import save_descriptors_as_matrix
import toolz as tz
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

file_in = "dataset_label_test.csv"
#file_in = "sink_into_return.csv"
json_file = "audiotestsplits.json"

features = loadtxt(file_in)

def pca(features):
    #Standardize the feature matrix
    std_features = StandardScaler().fit_transform(features.data)
    #Create a PCA that will retain 99% of variace
    pca = PCA(n_components=0.2, whiten=True) #svd_solver="randomized"
    features_pca = pca.fit_transform(std_features)
    print(features.shape[1])
    print(len(features_pca))
    print (features_pca.shape[1])
    print(features_pca[1])
    return features_pca

# Function to convert a CSV to JSON
def convert_write_json(data, json_file):
    with open(json_file, "w") as f:
        f.write(json.dumps(data, sort_keys=False, indent=1, separators=(',', ': '))) 

counter = 0

def makePCAByFileName(pcaOutput):
    myarr=[]
    global counter
    for i in range(len(pcaOutput)):
        filename = "{:06d}".format(counter)
        myarr.append({'file': filename, 'Features': pcaOutput[i]})
        counter = counter + 1
    return myarr

classes = []

def getclass():
    with open("dataset_label_test.csv") as archivo:
        lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(' ')
      classes.append(int(linea[-1]))
    return classes 

clases = getclass()
#print('soy len', len(str(clases)))
clases = str(clases)
#print(str(clases))
#print(trainset[0:10])
#print(classes[0:10])
#x = np.array(trainset, dtype=np.float32)
#y = np.array(classes, dtype=np.int32)
#%%
#data = pca(features)
#features = features.tolist()
#print(int(features[0][-1]))
#print(features.flatten()[-1])
#print(features)
#features = features.flatten()[-1]
#features = str(features.tolist())
#print(features)
#new_data = makePCAByFileName(features)
##writeJSON :: (fileName, data) -> None
#print("soy lambda", list(lambda clases: clases))
grouped_data = tz.groupby(lambda clases: clases, features.flatten())
print(grouped_data)
convert_write_json(grouped_data, 'audiotestsplits_2.json')
#convert_write_json(new_data, json_file)
#convert_write_json(grouped_data, json_file)

#%%
names = ['1', '2', '3', '4', '16', '3']
tz.groupby(len, names)  #doctest: +SKIP
# %%
