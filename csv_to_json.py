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

file_in = "features_music18_out.csv"
json_file = "pca_music18.json"

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
        #f["a"]["b"]["e"].append(dict(f=file_name))

counter = 0

#makePCAByFileName :: (PCAOutput) -> PCAByFileName
# def makePCAByFileName(pcaOutput):
#     myDict = {}
#     global counter
#     #i = 000000
#     for i in range(len(pcaOutput)):
#         filename = "{:06d}".format(counter)
#         myDict[filename] = pcaOutput[i]
#         #myDict['file'] = filename
#         #myDict['PCA'] = pcaOutput[i]
#         counter = counter + 1
#     return myDict

def makePCAByFileName(pcaOutput):
    myarr=[]
    global counter
    for i in range(len(pcaOutput)):
        filename = "{:06d}".format(counter)
        myarr.append({'file': filename, 'PCA': pcaOutput[i]})
        counter = counter + 1
    return myarr

data = pca(features)
new_data = makePCAByFileName(data.tolist())
#writeJSON :: (fileName, data) -> None
convert_write_json(new_data, json_file)