#import Extract_MFCCs as xmfccs
import glob
#import librosa
import csv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from numpy import loadtxt
import os
import matplotlib.pyplot as plt
from itertools import cycle
import shutil

file_test = "dataBaseAsMatrix_standard_test.csv"


features_test = loadtxt(file_test)
c = loadtxt("centros.csv")

#for test_feature in features_test:
#  print(test_feature)
#  center_i = 0
#  for center in c:
#    this_distance = np.linalg.norm(test_feature-center)
#    print(str(center_i) + ": " + str(this_distance))
#    center_i = center_i + 1

center_i = 0
for center in c:
  this_distance = np.linalg.norm(features_test[0,:]-center)
  print(str(center_i) + ": " + str(this_distance))
  center_i = center_i + 1


