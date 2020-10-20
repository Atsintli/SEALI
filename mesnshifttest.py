import numpy as np 
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import time
import torch

def mean_shift_pytorch(X, ncpus=8):
    ''' do mean shift clustering
    Input: 
        X: Tensor size (N,d) where N is data size, d is dimension of features
    Output: 
        centers: size (m,d) Center for each cluster 
        labels: size(N,) labels corresponding to input data
    '''
    # convert to numpy
    data = X.data.cpu().numpy()
    ms = MeanShift(n_jobs = ncpus)
    ms.fit(X)
    label = ms.labels_
    centers = ms.cluster_centers_
    # convert numpy to tensor
    labels = torch.tensor(label)
    centers = torch.tensor(centers) 
    return centers,labels

# Gen data
centers = [[1,1,1],[5,5,5],[5,10,10]]
X, y = make_blobs(n_samples = [5000,500,500], centers=centers, cluster_std = 1)
print (X,y)
X = torch.tensor(X)
y = torch.tensor(y)
print (X,y)

begin_time = time.time()
cluster_centers, labels = mean_shift_pytorch(X)
end_time = time.time()

print("Total time (s)", end_time- begin_time)
#print("accuracy", (labels == y).sum()true_divide.y.shape[0])
print(cluster_centers)
print(labels.shape)