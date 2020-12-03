from numpy import savetxt
import numpy as np
import json


def get_json(filename):
    resultF = open(filename, 'r')
    result = resultF.read()
    resultF.close()
    return json.loads(result)


def save_matrix_array(file_name, matrixes):
    with open(file_name, 'w') as f:
        for matrix in matrixes:
            savetxt(f, matrix)

def save_as_json(file_name, data):
    with open(file_name, 'w') as f:
        f.write(json.dumps(data))
        f.close()

def save_descriptors_as_matrix(file_name, features):
	with open(file_name, 'w') as f:
	    m = np.matrix(features)
	    savetxt(f, m, fmt='%s') #fmt='%s'
	print('Save Descriptors as Matrix: Done')