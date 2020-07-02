from numpy import savetxt
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