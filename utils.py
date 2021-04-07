from numpy import savetxt
import numpy as np
import json
import glob
import librosa
import os

def get_json(filename):
    resultF = open(filename, 'r')
    result = resultF.read()
    #resultF.close()
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
	    #m = np.matrix(features)
	    savetxt(f, features, fmt='%s') #fmt='%s'
	print('Save Descriptors as Matrix: Done')

###Write files by the number of clusters###
def writeFiles(n_clusters_, labels):
    folder = 'Clusters/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    files = []

    for audio_files in sorted(glob.glob( 'Segments/' + "*.wav" )):
        names = audio_files.split('/')[1]
        #print(names)
        files.append(names)

    with open('archivos_clases.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(files, labels))

    # concatenate classes in archives
    clases = open('archivos_clases.txt')
    clasescontent = clases.readlines()
    clases = [int(x.split(" ")[1]) for x in clasescontent]

    for clase in range(n_clusters_):
        print("iterando sobre " + str(clase))
        ele = np.where(np.array(clases) == clase)[0]
        print("indices de clase " + str(clase) + " son: " + str(ele))
        #print(ele)
        audiototal = np.array([])
        for elements in ele:
            num = 'Segments/{:06d}'.format(elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                #print(audiototal)
                librosa.output.write_wav("Clusters/" + "CLASE_"
                                         + str(clase) + ".wav", audiototal, sr)
            #print(audiototal)