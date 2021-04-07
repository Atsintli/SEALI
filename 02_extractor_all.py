import sys
import os
import subprocess

#rootDir = sys.argv[1]
#rootSalida = sys.argv[2]
rootDir = 'segments_music18'
rootSalida = 'music18_all_features'
extractor = '/../../home/atsintli/Desktop/essentia-extractor/streaming_extractor_music'
print(rootDir)
#os.mkdir(rootSalida)

def analizador(root, fileName):
    print("esta analizando " + fileName)
    file_name, file_extension = os.path.splitext(fileName)
    subprocess.call([extractor,
                     root + "/" + fileName, rootSalida + "/" + file_name + ".json"]) # 'profile.yml'

for root, dirs, files in os.walk(rootDir):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        #print(len(path) * '---', file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() == ".wav":
            print(len(path) * '---', file)
            analizador(root, file)
