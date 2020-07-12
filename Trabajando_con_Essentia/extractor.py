import sys
import os
import subprocess


# python3 extractor.py /Users/hugosg/Documents/OttoSalida/ /Users/hugosg/Documents/OttoSalidaData/
# entrada carpeta con pedacitos OJO aqui ya estan los pedacitos
# salida lugar de carpeta nueva para guardar analisis

rootDir = "/home/diego/code/aaron/maquina-que-escucha/Trabajando_con_Essentia/Segments/"
rootSalida = "/home/diego/code/aaron/maquina-que-escucha/Trabajando_con_Essentia/outy/"
print(rootDir)
# os.mkdir(rootSalida)


def analizador(root, fileName):
    print("esta analizando " + fileName)
    file_name, file_extension = os.path.splitext(fileName)
    subprocess.call(["./streaming_extractor_music",
                     root + "/" + fileName, rootSalida + "/" + file_name + ".json"])


for root, dirs, files in os.walk(rootDir):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        #print(len(path) * '---', file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() == ".wav":
            print(len(path) * '---', file)
            analizador(root, file)
