# import numpy as np
# from numpy import savetxt
import glob
import csv
import os

def append_classes():
    data = []
    f_in=open('mfccs.csv', 'r')
    f_out = 'mfccs_classes.csv'
    f_out=open('mfccs_clases.csv', 'w')
    #f_out = open(f_out, 'a')
    for line in f_in.readlines():
        data.append(line + '0')
        print (line)
        #f_out.write(line.split(",")[0]+"")
        with open('mfccs_clases.csv', 'wb', newline='') as csvfile:
          #csvfile.write(line.split(",")[0]+",")
          #filewriter = csv.writer(open("test_step1.csv", "wb"), delimiter=",", newline="")
          spamwriter = csv.writer(csvfile, delimiter=',')
          spamwriter.writerow(line)
    #file = open(file_name, 'w')

append_classes()

# with open('eggs.csv', 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
#     spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])