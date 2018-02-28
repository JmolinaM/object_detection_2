import os
import io
import pandas as pd
import tensorflow as tf
import numpy as np
#from PIL import image
import csv
N= 1
confusion_matrix= np.zeros((N+1,N+1))
rep=np.ones((N+1,N+1))
flags= np.zeros((N+1,N+1))
with open('/home/jesus.molina/matchGT_matrix.csv', 'r') as csv_file:
  newFileReader = csv.reader(csv_file)
  for row in newFileReader: 
#    print(row[0])
    confusion_matrix[int(row[1])][int(row[0])]=float(row[2]) +confusion_matrix[int(row[1])][int(row[0])]
    rep[int(row[1])][int(row[0])]=rep[int(row[1])][int(row[0])]+1
    flags[int(row[1])][int(row[0])]=1
with open('/home/jesus.molina/FP_matrix.csv', 'r') as csv_file:
  newFileReader = csv.reader(csv_file)
  for row in newFileReader:
    confusion_matrix[int(row[1])][int(row[0])]=float(row[2]) +confusion_matrix[int(row[1])][int(row[0])]
    rep[int(row[1])][int(row[0])]=rep[int(row[1])][int(row[0])]+1   
    flags[int(row[1])][int(row[0])]=1
confusion_matrix= np.divide(confusion_matrix, rep-flags)
print confusion_matrix
rep = rep-1
print rep

with open('/home/jesus.molina/confusion_matrix_N_1.csv', 'w') as csv_write:
  writer = csv.writer(csv_write, lineterminator='\n')
  writer.writerows(rep)
with open('/home/jesus.molina/scores_matrix_N_1.csv', 'w') as csv_write:
  writer = csv.writer(csv_write, lineterminator='\n')
  writer.writerows(confusion_matrix)
