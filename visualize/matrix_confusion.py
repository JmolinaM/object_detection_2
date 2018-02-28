#import
import os
import sys
import time
import copy
import h5py
import numpy as np

from tf_cnnvis import *

import tensorflow as tf
from scipy.misc import imread, imresize

# download InceptionV5 model if not
#if not os.path.exists("./inception5h.zip"):
#    os.system("python -m wget -o ./inception5h.zip https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip")


# importing InceptionV5 model
with tf.gfile.FastGFile("/home/jesus.molina/object_detection/models/research/objec_detection/model_SSD_N_7/"+"frozen_inference_graph.pb", 'rb') as f:
#with tf.gfile.FastGFile("tensorflow_inception_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.uint8, name='input') # define the input tensor
#imagenet_mean = np.uint8(117)
t_preprocessed = t_input-imagenet_mean
tf.import_graph_def(graph_def, {'image_tensor':t_preprocessed})
im = np.expand_dims(imresize(imread("/home/jesus.molina/object_detection/models/research/object_detection/test_images/image4.jpg"),(224,224)),axis = 0)
