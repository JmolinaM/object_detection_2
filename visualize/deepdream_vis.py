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

image_name="/home/jesus.molina/object_detection/models/research/object_detection/test_images/image4.jpg"
model= "/home/jesus.molina/object_detection/models/research/objec_detection/model_SSD_N_7/"+"frozen_inference_graph.pb"
# importing InceptionV5 model
with tf.gfile.FastGFile(model, 'rb') as f:
#with tf.gfile.FastGFile("tensorflow_inception_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.uint8, name='input') # define the input tensor
imagenet_mean = np.uint8(117)
t_preprocessed = t_input-imagenet_mean
tf.import_graph_def(graph_def, {'image_tensor':t_preprocessed})
im = np.expand_dims(imresize(imread(image_name),(224,224)),axis = 0)
#layer = "import/softmax2_pre_activation"
layer= "import/BoxPredictor_3/ClassPredictor/Conv2D"
#layer = "import/BoxPredictor_0/ClassPredictor/BiasAdd"
start = time.time()
#layer= "import/Preprocessor/mul"

# api call
#is_success = activation_visualization(sess_graph_path = tf.get_default_graph(), value_feed_dict ={t_input :im},layers=layer)
is_success = deepdream_visualization(sess_graph_path = tf.get_default_graph(), value_feed_dict ={t_input :im}, layer=layer, classes = [1, 2, 3, 4, 5, 6, 7])
start = time.time() - start
print("Total Time = %f" % (start))
