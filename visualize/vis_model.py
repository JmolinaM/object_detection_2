import os
import sys
from six import iteritems 
import time
import copy
import tf_cnnvis
import h5py
import numpy as np
from tf_cnnvis import *
import tensorflow as tf
from scipy.misc import imread, imresize


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.shape
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
with tf.gfile.FastGFile('tensorflow_inception_graph.pb', 'rb') as f:
#with tf.gfile.FastGFile("/home/jesus.molina/object_detection/models/research/objec_detection/model_SSD_N_7/"+"frozen_inference_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
mean= 117

im = np.expand_dims(imresize(imresize(imread("/home/jesus.molina/object_detection/models/research/object_detection/test_images/image1.jpg"), (256, 256)) - mean, (224, 224)), axis = 0)
#image_np = load_image_into_numpy_array(t_preprocessed)
image_np_expanded= np.expand_dims(t_input, axis=0)
tf.import_graph_def(graph_def, {'input':t_input})
sess = tf.Session(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())
layers = 'import/softmax2_pre_activation'
"""
with sess.as_default():
	is_success=tf_cnnvis.activation_visualization(sess_graph_path= None, value_feed_dict = {t_input : im}, input_tensor=None, layers=layers)
with sess.as_default():
	is_success=tf_cnnvis.deconv_visualization(sess_graph_path = None, value_feed_dict =  {t_input : im})
"""
with sess.as_default() as g:
#	for key_op, value in iteritems(im):
		tmp = get_tensor(graph=g, name = key_op.name)

	is_success = tf_cnnvis.deepdream_visualization(sess_graph_path= None, value_feed_dict = {t_input : im}, layer=layers, classes = [1, 2, 3, 4, 5])
sess_close()

