import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
#    model_filename ='tensorflow_inception_graph.pb'
    model_filename = '/home/jesus.molina/object_detection/models/research/objec_detection/model_SSD_N_7/frozen_inference_graph.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='Log2'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
