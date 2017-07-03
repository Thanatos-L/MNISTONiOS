from __future__ import print_function
from tensorflow.core.framework import graph_pb2
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u' %d  %s' % (i, n)) for i, n in enumerate(node.input)]

# read frozen graph and display nodes
graph = tf.GraphDef()
with tf.gfile.Open('./frozen_model_without_dropout.pb', 'r') as f:
    data = f.read()
    graph.ParseFromString(data)
    
display_nodes(graph.node)