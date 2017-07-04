from __future__ import print_function
from tensorflow.core.framework import graph_pb2
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/MNIST_data', one_hot=True)

def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u' %d  %s' % (i, n)) for i, n in enumerate(node.input)]
        
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def test_graph(graph_path, use_dropout):
    tf.reset_default_graph()
    graph_def = tf.GraphDef()
    
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        
    _ = tf.import_graph_def(graph_def, name='')
    sess = tf.Session()    
    prediction_tensor = sess.graph.get_tensor_by_name('model/y_pred:0') 
    
    feed_dict = {'input:0': mnist.test.images[:256]}
    if use_dropout:
        feed_dict['keep_prob:0'] = 1.0
        
    predictions = sess.run(prediction_tensor, feed_dict)
    result = accuracy(predictions, mnist.test.labels[:256])
    return result

# read frozen graph and display nodes
graph = tf.GraphDef()
with tf.gfile.Open('./frozen.pb', 'r') as f:
    data = f.read()
    graph.ParseFromString(data)
    
display_nodes(graph.node)

# Connect 'MatMul_1' with 'Relu_2'
graph.node[44].input[0] = 'Relu_2' # 44 -> MatMul_1
# Remove dropout nodes
nodes = graph.node[:29] + graph.node[40:] 
#nodes = graph.node[:38] + graph.node[42:] 
#nodes = graph.node[:38]
#del nodes[38] # inference/inference Equal

# Save graph
output_graph = graph_pb2.GraphDef()
output_graph.node.extend(nodes)
with tf.gfile.GFile('./frozen_model_without_dropout.pb', 'w') as f:
    f.write(output_graph.SerializeToString())

"""
# test graph via simple test
#result_1 = test_graph('./frozen_with_dropout.pb', use_dropout=True)
#result_2 = test_graph('./frozen_model_without_dropout.pb', use_dropout=False)

#print('with dropout:    %f' % result_1)
#print('without dropout: %f' % result_2)
"""