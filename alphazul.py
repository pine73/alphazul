import numpy as np
import tensorflow as tf


DTYPE = tf.float32

NUM_HIDDEN = 384
NUM_LAYER = 4




class InferenceNetwork(object):
    """docstring for InferenceNetwork"""
    def __init__(self, input_size, output_size, num_layer=NUM_LAYER, num_hidden = NUM_HIDDEN):
        self._graph = tf.Graph()

        with self._graph.as_default():
            with tf.name_scope('input_layer'):
                self._input_states = tf.placeholder(DTYPE, [None,input_size], 'input_states')
                self._mask = tf.placeholder(DTYPE, [None,output_size], 'mask')

            with tf.name_scope('labels'):
                self._label_value = tf.placeholder(DTYPE, [None,1], 'label_value')
                self._label_distribution = tf.placeholder(DTYPE, [None,output_size], 'label_distribution')

            with tf.name_scope('MLP'):
                layer_out = self._input_states
                for i in range(NUM_LAYER):
                    layer_out = tf.layers.dense(layer_out, NUM_HIDDEN, tf.nn.relu, name='MLP_layer_{}'.format(i))

            with tf.name_scope('value_header'):
                self._prediction_value = tf.layers.dense(layer_out, 1, tf.nn.tanh, name='value_layer')

            with tf.name_scope('distribution_header'):
                logits = tf.layers.dense(layer_out, output_size, name='logits')
                # logits_min = tf.reduce_min(logits, axis = 1)
                # masked_min = (self._mask - 1.) * logits_min
                # masked_logits = logits * self._mask - masked_min
                # masked_max = tf.reduce_max(masked_logits,axis=1)
                # self._prediction_distribution = tf.exp(logits-masked_max)*self._mask/tf.reduce_sum(tf.exp(masked_logits-masked_max)*self._mask,axis=1)

                masked_logits = logits + (self._mask - 1.) * tf.float32.max
                self._prediction_distribution = tf.nn.softmax(masked_logits)

            init = tf.global_variables_initializer()




        self._sess = tf.Session(graph = self._graph)
        self._sess.run(init)

        writer = tf.summary.FileWriter("./tensorboard/log/", self._sess.graph)
        writer.close()


    def predict(self, input_data, mask):
        feed_dict = {
            self._input_states:input_data,
            self._mask:mask,
            }
        value_p, distribution_p = self._sess.run([self._prediction_value,self._prediction_distribution],feed_dict=feed_dict)
        return value_p, distribution_p


    def train(self):
        pass


    def save(self, path):
        pass

    def restore(self, path):
        pass