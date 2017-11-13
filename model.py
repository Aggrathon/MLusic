"""
    The neural network model
"""

import tensorflow as tf
from config import NETWORK_FOLDER

def model_fn(features, labels, mode):
    input = features['input']
    prev_layer = input
    for i, s in enumerate((128, 64, 64, 128)):
        with tf.variable_scope("LSTM_%d"%i):
            cell = tf.nn.rnn_cell.LSTMCell(s, activation=tf.nn.relu)
            prev_layer, _ = tf.nn.dynamic_rnn(cell, prev_layer, dtype=tf.float32)
    prev_layer = tf.reshape(prev_layer[:, -1, :], (int(prev_layer.get_shape()[0]), int(prev_layer.get_shape()[2])))
    logits = tf.layers.dense(prev_layer, input.get_shape()[2], activation=None, name='logits')
    output = tf.nn.sigmoid(logits)
    if mode != tf.estimator.ModeKeys.PREDICT:
        label = tf.reshape(labels['output'], tf.shape(logits))
        loss = tf.losses.sigmoid_cross_entropy(label, logits)
        trainer = tf.train.AdamOptimizer(1e-6).minimize(loss, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={output:output},
            loss=loss,
            train_op=trainer
        )

def network():
    return tf.estimator.Estimator(model_fn, NETWORK_FOLDER)
