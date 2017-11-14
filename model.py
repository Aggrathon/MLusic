"""
    The neural network model
"""

import tensorflow as tf
from config import NETWORK_FOLDER, MAX_INSTRUMENTS, MAX_TONE

def model_fn(features, labels, mode):
    input = features['input']
    prev_layer = input
    for i, s in enumerate((160, 128, 160, 128)):
        with tf.variable_scope("LSTM_%d"%i):
            cell = tf.nn.rnn_cell.LSTMCell(s, activation=tf.nn.relu)
            prev_layer, _ = tf.nn.dynamic_rnn(cell, prev_layer, dtype=tf.float32)
    prev_layer = tf.reshape(prev_layer[:, -1, :], (int(prev_layer.get_shape()[0]), int(prev_layer.get_shape()[2])))
    logits = tf.layers.dense(prev_layer, input.get_shape()[2], activation=None, name='logits')
    output = tf.concat([
        tf.nn.softmax(logits[:, :MAX_INSTRUMENTS]),
        tf.nn.softmax(logits[:, MAX_INSTRUMENTS:MAX_INSTRUMENTS+MAX_TONE]),
        tf.nn.relu(logits[:, -2:])
    ], 1)
    if mode != tf.estimator.ModeKeys.PREDICT:
        label = tf.reshape(labels['output'], tf.shape(logits))
        with tf.variable_scope('Loss'):
            loss_instr = tf.losses.softmax_cross_entropy(label[:, :MAX_INSTRUMENTS], logits[:, :MAX_INSTRUMENTS])
            loss_tone = tf.losses.softmax_cross_entropy(label[:, MAX_INSTRUMENTS:MAX_INSTRUMENTS+MAX_TONE], logits[:, MAX_INSTRUMENTS:MAX_INSTRUMENTS+MAX_TONE])
            loss_len = tf.losses.mean_squared_error(label[:, -2], tf.nn.relu(logits[:, -2]))*0.0001
            loss_del = tf.losses.mean_squared_error(label[:, -1], tf.nn.relu(logits[:, -1]))*0.01
            tf.summary.scalar("Instrument", loss_instr)
            tf.summary.scalar("Tone", loss_tone)
            tf.summary.scalar("Length", loss_len)
            tf.summary.scalar("Delay", loss_del)
            loss = tf.add_n([loss_instr, loss_tone, loss_len, loss_del])
        trainer = tf.train.AdamOptimizer(5e-8).minimize(loss, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'output': output},
            loss=loss,
            train_op=trainer
        )
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'output': output}
        )

def network():
    return tf.estimator.Estimator(model_fn, NETWORK_FOLDER)
