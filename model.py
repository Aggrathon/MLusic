"""
    The neural network model
"""

import tensorflow as tf
from config import NETWORK_FOLDER, MAX_INSTRUMENTS, MAX_TONE, BATCH_SIZE

def model_fn(features, labels, mode, params):
    input = features['input']
    prev_layer = input
    for i, s in enumerate((160, 128, 160, 128)):
        with tf.variable_scope("LSTM_%d"%i):
            cell = tf.nn.rnn_cell.LSTMCell(s)
            if mode == tf.estimator.ModeKeys.TRAIN:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 0.8)
            prev_layer, _ = tf.nn.dynamic_rnn(cell, prev_layer, dtype=tf.float32)
    prev_layer = tf.reshape(prev_layer[:, -1, :], (params['batch_size'], int(prev_layer.get_shape()[2])))
    logits = tf.layers.dense(prev_layer, input.get_shape()[2], activation=None, name='logits')
    output = tf.concat([
        tf.nn.softmax(logits[:, :MAX_INSTRUMENTS]+tf.random_uniform((params['batch_size'], MAX_INSTRUMENTS), -0.1, 0.1)),
        tf.nn.softmax(logits[:, MAX_INSTRUMENTS:MAX_INSTRUMENTS+MAX_TONE]+tf.random_uniform((params['batch_size'], MAX_TONE), -0.1, 0.1)),
        tf.nn.relu(logits[:, -2:])
    ], 1)
    if mode != tf.estimator.ModeKeys.PREDICT:
        label = tf.reshape(labels['output'], tf.shape(logits))
        with tf.variable_scope('Loss'):
            loss_instr = tf.losses.softmax_cross_entropy(label[:, :MAX_INSTRUMENTS], logits[:, :MAX_INSTRUMENTS])
            loss_tone = tf.losses.softmax_cross_entropy(label[:, MAX_INSTRUMENTS:MAX_INSTRUMENTS+MAX_TONE], logits[:, MAX_INSTRUMENTS:MAX_INSTRUMENTS+MAX_TONE])
            loss_len = tf.losses.mean_squared_error(label[:, -2], tf.nn.relu(logits[:, -2]))*0.001
            loss_del = tf.losses.mean_squared_error(label[:, -1], tf.nn.relu(logits[:, -1]))*0.01
            tf.summary.scalar("Instrument", loss_instr)
            tf.summary.scalar("Tone", loss_tone)
            tf.summary.scalar("Length", loss_len)
            tf.summary.scalar("Delay", loss_del)
            loss = tf.add_n([loss_instr, loss_tone, loss_len, loss_del])
        trainer = tf.train.AdamOptimizer(1e-6).minimize(loss, tf.train.get_global_step())
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

def network(batch_size=BATCH_SIZE):
    return tf.estimator.Estimator(model_fn, NETWORK_FOLDER, None, {'batch_size': batch_size})
