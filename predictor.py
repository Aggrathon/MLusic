"""
    A simple generator based on predicting the next note
"""

import sys
import tensorflow as tf
# pylint: disable=E0611
from tensorflow.contrib.data import CsvDataset
# pylint: enable=E0611
from convert_input import DATA_FILE

SEQUENCE_LENGTH = 60
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

def _input():
    with tf.device('/cpu:0'):
        data = CsvDataset(DATA_FILE, [tf.float32, tf.float32, tf.int32, tf.int32, tf.float32], 25*1024*1024)
        data = data.map(lambda x, y, z, v, w: (tf.stack((x, y, w)), z, v), 8).cache().repeat()
        data = data.batch(SEQUENCE_LENGTH).shuffle(BATCH_SIZE*64).batch(BATCH_SIZE+1)
    times, instrument, tone = data.prefetch(8).make_one_shot_iterator().get_next()
    return {"times":times, "instrument": instrument, "tone": tone}, {}

def _model(features, labels, mode):
    #pylint: disable=unused-argument
    #pylint: enable=unused-argument
    times = features["times"]
    instrument = features["instrument"]
    tone = features["tone"]
    instrument_onehot = tf.one_hot(instrument, 129)
    tone_onehot = tf.one_hot(tone, 128)
    data = tf.concat([times[:, :-1, :], instrument_onehot[:, :-1, :], tone_onehot[:, :-1, :]], -1)
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Recurrent NN
    resize = tf.nn.rnn_cell.BasicRNNCell(128, activation=tf.nn.relu)
    def _create_cell():
        cell = tf.nn.rnn_cell.GRUCell(128)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
        cell = tf.nn.rnn_cell.ResidualWrapper(cell)
        return cell
    cells = tf.nn.rnn_cell.MultiRNNCell([resize] + [_create_cell() for _ in range(8)])
    outputs, _ = tf.nn.dynamic_rnn(cells, data, dtype=tf.float32, parallel_iterations=64)

    # Output
    output = tf.reshape(outputs[:, -10:, :], (-1, 128))
    out_times = tf.layers.dense(output, 3)
    out_instr = tf.layers.dense(output, 129)
    out_tone = tf.layers.dense(output, 128)

    # Losses
    loss = tf.losses.compute_weighted_loss([
        tf.losses.softmax_cross_entropy(tf.reshape(instrument_onehot[:, -10:, :], (-1, 129)), out_instr),
        tf.losses.softmax_cross_entropy(tf.reshape(tone_onehot[:, -10:, :], (-1, 128)), out_tone),
        tf.losses.mean_squared_error(tf.reshape(times[:, -10:, :], (-1, 3)), out_times, [[10, 2, 0.5]])
    ])

    # Training
    if training:
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def _handle_start():
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        tf.logging.set_verbosity(tf.logging.INFO)
        est = tf.estimator.Estimator(_model, "network/predictor")
        est.train(_input, steps=10000)
    if len(sys.argv) == 2 and sys.argv[1] == "generate":
        pass
    else:
        print("You must specify what you want to do (train / generate)")


if __name__ == "__main__":
    _handle_start()
