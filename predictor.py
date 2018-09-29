"""
    A simple generator based on predicting the next note
"""
import sys
from uuid import uuid4
import tensorflow as tf
# pylint: disable=E0611
from tensorflow.contrib.data import CsvDataset
# pylint: enable=E0611
import numpy as np
from convert_input import DATA_FILE
from convert_output import save_and_convert_song
from song import Song

SEQUENCE_LENGTH = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-5

def _input():
    with tf.device('/cpu:0'):
        data = CsvDataset(DATA_FILE, [tf.float32, tf.float32, tf.int32, tf.int32, tf.float32], 25*1024*1024)
        data = data.map(lambda x, y, z, v, w: (tf.stack((x, y, w)), z, v), 8).cache().repeat()
        data = data.batch(SEQUENCE_LENGTH).shuffle(BATCH_SIZE*128).batch(BATCH_SIZE+1)
    times, instrument, tone = data.prefetch(8).make_one_shot_iterator().get_next()
    return {"times":times, "instrument": instrument, "tone": tone}, {}

def _model(features, labels, mode):
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
    outputs, states = tf.nn.dynamic_rnn(cells, data, dtype=tf.float32, parallel_iterations=64)

    # Output
    if mode == tf.estimator.ModeKeys.PREDICT:
        output = tf.reshape(outputs[:, -1:, :], (-1, 128))
    else:
        output = tf.reshape(outputs[:, -30:, :], (-1, 128))
    out_times = tf.layers.dense(output, 3, name="dense")
    out_instr = tf.layers.dense(output, 129, name="dense_1")
    out_tone = tf.layers.dense(output, 128, name="dense_2")

    if mode == tf.estimator.ModeKeys.PREDICT:
        ins = [tf.argmax(out_instr, -1, output_type=tf.int32)]
        tim = [tf.nn.relu(out_times)]
        ton = [tf.argmax(out_tone, -1, output_type=tf.int32)]
        for _ in range(100):
            data = tf.concat([tim[-1], tf.one_hot(ins[-1], 129), tf.one_hot(ton[-1], 128)], -1)
            output, states = cells(data, states)
            tim.append(tf.nn.relu(tf.layers.dense(output, 3, reuse=True, name="dense")))
            ins.append(tf.argmax(tf.layers.dense(output, 129, reuse=True, name="dense_1"), -1, output_type=tf.int32))
            ton.append(tf.argmax(tf.layers.dense(output, 128, reuse=True, name="dense_2"), -1, output_type=tf.int32))
        predictions = {
            "times": tf.stack(tim, 1),
            "instrument": tf.stack(ins, 1),
            "tone": tf.stack(ton, 1)}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # Losses
    loss = tf.losses.compute_weighted_loss([
        tf.losses.softmax_cross_entropy(tf.reshape(instrument_onehot[:, -30:, :], (-1, 129)), out_instr),
        tf.losses.softmax_cross_entropy(tf.reshape(tone_onehot[:, -30:, :], (-1, 128)), out_tone),
        tf.losses.mean_squared_error(tf.reshape(times[:, -30:, :], (-1, 3)), out_times, [[10, 2, 0.5]]),
        tf.reduce_sum(tf.square(tf.maximum(tf.negative(out_times), 0.0)))
    ])

    # Training
    if training:
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def train():
    """
    Train the model
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    est = tf.estimator.Estimator(_model, "network/predictor")
    est.train(_input, steps=10000)

def generate(songs=1):
    """
    Generate midi with the model
    """
    s = Song().read_data(DATA_FILE)
    def _gen():
        for _ in range(songs):
            start = np.random.randint(0, s.times.shape[0]-50)
            end = start + 40
            times = s.times[start:end, :]
            instruments = s.notes[start:end, 0]
            notes = s.notes[start:end, 1]
            yield times, instruments, notes
    def _inp():
        data = tf.data.Dataset.from_generator(
            _gen, (tf.float32, tf.int32, tf.int32), ((40, 3), (40,), (40,)))
        tim, ins, ton = data.batch(1).make_one_shot_iterator().get_next()
        return {"times": tim, "instrument": ins, "tone": ton}, {}
    est = tf.estimator.Estimator(_model, "network/predictor")
    for out in est.predict(_inp, yield_single_examples=False):
        times = out["times"]
        instruments = out["instrument"]
        tones = out["tone"]
        times.shape = times.shape[1:]
        tones.shape = tones.shape[1]
        instruments.shape = instruments.shape[1]
        s.set_data(times, np.stack((instruments, tones), 1))
        save_and_convert_song(s, "output/song_"+str(uuid4())+".csv", False)

def _handle_start():
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == "generate":
        generate(5)
    else:
        print("You must specify what you want to do (train / generate)")


if __name__ == "__main__":
    _handle_start()
