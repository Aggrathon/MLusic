"""
    Transformer for predicting the next "pseudo" midi events
    (Currently probably broken)
"""
import sys
from time import time as timer
from uuid import uuid4
import tensorflow as tf
import numpy as np
from convert_input import DATA_FILE, META_FILE
from convert_output import OUTPUT_FOLDER
from convert import write_csv
from utils import input_max_ins
from transformer import *

SEQUENCE_LENGTH = 257
BATCH_SIZE = 64
LEARNING_RATE = 1e-6
CHECKPOINT_PATH = "./network/transformer_midi"


def _input_tf(file=DATA_FILE, batch: int = BATCH_SIZE, sequence: int = SEQUENCE_LENGTH, relative: bool = True):
    """
    Read the input dataset

    Keyword Arguments:
        input {} -- input file name (default: {DATA_FILE})
        batch {int} -- batch size (default: {BATCH_SIZE})
        sequence {int} -- sequence length (default: {SEQUENCE_LENGTH})

    Returns:
        PrefetchDataset -- (time, instrument, tone, state)
    """
    data = tf.data.experimental.make_csv_dataset(
        file_pattern=str(file),
        batch_size=sequence,
        column_names=["time", "instrument", "note", "state"],
        column_defaults=[0, 0, 0, 0],
        shuffle=False,
        header=False)
    if relative:
        data = data.map(lambda row: (
            tf.concat(([0.0], tf.cast(tf.cast(row["time"][1:] - row["time"][:-1], tf.float64)/100_000, tf.float32)), -1),
            row["instrument"],
            tf.cast(row["note"], tf.float32),
            tf.cast(row["state"], tf.float32)))
    else:
        data = data.map(lambda row: (
            tf.cast(tf.cast(row["time"] - tf.reduce_min(row["time"]), tf.float64)/100_000, tf.float32),
            row["instrument"],
            tf.cast(row["note"], tf.float32),
            tf.cast(row["state"], tf.float32)))
    data = data.shuffle(batch*80).batch(batch)
    return data.prefetch(tf.data.experimental.AUTOTUNE)

def _input_np(file=DATA_FILE, sequence: int = SEQUENCE_LENGTH, relative: bool = True):
    """
        Read the input dataset to numpy arrays
    """
    with open(file) as f:
        lines = f.readlines()
        start = np.random.randint(len(lines) - sequence)
        lines = lines[start:(start + sequence)]
        time = []
        inst = []
        note = []
        stat = []
        for l in lines:
            s = l.split(", ")
            time.append(int(s[0]))
            inst.append(int(s[1]))
            note.append(float(s[2]))
            stat.append(float(s[3]))
        time = np.asarray(time)
        if relative:
            time = np.concatenate(([0], ((time[1:] - time[:-1]).astype(np.float64)/100_000).astype(np.float32)))
        else:
            time = ((time - np.min(time)).astype(np.float64)/100_000).astype(np.float32)
    return (time, np.asarray(inst), np.asarray(note), np.asarray(stat))

def _output(time, instrument, note, state, file=OUTPUT_FOLDER / (uuid4().hex + ".csv"), relative: bool = True):
    print("Saving to", file)
    if relative:
        time = np.asarray(time)
        time = (time.astype(np.float64)*1_000_000).astype(np.int32)
        for i in range(1, len(time)):
            time[i] += time[i-1]
    else:
        time = (time.astype(np.float64)*1_000_000).astype(np.int32)
    instrument = np.asarray(instrument)
    note = np.asarray(note).astype(np.int32)
    state = np.asarray(state).astype(np.int32)
    write_csv(file, META_FILE, list(zip(time, instrument, note, state)))


def _loss(vec, time, instrument, note, state):
    loss_state = tf.reduce_mean(tf.losses.binary_crossentropy(state, vec[:, :, -1], from_logits=True, label_smoothing=0.1))
    loss_note = tf.reduce_mean(tf.losses.MSE(note, vec[:, :, -2])) / 10
    loss_time = tf.reduce_mean(tf.losses.logcosh(time, vec[:, :, 0])) * 10
    loss_inst = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(instrument, vec[:, :, 1:-2], from_logits=True))
    return loss_time + loss_inst + loss_note + loss_state, loss_time, loss_inst, loss_note, loss_state


@tf.function()
def _train_step(time, instrument, note, state, transformer, optimiser, max_ins=129):
    mask = 1 - _mask(1, tf.shape(instrument)[-1])
    data = tf.concat((
        tf.expand_dims(time, -1),
        tf.one_hot(instrument, max_ins, dtype=tf.float32),
        tf.expand_dims(note, -1),
        tf.expand_dims(state, -1)), -1)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(data[:, :-1, :], data[:, :-1, :], True, mask, mask, mask)
        loss = _loss(predictions, time[:, 1:], instrument[:, 1:], note[:, 1:], state[:, 1:])
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimiser.apply_gradients(zip(gradients, transformer.trainable_variables))
    return loss


def train():
    """
    Train the model
    """
    data = _input_tf(sequence=SEQUENCE_LENGTH)
    ins = input_max_ins()
    optimiser = tf.keras.optimizers.Adam(LEARNING_RATE, 0.9, 0.98, 1e-9)
    vec_size = 3 + ins
    transformer = Transformer(SEQUENCE_LENGTH-1, 4, 128, 8, 512, vec_size, 0.1)
    checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimiser)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint)
    loss_total = tf.keras.metrics.Mean(name='loss_total')
    loss_state = tf.keras.metrics.Mean(name='loss_state')
    loss_note = tf.keras.metrics.Mean(name='loss_note')
    loss_time = tf.keras.metrics.Mean(name='loss_time')
    loss_inst = tf.keras.metrics.Mean(name='loss_inst')
    data = enumerate(data)
    next(data)
    try:
        start = timer()
        print("Starting training")
        for (i, (time, instrument, note, state)) in data:
            loss = _train_step(time, instrument, note, state, transformer, optimiser, ins)
            loss_time(loss[1])
            loss_inst(loss[2])
            loss_note(loss[3])
            loss_state(loss[4])
            loss_total(loss[0])
            if i % 50 == 0:
                print("Batch: {:6d}      Loss: {:5.3f} ({:4.2f} {:4.2f} {:4.2f} {:4.2f})      Time: {:6.2f}".format(
                    i,
                    loss_total.result(),
                    loss_time.result(),
                    loss_inst.result(),
                    loss_note.result(),
                    loss_state.result(),
                    timer() - start))
                if i % 500 == 0:
                    print("Saving checkpoint")
                    manager.save()
                    loss_total.reset_states()
                    loss_state.reset_states()
                    loss_note.reset_states()
                    loss_time.reset_states()
                    loss_inst.reset_states()
                start = timer()
    except KeyboardInterrupt:
        print("Saving checkpoint")
        manager.save()

def _output_to_int(pred):
    return (
        tf.nn.relu(pred[:, :, 0]),
        tf.argmax(pred[:, :, 1:-3], -1, tf.int32),
        tf.round(pred[:, :, -2]),
        tf.cast(pred[:, :, -1] >= 0, tf.float32))

@tf.function()
def _infer(time, instrument, note, state, transformer, index, max_ins=129):
    mask = 1 - _mask(1, tf.shape(instrument)[-1])
    data = tf.expand_dims(tf.concat((
        tf.expand_dims(tf.cast(time, tf.float32), -1),
        tf.one_hot(instrument, max_ins, dtype=tf.float32),
        tf.expand_dims(tf.cast(note, tf.float32), -1),
        tf.expand_dims(tf.cast(state, tf.float32), -1)), -1), 0)
    predictions, _ = transformer(data[:, :-1, :], data[:, :-1, :], True, mask, mask, mask)
    return _output_to_int(predictions[:, index:(index + 1), :])

def generate():
    """
    Generate midi with the model
    """
    time, instrument, note, state = _input_np(sequence=SEQUENCE_LENGTH)
    ins = input_max_ins()
    vec_size = 3 + ins
    transformer = Transformer(SEQUENCE_LENGTH-1, 4, 128, 8, 512, vec_size, 0.1)
    checkpoint = tf.train.Checkpoint(transformer=transformer)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("No checkpoint to load")
        return
    for j in range(20, SEQUENCE_LENGTH-1):
        t, i, n, s = _infer(time, instrument, note, state, transformer, 20, ins)
        time[j+1] = t
        instrument[j+1] = i
        note[j+1] = n
        state[j+1] = s
        print(".", end="")
    print("")
    _output(time, instrument, note, state)

def _handle_start():
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == "generate":
        generate()
    else:
        print("You must specify what you want to do (train / generate)")


if __name__ == "__main__":
    _handle_start()
