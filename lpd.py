"""
    Transformer for predicting pianorolls from the lpd dataset
"""
from uuid import uuid4
import sys
import os
import random
from time import time as timer
from pathlib import Path
import numpy as np
import tensorflow as tf
from pypianoroll import Multitrack, Track, load, write as piano_write
from transformer import *


DATA_DIR = Path("input/lpd_cleansed/")
CONVERTED_DIR = Path("input/lpd_converted")
OUTPUT_DIR = Path("output")
SEQUENCE_LENGTH = 512
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "./network/transformer_lpd"


def convert():
    """
        Convert all .npz files in DATA_DIR to simpler .npz in CONVERTED_DIR
        It is possible to run multiple instances concurrently
    """
    CONVERTED_DIR.mkdir(exist_ok=True, parents=True)
    files = list(DATA_DIR.glob("**/*.npz"))
    random.shuffle(files)
    print("Converting lpd data")
    for p in files:
        out = CONVERTED_DIR / p.name
        if out.exists():
            continue
        out.touch()
        mt = load(str(p))
        st = mt.get_merged_pianoroll("max")
        st = Track(st)
        st.trim_trailing_silence()
        pr = st.pianoroll
        np.savez_compressed(out, data=pr)
        print(".", end="", flush=True)


def read_generator():
    files = list(CONVERTED_DIR.glob("*.npz"))
    while True:
        random.shuffle(files)
        for p in files:
            file = np.load(p)
            yield file["data"].astype(np.uint8)
            file.close()

def read(sequence=SEQUENCE_LENGTH+1, batch=BATCH_SIZE):
    buffer = 10000 if batch > 1 else 100
    data = tf.data.Dataset.from_generator(read_generator, tf.uint8, (None, 128))
    data = data.unbatch().batch(sequence).shuffle(buffer).batch(batch)
    data = data.map(lambda x: tf.cast(x, tf.float32))
    data = data.map(lambda x: x / 127)
    return data.prefetch(tf.data.experimental.AUTOTUNE)

def write(track, file=OUTPUT_DIR / (uuid4().hex + ".midi")):
    track[track < 20] = 0
    track = Track(track)
    track = Multitrack(tracks=[track])
    print("Writing file " + str(file))
    piano_write(track, str(file))
    os.startfile(file)


@tf.function()
def train_step(data, transformer, optimiser):
    mask = look_ahead_mask(tf.shape(data)[1]-1)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(data[:, :-1, :], data[:, :-1, :], True, None, mask, None)
        predictions = tf.nn.leaky_relu(predictions, 0.1)
        loss = tf.nn.softmax_cross_entropy_with_logits(data[:, 1:, :], predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimiser.apply_gradients(zip(gradients, transformer.trainable_variables))
    return loss

def train():
    tra = Transformer(4, 64, 8, 256, 128, 128, 0.1, False)
    opt = tf.keras.optimizers.Adam(LEARNING_RATE, 0.8, 0.95)
    checkpoint = tf.train.Checkpoint(tra=tra, opt=opt)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint)
    loss = tf.keras.metrics.Mean(name='loss')
    data = enumerate(read())
    try:
        print("Preparing training")
        loss(train_step(next(data)[1], tra, opt))
        print("Starting training")
        start = timer()
        for i, dat in data:
            loss(train_step(dat, tra, opt))
            if i % 50 == 0:
                print("Step: {:6d}      Loss: {:6.4f}      Time: {:6.2f}".format(i, loss.result(), timer() - start))
                if i % 500 == 0:
                    print("Saving checkpoint")
                    manager.save()
                    loss.reset_states()
                start = timer()
    except KeyboardInterrupt:
        print("Saving checkpoint")
        manager.save()


@tf.function()
def _infer(data, tra):
    mask = look_ahead_mask(tf.shape(data)[1])
    tmp, _ = tra(data, data, False, mask, mask, mask)
    tmp = tf.nn.softmax(tmp)[:, -1, :]
    tmp = tmp / tf.maximum(tf.reduce_max(tmp), 0.5)
    return tmp

def generate(length=SEQUENCE_LENGTH):
    """
    Generate midi with the model
    """
    tra = Transformer(4, 64, 8, 256, 128, 128, 0.1, False)
    checkpoint = tf.train.Checkpoint(tra=tra)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("No checkpoint to load")
        return
    data = next(iter(read(SEQUENCE_LENGTH + length, 1)))
    data = data.numpy()
    for i in range(SEQUENCE_LENGTH, SEQUENCE_LENGTH + length):
        data[:, i, :] = _infer(data[:, (i-SEQUENCE_LENGTH):i, :], tra).numpy()
    write(data[0, :, :]*127)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "convert":
        convert()
    elif len(sys.argv) == 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == "generate":
        generate()
    else:
        print("You must specify what you want to do (train / generate / convert)")
