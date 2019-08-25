"""
    Util functions for the lpd dataset
"""
from uuid import uuid4
import sys
import os
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from pypianoroll import Multitrack, Track, load, write as piano_write


DATA_DIR = Path("input/lpd_cleansed/")
CONVERTED_DIR = Path("input/lpd_converted")
OUTPUT_DIR = Path("output")
SEQUENCE_LENGTH = 512
BATCH_SIZE = 32


def convert(remove_drum=True):
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
        if remove_drum:
            mt.remove_tracks([i for i, t in enumerate(mt.tracks) if t.is_drum])
        if not mt.tracks:
            continue
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

def write(track, file=None):
    if file is None:
        file = OUTPUT_DIR / (uuid4().hex + ".midi")
    track[track < 20] = 0
    track = Track(track)
    track = Multitrack(tracks=[track])
    print("Writing file " + str(file))
    piano_write(track, str(file))
    os.startfile(file)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "convert":
        convert()
    else:
        print("")
        print("")
        print("You must first download the lpd dataset and unzip it to: ", DATA_DIR)
        print("The dataset can be downloaded from: https://salu133445.github.io/lakh-pianoroll-dataset/dataset")
        print("Then you can rerun this script with: python %s convert"%sys.argv[0])
        print("")
