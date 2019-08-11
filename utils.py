
from uuid import uuid4
import numpy as np
import tensorflow as tf
from convert_input import DATA_FILE, META_FILE
from convert_output import OUTPUT_FOLDER
from convert import write_csv

def input_max_ins():
    """
        Get the number of instruments
    """
    num_instruments = 129
    with open(META_FILE) as file:
        num_instruments = len(file.readlines())
    return num_instruments

def input_note_range():
    """
        Get the range of notes (min, max)
    """
    with open(META_FILE) as file:
        split = file.readline().split(", ")
        return (int(split[-1]), int(split[-2]))


class AddNoiseToInput(tf.keras.layers.Layer):
    """
        Add noise and one_hot data from the dataset
    """
    def __init__(self, num_instruments=0, note_range=(0, 0), time_multiplier=1, relative: bool = True):
        super(AddNoiseToInput, self).__init__()
        if num_instruments == 0:
            num_instruments = input_max_ins()
        if note_range == (0, 0):
            note_range = input_note_range()
        self.instruments = num_instruments
        self.minnote = note_range[0]
        self.maxnote = note_range[1]
        self.time = time_multiplier
        self.relative = relative

    def call(self, time, instrument, note, state):
        time = tf.cast(time, tf.float32) * tf.random.normal(shape=tf.shape(time), mean=self.time, stddev=self.time*0.05)
        instrument = tf.one_hot(instrument, self.instruments, dtype=tf.float32)
        instrument = instrument + tf.random.normal(shape=tf.shape(instrument), stddev=0.1)
        instrument = tf.nn.softmax(instrument)
        note = tf.one_hot(note, self.maxnote - self.minnote + 1, dtype=tf.float32)
        note = note + tf.random.normal(shape=tf.shape(note), stddev=0.1)
        note = tf.nn.softmax(note)
        state = tf.nn.sigmoid((tf.cast(state, tf.float32) * 3.0 - 1.5) + tf.random.normal(shape=tf.shape(state), stddev=0.2))
        return tf.concat((
            tf.expand_dims(time, -1),
            instrument,
            note,
            tf.expand_dims(state, -1)
            ), -1)

    def no_noise(self, time, instrument, note, state):
        time = tf.cast(time, tf.float32) * self.time
        instrument = tf.one_hot(instrument, self.instruments, dtype=tf.float32)
        note = tf.one_hot(note, self.maxnote - self.minnote + 1, dtype=tf.float32)
        state = tf.cast(state, tf.float32)
        return tf.concat((
            tf.expand_dims(time, -1),
            instrument,
            note,
            tf.expand_dims(state, -1)
            ), -1)

    def read_dataset(self, file=DATA_FILE, batch: int = 1, sequence: int = 1):
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
        if self.relative:
            data = data.map(lambda row: (
                tf.concat(([0.0], tf.cast(tf.cast(row["time"][1:] - row["time"][:-1], tf.float64)/100_000, tf.float32)), -1),
                row["instrument"], row["note"] - self.minnote, row["state"]))
        else:
            data = data.map(lambda row: (
                tf.cast(tf.cast(row["time"] - tf.reduce_min(row["time"]), tf.float64)/100_000, tf.float32),
                row["instrument"], row["note"] - self.minnote, row["state"],))
        data = data.shuffle(batch*80).batch(batch)
        return data.prefetch(tf.data.experimental.AUTOTUNE)

class CleanOutput(tf.keras.layers.Layer):
    """
        Clean the output from a transformer
    """
    def __init__(self, num_instruments=0, note_range=(0, 0), time_multiplier=1, as_output=False, relative: bool = True):
        super(CleanOutput, self).__init__()
        if num_instruments == 0:
            num_instruments = input_max_ins()
        if note_range == (0, 0):
            note_range = input_note_range()
        self.instruments = num_instruments
        self.minnote = note_range[0]
        self.maxnote = note_range[1]
        self.time = time_multiplier
        self.as_output = as_output
        self.relative = relative

    def call(self, output):
        if self.as_output:
            return (
                tf.nn.relu(output[:, :, 0]) / self.time,
                tf.argmax(output[:, :, 1:(1+self.instruments)], -1, tf.int32),
                tf.argmax(output[:, :, (1+self.instruments):-1], -1, tf.int32) + self.minnote,
                tf.cast(output[:, :, -1] >= 0, tf.int32))
        else:
            time = tf.nn.leaky_relu(output[:, :, 0], 0.1)
            instrument = tf.nn.softmax(output[:, :, 1:(1+self.instruments)])
            note = tf.nn.softmax(output[:, :, (1+self.instruments):-1])
            state = tf.nn.sigmoid(output[:, :, -1])
            return tf.concat((
                tf.expand_dims(time, -1),
                instrument,
                note,
                tf.expand_dims(state, -1)
                ), -1)

    def write_to_file(self, time, instrument, note, state, file=OUTPUT_FOLDER / (uuid4().hex + ".csv")):
        print("Saving to", file)
        if self.relative:
            time = np.asarray(time)
            time = (time.astype(np.float64)*1_000_000).astype(np.int32)
            for i in range(1, len(time)):
                time[i] += time[i-1]
        else:
            time = (time.astype(np.float64)*1_000_000).astype(np.int32)
        instrument = np.asarray(instrument).astype(np.int32)
        note = np.asarray(note).astype(np.int32)
        state = np.asarray(state).astype(np.int32)
        write_csv(file, META_FILE, list(zip(time.ravel(), instrument.ravel(), note.ravel(), state.ravel())))
