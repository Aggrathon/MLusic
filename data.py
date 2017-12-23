"""
    Reads ogg files and saves them as a tensor stream
"""
import os
import subprocess
import datetime
import tensorflow as tf
import numpy as np

from model import SEQUENCE_LENGTH, BATCH_SIZE

DATA_FOLDER = "data"
SAMPLE_RATE = 44100
AUDIO_FORMAT = 'ogg'

def ffmpeg_load_audio(filename, sample_rate=SAMPLE_RATE):
    """
        Read an audiofile with ffmpeg
    """
    channels = 1
    command = [
        'ffmpeg',
        '-i', filename,
        '-f', 'f32le',
        '-acodec', 'pcm_f32le',
        '-ar', str(sample_rate),
        '-ac', str(channels),
        '-']
    p = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=4096)
    bytes_per_sample = np.dtype(np.float32).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sample_rate # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.fromstring(raw, dtype=np.float32).astype(np.float32)
    if audio.size == 0:
        return audio
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    return audio

def read_data(folder=DATA_FOLDER, sample=SAMPLE_RATE, ending=AUDIO_FORMAT):
    """
        Get a combined audio sequence from a folder
    """
    files = [name for name in os.listdir(folder) if name[-len(ending):] == ending]
    np.random.shuffle(files)
    return np.concatenate([ ffmpeg_load_audio(os.path.join(DATA_FOLDER, name), sample) for name in files ])

def write_data(data):
    """
        Write the data to a *.tfrecords file
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)
    timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
    with tf.python_io.TFRecordWriter(os.path.join(DATA_FOLDER, "data_%s.tfrecords"%(timestamp))) as writer:
        for d in data:
            record = tf.train.Example(features=tf.train.Features(feature={
                'sound': tf.train.Feature(float_list=tf.train.FloatList(value=(d,)))
            }))
            writer.write(record.SerializeToString())

def input_fn():
    """
        Read *.tfrecords files and return input dicts
    """
    files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if '.tfrecord' in f]
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(lambda r: tf.parse_single_example(r, { 'sound': tf.FixedLenFeature((), tf.float32) }))
    dataset = dataset.repeat()
    dataset = dataset.batch(SEQUENCE_LENGTH)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator().get_next()
    print(iterator.get_shape())
    return {'input': iterator}, None


def convert_input():
    print("Reading Data")
    data = read_data()
    print("Writing Data")
    write_data(data)

if __name__ == "__main__":
    convert_input()
