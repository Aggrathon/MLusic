"""
    Reads ogg files and saves them as a tensor stream
"""
import os
import subprocess
from multiprocessing import Pool
import tensorflow as tf
import numpy as np

from model import SEQUENCE_LENGTH, BATCH_SIZE

DATA_FOLDER = "data"
SAMPLE_RATE = 32000 #44100
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

def _convert(file):
    data = ffmpeg_load_audio(file)
    with open(file+".csv", "w") as f:
        for d in data:
            f.write(str(d)+'\n')
    print("Converted:", file)


def convert_input(folder=DATA_FOLDER, ending=AUDIO_FORMAT):
    """
        Converts all inputs in a folder 
    """
    files = [os.path.join(DATA_FOLDER, name) for name in os.listdir(folder) if name.endswith(ending)]
    np.random.shuffle(files)
    Pool().map(_convert, files)


def input_fn():
    """
        Read converted input files and return input dicts
    """
    def gen():
        files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
        np.random.shuffle(files)
        for file in files:
            with open(file) as f:
                line = f.readline()
                if line:
                    try:
                        yield float(line[:-1])
                    except:
                        pass
    dataset = tf.data.Dataset.from_generator(gen, tf.float32, ())
    dataset = dataset.cache().repeat()
    dataset = dataset.batch(SEQUENCE_LENGTH)
    dataset = dataset.shuffle(5000)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    return {'input': iterator.get_next()[0]}, None


if __name__ == "__main__":
    convert_input()
