"""
    Reads ogg files and saves them as a tensor stream
"""
import os
import subprocess
import random
from multiprocessing import Pool
import tensorflow as tf
import numpy as np

from model import SEQUENCE_LENGTH, BATCH_SIZE

DATA_FOLDER = "data"
SAMPLE_RATE = 32000 #44100
AUDIO_FORMAT = '.ogg'

def ffmpeg_load_audio(filename, sample_rate=SAMPLE_RATE):
    """
        Read an audiofile with ffmpeg
    """
    channels = 1
    command = [
        'ffmpeg',
        "-loglevel", "error",
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

def ffmpeg_write_audio(filename, data, sample_rate=SAMPLE_RATE):
    cmd = [ 'ffmpeg', '-y',
        '-f', 'f32le',
        '-acodec', 'pcm_f32le',
        '-ac', "1",
        '-ar', str(sample_rate),
        '-i', '-',
        filename,
        '-vn',
        '-ac', '1',
        '-acodec', 'pcm_u8']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(data.tobytes())
    p.stdin.close()


def convert_input(folder=DATA_FOLDER, sample=SAMPLE_RATE, ending=AUDIO_FORMAT):
    """
        Converts all inputs in a folder 
    """
    files = [(os.path.join(DATA_FOLDER, name), sample) for name in os.listdir(folder) if name.endswith(ending)]
    np.random.shuffle(files)
    data = np.concatenate(Pool().starmap(ffmpeg_load_audio, files))
    path = os.path.join(folder, 'data.npy')
    np.save(path, data)
    print("Saved data to:", path)


def input_fn():
    """
        Read converted input files and return input dicts
    """
    def gen():
        path = os.path.join(DATA_FOLDER, 'data.npy')
        data = np.load(path)
        if data is None or data.shape[0] == 0:
            print("Could not read the data file")
            import sys
            sys.exit()
        #for d in data:
        #    yield d
        while True:
            rnd = random.randrange(0, data.shape[0]-SEQUENCE_LENGTH-1)
            yield data[rnd:rnd+SEQUENCE_LENGTH+1]
    dataset = tf.data.Dataset.from_generator(gen, tf.float32)
    #dataset = dataset.cache().repeat()
    #dataset = dataset.batch(SEQUENCE_LENGTH+1)
    #dataset = dataset.shuffle(10000)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator().get_next()
    return {'input': iterator[:,:-1]}, {'output': iterator[:,-1:]}


if __name__ == "__main__":
    convert_input()
