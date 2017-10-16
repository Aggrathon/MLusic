"""
    The AudioGAN neural network
"""
import tensorflow as tf

DATA_FOLDER = "data"
SAMPLE_RATE = 44100
SEQUENCE_LENGTH = 10*SAMPLE_RATE
AUDIO_FORMAT = 'wav'


def read_data(folder=DATA_FOLDER, sample=SAMPLE_RATE, length=SEQUENCE_LENGTH, format=AUDIO_FORMAT):
    """
        Get batched audio sequences from a folder
    """
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(folder+"/*."+format))
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    audio = tf.contrib.ffmpeg.decode_audio(value, format, sample, 1)
    slices = [tf.random_crop(audio, (length, 1)) for _ in range(100)]
    batch = tf.train.shuffle_batch([slices], 32, 5000, 100, 4, enqueue_many=True)
    return batch


