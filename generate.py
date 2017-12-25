"""
    Generate a soundfile with the GAN
"""
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from model import model_fn, NETWORK_FOLDER, SEQUENCE_LENGTH
from data import ffmpeg_write_audio, SAMPLE_RATE

OUTPUT_FOLDER = 'output'

def input_fn():
    """
        Empty input_fn for estimator
    """
    return dict(input=None), dict()

def generate():
    tf.logging.set_verbosity(tf.logging.INFO)
    placeholder = None
    def input_fn():
        global placeholder
        placeholder = tf.placeholder(tf.float32, (1, SEQUENCE_LENGTH))
        return dict(input=placeholder), None
    song = [np.clip(np.random.normal(0.0, 0.3), -1.0, 1.0) for n in range(SEQUENCE_LENGTH)]
    def feed_fn():
        global song
        global placeholder
        return { placeholder: song[-SEQUENCE_LENGTH:] }
    model = tf.estimator.Estimator(model_fn, NETWORK_FOLDER, params=dict(batch_size=1))
    for i, res in enumerate(model.predict(input_fn, hooks=[tf.train.FeedFnHook(feed_fn)])):
        song.append(res['output'][0])
        if i > SEQUENCE_LENGTH*2+SAMPLE_RATE*10:
            break
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    ffmpeg_write_audio(os.path.join(OUTPUT_FOLDER, 'wavenet_{:%y%m%d%H%M%S}.wav'.format(datetime.now())), song[SEQUENCE_LENGTH*2:])

if __name__ == "__main__":
    generate()
