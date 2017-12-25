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

def generate():
    tf.logging.set_verbosity(tf.logging.INFO)
    placeholder = []
    def input_fn():
        nonlocal placeholder
        placeholder = tf.placeholder(tf.float32, (SEQUENCE_LENGTH))
        return dict(input=tf.reshape(placeholder, (1, SEQUENCE_LENGTH))), None
    song = [np.clip(np.random.uniform(-1.0, 1.0), -1.0, 1.0) for i in range(SEQUENCE_LENGTH)]
    def feed_fn():
        nonlocal placeholder, song
        return { placeholder: song[-SEQUENCE_LENGTH:] }
    model = tf.estimator.Estimator(model_fn, NETWORK_FOLDER, params=dict(batch_size=1))
    for i, res in enumerate(model.predict(input_fn, hooks=[tf.train.FeedFnHook(feed_fn)])):
        song.append(res['output'])
        if (i - SEQUENCE_LENGTH)%SAMPLE_RATE == 0:
            print((i - SEQUENCE_LENGTH)//SAMPLE_RATE, 'seconds')
        if i > SEQUENCE_LENGTH+SAMPLE_RATE*10:
            break
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    ffmpeg_write_audio(
        os.path.join(OUTPUT_FOLDER, 'wavenet_{:%y%m%d%H%M%S}.wav'.format(datetime.now())), 
        np.asarray(song[SEQUENCE_LENGTH:]))

if __name__ == "__main__":
    generate()
