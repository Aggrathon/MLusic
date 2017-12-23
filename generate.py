"""
    Generate a soundfile with the GAN
"""
import os
import tensorflow as tf
from model import model_fn, NETWORK_FOLDER
from data import ffmpeg_write_audio, AUDIO_FORMAT

OUTPUT_FOLDER = 'output'

def input_fn():
    """
        Empty input_fn for estimator
    """
    return dict(input=None), dict()

def generate():
    tf.logging.set_verbosity(tf.logging.INFO)
    model = tf.estimator.Estimator(model_fn, NETWORK_FOLDER, params=dict(batch_size=1))
    for res in model.predict(input_fn):
        output = res['output']
        break
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    ffmpeg_write_audio(os.path.join(OUTPUT_FOLDER, 'data.wav'), output)

if __name__ == "__main__":
    generate()
