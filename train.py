"""
    Train the AudioGAN
"""
import tensorflow as tf
from model import model_fn, input_fn

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    model = tf.estimator.Estimator(model_fn, 'network')
    model.train(input_fn, None, 10000)
