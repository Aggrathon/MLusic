"""
    Generate a soundfile with the GAN
"""
import tensorflow as tf
from model import model_fn, SAMPLE_RATE

def input_fn():
    """
        Empty input_fn for estimator
    """
    return dict(input=None), dict()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    model = tf.estimator.Estimator(model_fn, 'network', params=dict(batch_size=1))
    for res in model.predict(input_fn):
        output = res['output']
        break
    with tf.Session() as sess:
        output = tf.convert_to_tensor(output)
        output = tf.contrib.ffmpeg.encode_audio(output, 'wav', SAMPLE_RATE)
        sess.run(tf.write_file('output.wav', output))
