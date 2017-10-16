"""
    The AudioGAN neural network
"""
import tensorflow as tf
import sys

DATA_FOLDER = "data"
SAMPLE_RATE = 44100
SEQUENCE_LENGTH = 680639    #Based on the output from the convolutional transpose layers (~15s)
AUDIO_FORMAT = 'ogg'


def read_data(folder=DATA_FOLDER, sample=SAMPLE_RATE, length=SEQUENCE_LENGTH, format=AUDIO_FORMAT):
    """
        Get batched audio sequences from a folder
    """
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(folder+"/*."+format), capacity=1000)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    audio = tf.contrib.ffmpeg.decode_audio(value, format, sample, 1)
    slices = [tf.to_float(tf.random_crop(audio, (length, 1))) for _ in range(10)]
    batch = tf.train.shuffle_batch([slices], 32, 500, 100, 4, enqueue_many=True)
    return batch


def input_fn():
    """
        Training input to the estimator
    """
    return dict(input=read_data()), dict()


def model_fn(features, labels, mode):
    """
        The function that generates the network for the estimator
    """
    audio = features['input']
    batch_size = 1 if audio is None else audio.get_shape()[0].value
    with tf.variable_scope("generator"):
        prev_layer = tf.random_uniform((batch_size, 100), 0, 1, tf.float32)
        prev_layer = tf.layers.dense(prev_layer, 1000, tf.nn.relu)
        prev_layer = tf.reshape(prev_layer, (batch_size, 10, 1, 100))
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 112, (21, 1), (14, 1), activation=tf.nn.relu, name='conv_transpose_0')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 96, (13, 1), (8, 1), activation=tf.nn.relu, name='conv_transpose_1')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 80, (9, 1), (6, 1), activation=tf.nn.relu, name='conv_transpose_2')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 64, (7, 1), (4, 1), activation=tf.nn.relu, name='conv_transpose_3')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 48, (7, 1), (4, 1), activation=tf.nn.relu, name='conv_transpose_4')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 32, (5, 1), (3, 1), activation=tf.nn.relu, name='conv_transpose_5')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 1, (3, 1), (2, 1), activation=tf.nn.relu, name='conv_transpose_6') # MAYBE tanh
        prev_layer = tf.reshape(prev_layer, prev_layer.get_shape().as_list()[:3])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=dict(output=prev_layer))
    
    prev_layer = tf.concat((prev_layer, audio), 0)
    labels = [[0]]*batch_size + [[1]]*batch_size
    with tf.variable_scope('discriminator'):
        prev_layer = tf.reshape(prev_layer, prev_layer.get_shape().as_list()+[1])
        prev_layer = tf.layers.conv2d(prev_layer, 32, (5, 1), (2, 1), activation=tf.nn.relu, name='conv_0')
        prev_layer = tf.layers.conv2d(prev_layer, 40, (5, 1), (3, 1), activation=tf.nn.relu, name='conv_1')
        prev_layer = tf.layers.conv2d(prev_layer, 48, (7, 1), (3, 1), activation=tf.nn.relu, name='conv_2')
        prev_layer = tf.layers.conv2d(prev_layer, 56, (7, 1), (4, 1), activation=tf.nn.relu, name='conv_3')
        prev_layer = tf.layers.conv2d(prev_layer, 64, (9, 1), (4, 1), activation=tf.nn.relu, name='conv_4')
        prev_layer = tf.layers.conv2d(prev_layer, 72, (9, 1), (5, 1), activation=tf.nn.relu, name='conv_5')
        prev_layer = tf.contrib.layers.flatten(prev_layer)
        prev_layer = tf.layers.dense(prev_layer, 2048, activation=tf.nn.relu, name='fc_0')
        prev_layer = tf.layers.dense(prev_layer, 512, activation=tf.nn.relu, name='fc_1')
        prev_layer = tf.layers.dense(prev_layer, 64, activation=tf.nn.relu, name='fc_2')
        logits = tf.layers.dense(prev_layer, 1, name="logits")
        #predictions = tf.nn.sigmoid(logits, name="predictions")

    with tf.variable_scope('training'):
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        adam = tf.train.AdamOptimizer(1e-4)
        trainer = adam.minimize(loss, global_step=tf.contrib.framework.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=trainer)
