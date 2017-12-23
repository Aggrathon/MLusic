"""
    The AudioGAN neural network
"""
import os
import tensorflow as tf

DATA_FOLDER = "data"
SAMPLE_RATE = 44100
AUDIO_FORMAT = 'ogg'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 648519 #Based on the transpose convolutional layers (~15s)


def read_data(folder=DATA_FOLDER, sample=SAMPLE_RATE, format=AUDIO_FORMAT):
    """
        Get a combined audio sequence from a folder
    """
    files = []
    for name in os.listdir(folder):
        if name[-3:] == format:
            data = tf.read_file(os.path.join(folder, name))
            audio = tf.contrib.ffmpeg.decode_audio(data, format, sample, 1)
            files.append(audio)
    with tf.Session() as sess:
        return tf.concat(files, 0).eval(session=sess)

def input_fn():
    """
        Basic input to the Estimator
    """
    data = tf.convert_to_tensor(read_data(), dtype=tf.float32, name='should_not_be_saved_in_network')
    rnd = tf.stack([tf.random_crop(data, (SEQUENCE_LENGTH, 1)) for _ in range(BATCH_SIZE)])
    return dict(input=rnd), dict()

def model_fn(features, labels, mode, params=dict()):
    """
        The function that generates the network for the estimator
    """
    audio = features['input']
    batch_size = params.get('batch_size', BATCH_SIZE)
    learning_rate = params.get('learning_rate', LEARNING_RATE)
    with tf.variable_scope("generator"):
        prev_layer = tf.random_uniform((batch_size, 100), 0, 1, tf.float32)
        prev_layer = tf.layers.dense(prev_layer, 1000, tf.nn.relu)
        prev_layer = tf.layers.dense(prev_layer, 2000, tf.nn.relu)
        prev_layer = tf.reshape(prev_layer, (batch_size, 20, 1, 100))
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 80, (33, 1), (16, 1), activation=tf.nn.relu, name='conv_transpose_0')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 64, (17, 1), (10, 1), activation=tf.nn.relu, name='conv_transpose_1')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 48, (13, 1), (8, 1), activation=tf.nn.relu, name='conv_transpose_2')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 32, (9, 1), (6, 1), activation=tf.nn.relu, name='conv_transpose_3')
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 1, (7, 1), (4, 1), activation=tf.nn.relu, name='conv_transpose_4') # MAYBE tanh
        prev_layer = tf.reshape(prev_layer, prev_layer.get_shape().as_list()[:3])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=dict(output=prev_layer))
    
    prev_layer = tf.concat((prev_layer, audio), 0)
    labels = tf.convert_to_tensor([[0.05]]*batch_size + [[0.95]]*batch_size)
    with tf.variable_scope('discriminator'):
        prev_layer = tf.reshape(prev_layer, prev_layer.get_shape().as_list()+[1])
        prev_layer = tf.layers.conv2d(prev_layer, 40, (9, 1), (6, 1), activation=tf.nn.relu, name='conv_0')
        prev_layer = tf.layers.conv2d(prev_layer, 48, (9, 1), (6, 1), activation=tf.nn.relu, name='conv_1')
        prev_layer = tf.layers.conv2d(prev_layer, 56, (9, 1), (6, 1), activation=tf.nn.relu, name='conv_2')
        prev_layer = tf.layers.conv2d(prev_layer, 64, (13, 1), (8, 1), activation=tf.nn.relu, name='conv_3')
        prev_layer = tf.layers.conv2d(prev_layer, 72, (15, 1), (10, 1), activation=tf.nn.relu, name='conv_4')
        prev_layer = tf.contrib.layers.flatten(prev_layer)
        prev_layer = tf.layers.dense(prev_layer, 1024, activation=tf.nn.relu, name='fc_0')
        prev_layer = tf.layers.dense(prev_layer, 256, activation=tf.nn.relu, name='fc_1')
        prev_layer = tf.layers.dense(prev_layer, 64, activation=tf.nn.relu, name='fc_2')
        logits = tf.layers.dense(prev_layer, 1, name="logits")
        #predictions = tf.nn.sigmoid(logits, name="predictions")

    with tf.variable_scope('training'):
        loss_disc = tf.losses.sigmoid_cross_entropy(labels, logits)
        loss_gen = tf.losses.sigmoid_cross_entropy(1.0-labels, logits)
        adam = tf.train.AdamOptimizer(learning_rate)
        trainer_disc = adam.minimize(loss_disc, global_step=tf.contrib.framework.get_global_step(), var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
        trainer_gen = adam.minimize(loss_gen, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss_disc+loss_gen,
            train_op=tf.group(trainer_disc, trainer_gen))
