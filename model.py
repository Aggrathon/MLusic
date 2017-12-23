"""
    The AudioGAN neural network
"""
import os
import tensorflow as tf

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 648519 #Based on the transpose convolutional layers (~15s)
NETWORK_FOLDER = 'network'


def model_fn(features, labels, mode, params=dict()):
    """
        The function that generates the network for the estimator
    """
    audio = features['input']
    batch_size = params.get('batch_size', BATCH_SIZE)
    learning_rate = params.get('learning_rate', LEARNING_RATE)
    sequence_length = params.get('sequence_length', SEQUENCE_LENGTH)
    training = mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope("generator"):
        prev_layer = tf.random_uniform((batch_size, 100), 0, 1, tf.float32)
        prev_layer = tf.layers.dense(prev_layer, 1000, tf.nn.relu)
        prev_layer = tf.layers.dense(prev_layer, 2000, tf.nn.relu)
        prev_layer = tf.reshape(prev_layer, (batch_size, 20, 1, 100))
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 80, (33, 1), (16, 1), activation=tf.nn.relu, name='conv_transpose_0')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 64, (17, 1), (10, 1), activation=tf.nn.relu, name='conv_transpose_1')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 48, (13, 1), (8, 1), activation=tf.nn.relu, name='conv_transpose_2')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 32, (9, 1), (6, 1), activation=tf.nn.relu, name='conv_transpose_3')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d_transpose(prev_layer, 1, (7, 1), (4, 1), activation=tf.nn.tanh, name='conv_transpose_4')
        output = tf.reshape(prev_layer, (batch_size, sequence_length))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=dict(output=output))
    
    prev_layer = tf.concat((output, audio), 0)
    labels = tf.convert_to_tensor([[0.1]]*batch_size + [[0.9]]*batch_size)
    with tf.variable_scope('discriminator'):
        prev_layer = tf.reshape(prev_layer, (batch_size*2, sequence_length, 1, 1))
        prev_layer = tf.layers.conv2d(prev_layer, 40, (9, 1), (6, 1), activation=tf.nn.relu, name='conv_0')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d(prev_layer, 48, (9, 1), (6, 1), activation=tf.nn.relu, name='conv_1')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d(prev_layer, 56, (9, 1), (6, 1), activation=tf.nn.relu, name='conv_2')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d(prev_layer, 64, (13, 1), (8, 1), activation=tf.nn.relu, name='conv_3')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.layers.conv2d(prev_layer, 72, (15, 1), (10, 1), activation=tf.nn.relu, name='conv_4')
        prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
        prev_layer = tf.contrib.layers.flatten(prev_layer)
        prev_layer = tf.layers.dense(prev_layer, 1024, activation=tf.nn.relu, name='fc_0')
        prev_layer = tf.layers.dense(prev_layer, 256, activation=tf.nn.relu, name='fc_1')
        prev_layer = tf.layers.dense(prev_layer, 64, activation=tf.nn.relu, name='fc_2')
        logits = tf.layers.dense(prev_layer, 1, name="logits")

    with tf.variable_scope('training'):
        loss_disc = tf.losses.sigmoid_cross_entropy(labels, logits)
        loss_gen = tf.losses.sigmoid_cross_entropy(1.0-labels, logits)
        loss_sim = tf.reduce_mean(tf.abs(0.5 - tf.abs(output-tf.reduce_mean(output, 0, True))))
        tf.summary.scalar("DiscriminatorLoss", loss_disc)
        tf.summary.scalar("GeneratorLoss", loss_gen)
        tf.summary.scalar("SimilarityLoss", loss_sim)
        adam = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            trainer_disc = adam.minimize(loss_disc, global_step=tf.train.get_global_step(), var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
            trainer_gen = adam.minimize(loss_gen*(1.0-loss_disc)**2+loss_sim*0.05, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss_disc+loss_gen,
            train_op=tf.group(trainer_disc, trainer_gen))
