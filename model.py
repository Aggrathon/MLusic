"""
    The AudioGAN neural network
"""
import os
import tensorflow as tf

BATCH_SIZE = 32
LEARNING_RATE = 2e-5
SEQUENCE_LENGTH = 26743 #Based on the transpose convolutional layers (sequence_length/sample_rate ~= 1s)
VARIABLE_SIZE = 100
NETWORK_FOLDER = 'network'


def model_fn(features, labels, mode, params=dict()):
    """
        The function that generates the network for the estimator
    """
    batch_size = params.get('batch_size', BATCH_SIZE)
    learning_rate = params.get('learning_rate', LEARNING_RATE)
    sequence_length = params.get('sequence_length', SEQUENCE_LENGTH)
    variable_size = params.get('variable_size', VARIABLE_SIZE)
    training = mode == tf.estimator.ModeKeys.TRAIN
    layers = [
        (64, 9, 2),
        (64, 7, 2),
        (48, 7, 2),
        (40, 5, 2),
        (36, 5, 2),
        (32, 5, 2),
        (28, 5, 2),
        (24, 3, 2),
        (20, 3, 2)
    ]

    if mode != tf.estimator.ModeKeys.PREDICT:
        audio = tf.reshape(features['input'], (batch_size, sequence_length, 1, 1))
        with tf.variable_scope("encoder"):
            prev_layer = audio
            for i, l in enumerate(list(reversed(layers))):
                prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
                prev_layer = tf.layers.conv2d(prev_layer, l[0], (l[1], 1), (l[2], 1), activation=tf.nn.relu, name='conv_transpose_%d'%i)
                #print('enc', i, prev_layer.get_shape())
            prev_layer = tf.reshape(prev_layer, (batch_size, -1))
            var_stddv = tf.layers.dense(prev_layer, variable_size, tf.nn.tanh, name='var_stddv')
            var_mean = tf.layers.dense(prev_layer, variable_size, tf.nn.tanh, name='var_mean')
            encoding = tf.add(tf.random_normal([batch_size, variable_size], 0.0, 1.0, dtype=tf.float32)*var_stddv, var_mean, name="encoding")
    else:
        encoding = tf.random_normal((batch_size, variable_size), -1.0, 1.0, tf.float32, name='encoding')
    
    with tf.variable_scope("decoder"):
        prev_layer = tf.reshape(encoding, (batch_size, variable_size, 1, 1))
        for i, l in enumerate(layers[1:]):
            if i != 0:
                prev_layer = tf.layers.batch_normalization(prev_layer, training=training)
            prev_layer = tf.layers.conv2d_transpose(prev_layer, l[0], (l[1], 1), (l[2], 1), activation=tf.nn.relu, name='conv_transpose_%d'%i)
            #print('dec', i, prev_layer.get_shape())
        output = tf.layers.conv2d_transpose(prev_layer, 1, (1, 1), (1, 1), activation=tf.nn.tanh, name='output')    
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=dict(output=tf.reshape(output, (batch_size, -1))))

    with tf.variable_scope('training'):
        loss_mse = tf.losses.mean_squared_error(audio, output)
        loss_latent = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(var_mean) + tf.square(var_stddv) - tf.log(tf.square(var_stddv)) - 1.0, 1))
        loss = loss_latent*0.2 + loss_mse
        tf.summary.scalar('LossMSE', loss_mse)
        tf.summary.scalar('LossLatent', loss_latent)
        _, stddv = tf.nn.moments(output, 0)
        stddv = tf.reduce_mean(stddv)
        tf.summary.scalar('StandardDeviation', stddv)
        adam = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            trainer = adam.minimize(loss-stddv*0.5, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=trainer)
