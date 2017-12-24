"""
    The AudioGAN neural network
"""
import os
import tensorflow as tf

BATCH_SIZE = 8
LEARNING_RATE = 1e-6
SEQUENCE_LENGTH = 4841 #Based on the transpose convolutional layers (sequence_length/sample_rate ~= 1s)
VARIABLE_SIZE = 200
NETWORK_FOLDER = 'network'


def encoder(input, training=True, variable_size=VARIABLE_SIZE, reuse=None):
    """
        Create an encoder for a variational autoencoder
    """
    with tf.variable_scope("Encoder"):
        prev_layer = input
        layers = [
            (32, 5, 3),
            (36, 5, 2),
            (40, 5, 2),
            (48, 3, 2),
            (52, 3, 2)
        ]
        for i, l in enumerate(layers):
            prev_layer = tf.layers.batch_normalization(prev_layer, training=training, reuse=reuse)
            prev_layer = tf.layers.conv2d(prev_layer, l[0], (l[1], 1), (l[2], 1), activation=tf.nn.relu, reuse=reuse, name='conv_%d'%i)
            print('enc', i, prev_layer.get_shape())
        shape = prev_layer.get_shape()
        prev_layer = tf.reshape(prev_layer, (shape[0], shape[1]*shape[3]))
        var_stddv = tf.layers.dense(prev_layer, 1000, tf.nn.relu, reuse=reuse)
        var_stddv = tf.layers.dense(var_stddv, variable_size, tf.nn.tanh, reuse=reuse, name='var_stddv')
        var_mean = tf.layers.dense(prev_layer, 1000, tf.nn.relu, reuse=reuse)
        var_mean = tf.layers.dense(var_mean, variable_size, tf.nn.tanh, reuse=reuse, name='var_mean')
        return var_mean, var_stddv

def decoder(input, training=True, reuse=None):
    """
        Create an decoder for a variational autoencoder
    """
    with tf.variable_scope("Decoder"):
        shape = input.get_shape()
        prev_layer = tf.reshape(input, (shape[0], shape[1], 1, 1))
        layers = [
            (48, 3, 2),
            (40, 5, 2),
            (36, 5, 2),
            (32, 5, 3)
        ]
        for i, l in enumerate(layers):
            if i != 0:
                prev_layer = tf.layers.batch_normalization(prev_layer, training=training, reuse=reuse)
            prev_layer = tf.layers.conv2d_transpose(prev_layer, l[0], (l[1], 1), (l[2], 1), activation=tf.nn.relu, reuse=reuse, name='conv_transpose_%d'%i)
            print('dec', i, prev_layer.get_shape())
        return tf.layers.conv2d_transpose(prev_layer, 1, (1, 1), (1, 1), activation=tf.nn.tanh, reuse=reuse, name='output')   

def autoencoder_loss(input, output, var_mean, var_stddv):
    """
        Create loss for a variational autoencoder
    """
    with tf.variable_scope('Loss'):
        # Standard Deviation Summaries
        _, stddv = tf.nn.moments(output, 0)
        stddv = tf.reduce_mean(stddv)
        tf.summary.scalar('OutputDeviation', stddv)
        _, stddv = tf.nn.moments(input, 0)
        stddv = tf.reduce_mean(stddv)
        tf.summary.scalar('InputDeviation', stddv)
        # Losses
        loss_mse = tf.losses.mean_squared_error(input, output)
        tf.summary.scalar('MSE', loss_mse)
        #input = 0.5+input*0.5
        #output = 0.5+output*0.5
        #loss_mse = -tf.reduce_mean(input * tf.log(output + 1e-8) + (1. - input) * tf.log((1. - output) + 1e-8))
        #tf.summary.scalar('Log', loss_mse)
        loss_latent = tf.reduce_mean(0.5 * tf.reduce_mean(tf.square(var_mean) + tf.square(var_stddv) - tf.log(tf.square(var_stddv)) - 1.0, 1))
        tf.summary.scalar('Latent', loss_latent)
        return loss_latent + loss_mse


def model_fn(features, labels, mode, params=dict()):
    """
        The function that generates the network for the estimator
    """
    batch_size = params.get('batch_size', BATCH_SIZE)
    learning_rate = params.get('learning_rate', LEARNING_RATE)
    sequence_length = params.get('sequence_length', SEQUENCE_LENGTH)
    variable_size = params.get('variable_size', VARIABLE_SIZE)
    training = mode == tf.estimator.ModeKeys.TRAIN
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        encoding = tf.random_normal((batch_size, variable_size), -1.0, 1.0, tf.float32, name='encoding')
        output = decoder(encoding, False, False) 
        output = tf.reshape(output, (batch_size, -1))
        return tf.estimator.EstimatorSpec(mode=mode, predictions=dict(output=output))

    audio = tf.reshape(features['input'], (batch_size, sequence_length, 1, 1))
    var_mean, var_stddv = encoder(audio, training, variable_size, False)
    encoding = tf.add(tf.random_normal([batch_size, variable_size], 0.0, 1.0, dtype=tf.float32)*var_stddv, var_mean, name="encoding")
    output = decoder(encoding, training, False)
    loss = autoencoder_loss(audio, output, var_mean, var_stddv)

    with tf.variable_scope('training'):
        adam = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            trainer = adam.minimize(loss, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=trainer)
