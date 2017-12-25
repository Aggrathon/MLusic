"""
    The AudioGAN neural network
"""
import tensorflow as tf
from ops import dilated_casual_gated_convolution, one_hot_mu_law, argmax_mu_law

BATCH_SIZE = 64
LEARNING_RATE = 1e-5
SEQUENCE_LENGTH = 4096 #Based on the transpose convolutional layers (sequence_length/sample_rate ~= 1s)
MU = 256
NETWORK_FOLDER = 'network'


def model_fn(features, labels, mode, params=dict()):
    """
        The function that generates the network for the estimator
    """
    batch_size = params.get('batch_size', BATCH_SIZE)
    learning_rate = params.get('learning_rate', LEARNING_RATE)
    sequence_length = params.get('sequence_length', SEQUENCE_LENGTH)
    mu = params.get('mu', MU)
    training = mode == tf.estimator.ModeKeys.TRAIN
    audio = features['input']

    prev_layer = tf.reshape(audio, (batch_size, sequence_length, 1, 1))
    for i in range(12):
        prev_layer = dilated_casual_gated_convolution(prev_layer, 128, False, 'conv_dcg_%d'%i)
    print(prev_layer.get_shape())

    prev_layer = tf.layers.dense(prev_layer, 128, tf.nn.relu)
    logits = tf.layers.dense(prev_layer, mu, name='logits')
    prediction = tf.nn.softmax(logits)
    output = argmax_mu_law(prediction, mu)

    with tf.variable_scope('Loss'):
        labels = one_hot_mu_law(labels['output'], mu)
        loss = tf.losses.softmax_cross_entropy(labels, logits)

    with tf.variable_scope('training'):
        adam = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            trainer = adam.minimize(loss, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=trainer,
            predictions=dict(output=output))
