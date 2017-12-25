"""
    Neural Network ops for various models
"""
import tensorflow as tf

#region: wavenet

def dilated_casual_gated_convolution(input, filters, reuse=None, name='conv_dcg'):
    """
        Convolution for wavenet (simple implementation)
    """
    tanh = tf.layers.conv2d(input, filters, (2, 1), (2, 1), 'valid', activation=tf.nn.tanh, reuse=reuse, name=name+'_filter')
    gate = tf.layers.conv2d(input, filters, (2, 1), (2, 1), 'valid', activation=tf.nn.sigmoid, reuse=reuse, name=name+'_gate')
    return tanh*gate

def one_hot_mu_law(tensor, mu=256):
    """
    Get the onehot encoded tensor using mu-law scaling
    """
    with tf.name_scope('one_hot_mu_law'):
        mu1 = tf.to_float(mu - 1)
        y = tf.sign(tensor)*tf.log1p(mu1*tf.abs(tensor))/tf.log1p(mu1)
        yt = tf.to_int32((y*0.5+0.5)*mu1)
        one_hot = tf.one_hot(yt, mu, 0.9, 0.1)
        return one_hot

def argmax_mu_law(tensor, mu=256):
    """
    Get the argmax tensor using mu-law scaling
    """
    with tf.name_scope('argmax_mu_law'):
        x = tf.argmax(tensor, 1, output_type=tf.float32)
        mu1 = tf.to_float(mu - 1)
        xt = x/mu1*2.0-1.0
        y = (1.0 / mu1) * ((1.0 + mu1)**tf.abs(xt) - 1.0)
        return tf.sign(xt)*y


#endregion

#region: variational autoencoder

def encoder(input, training=True, variable_size=100, reuse=None):
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

#endregion: variational autoencoder