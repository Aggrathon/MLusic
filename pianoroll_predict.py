"""
    Transformer for predicting pianorolls from the lpd dataset
"""
import sys
from time import time as timer
import tensorflow as tf
from lpd import read, write
from transformer import DecodingTransformer, look_ahead_mask
from keras_radam.training import RAdamOptimizer


SEQUENCE_LENGTH = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
CHECKPOINT_PATH = "./network/pianoroll_predict"


@tf.function()
def train_step(data, transformer, optimiser):
    mask = look_ahead_mask(tf.shape(data)[1]-1)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(data[:, :-1, :], True, mask)
        predictions = tf.nn.sigmoid(predictions)
        loss = tf.keras.losses.mean_squared_error(data[:, 33:, :], predictions[:, 32:, :])
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimiser.apply_gradients(zip(gradients, transformer.trainable_variables))
    return tf.reduce_sum(loss, -1)

def train():
    tra = DecodingTransformer(4, 128, 8, 512, 128, 0.15, False)
    opt = RAdamOptimizer(LEARNING_RATE, 0.8, 0.95)
    checkpoint = tf.train.Checkpoint(tra=tra, opt=opt)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint)
    loss = tf.keras.metrics.Mean(name='loss')
    data = enumerate(read())
    try:
        print("Preparing training")
        next(data)
        starttime = timer()
        chpinterval = 600
        chptime = starttime + chpinterval
        logtime = timer()
        print("Starting training")
        for i, dat in data:
            loss(train_step(dat, tra, opt))
            if i % 100 == 0:
                print("Step: {:6d}      Loss: {:6.3f}      Time: {:5.2f}".format(
                    i, loss.result(), timer() - logtime))
                if chptime < timer():
                    print("Saving checkpoint (%d min)"%int((timer() - starttime)/60))
                    manager.save()
                    chptime += chpinterval
                loss.reset_states()
                logtime = timer()
    except KeyboardInterrupt:
        print("Saving checkpoint")
        manager.save()


@tf.function()
def _infer(data, tra):
    mask = look_ahead_mask(tf.shape(data)[1])
    tmp, _ = tra(data, False, mask)
    tmp = tf.nn.sigmoid(tmp)[:, -1, :]
    return tmp

def generate(length=SEQUENCE_LENGTH*2):
    """
    Generate midi with the model
    """
    tra = DecodingTransformer(4, 128, 8, 512, 128, 0.15, False)
    checkpoint = tf.train.Checkpoint(tra=tra)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("No checkpoint to load")
        return
    data = next(iter(read(SEQUENCE_LENGTH + length, 1)))
    data = data.numpy()
    for i in range(SEQUENCE_LENGTH, SEQUENCE_LENGTH + length):
        data[:, i, :] = _infer(data[:, (i-SEQUENCE_LENGTH):i, :], tra).numpy()
    # import numpy
    # print(numpy.sum(data[0, SEQUENCE_LENGTH:, :], -1))
    write(data[0, :, :]*127)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == "generate":
        generate()
    else:
        print("You must specify what you want to do (train / generate)")
