"""
    Transformer for predicting pianorolls from the lpd dataset (with a discriminator)
"""
import sys
from time import time as timer
import tensorflow as tf
from lpd import read, write
from transformer import DecodingTransformer, look_ahead_mask, positional_encoding
from keras_radam.training import RAdamOptimizer


SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "./network/pianoroll_hybrid"


class Discriminator(DecodingTransformer):
    def build(self, input_shape):
        if not self.relative:
            pos_enc = positional_encoding(input_shape[1]//2+1, self.d_model)
            self.pos_enc = tf.concat((pos_enc[:, :-1, :], pos_enc[:, 1:, :]), 1)

    def look_ahead_mask(self, size):
        mask_std = look_ahead_mask(size)
        mask_prd = 1 - tf.linalg.eye(size)
        return tf.concat((tf.concat((mask_std, mask_prd), 1), tf.zeros((size, 2*size))), 0)

@tf.function()
def train_step(data, generator: DecodingTransformer, discriminator: Discriminator, gen_optimiser, dis_optimiser):
    seq_len = tf.shape(data)[1]-1
    mask = look_ahead_mask(seq_len)
    mask_dis = discriminator.look_ahead_mask(seq_len)
    with tf.GradientTape(persistent=True) as tape:
        predictions, _ = generator(data[:, :-1, :], True, mask)
        predictions = tf.nn.sigmoid(predictions)
        fake, _ = discriminator(tf.concat((data[:, :-1, :], predictions), 1), True, mask_dis)
        real, _ = discriminator(tf.concat((data[:, :-1, :], data[:, 1:, :]), 1), True, mask_dis)
        fake = fake[:, 32:seq_len]
        real = real[:, 32:seq_len]
        loss_gen = tf.losses.binary_crossentropy(tf.ones_like(fake), fake, True, 0.1)
        loss_fake = tf.losses.binary_crossentropy(tf.zeros_like(fake), fake, True, 0.1)
        loss_real = tf.losses.binary_crossentropy(tf.ones_like(real), real, True, 0.1)
        lsgm = tf.reduce_mean(loss_gen)
        lsrm = tf.reduce_mean(loss_real)
        lsdi = tf.reduce_mean(loss_fake) + lsrm
    def gen_upd():
        gen_grad = tape.gradient(lsgm, generator.trainable_variables)
        gen_optimiser.apply_gradients(zip(gen_grad, generator.trainable_variables))
        return 1
    balance = tf.cond(lsgm > lsrm, gen_upd, lambda: 0)
    def dis_upd():
        dis_grad = tape.gradient(lsdi, discriminator.trainable_variables)
        dis_optimiser.apply_gradients(zip(dis_grad, discriminator.trainable_variables))
        return -1
    balance += tf.cond(tf.reduce_mean(tf.sigmoid(fake)) > 0.11, dis_upd, lambda: 0)
    del tape
    return lsgm, lsdi, balance

def train():
    gen = DecodingTransformer(4, 64, 8, 256, 128, 0.1, False)
    dis = Discriminator(4, 64, 8, 256, 1, 0.1, False)
    gen_opt = RAdamOptimizer(LEARNING_RATE, 0.75, 0.95)
    dis_opt = RAdamOptimizer(LEARNING_RATE, 0.75, 0.95)
    checkpoint = tf.train.Checkpoint(gen=gen, dis=dis, gen_opt=gen_opt, dis_opt=dis_opt)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint)
    loss_gen = tf.keras.metrics.Mean(name='loss_gen')
    loss_dis = tf.keras.metrics.Mean(name='loss_dis')
    balance = tf.keras.metrics.Mean(name='balance')
    def loss(loss):
        loss_gen(loss[0])
        loss_dis(loss[1])
        balance(loss[2])
    data = enumerate(read(SEQUENCE_LENGTH + 1, BATCH_SIZE))
    try:
        print("Preparing training")
        next(data)
        starttime = timer()
        chpinterval = 600
        chptime = starttime + chpinterval
        logtime = timer()
        print("Starting training")
        for i, dat in data:
            loss(train_step(dat, gen, dis, gen_opt, dis_opt))
            if i % 100 == 0:
                if i == 100:
                    print("\n                        Genera Discri Balan")
                print("Step: {:6d}      Loss: {:6.4f} {:6.4f} {:5.2f}      Time: {:6.2f}".format(
                    i, loss_gen.result(), loss_dis.result(), balance.result(), timer() - logtime))
                if chptime < timer():
                    print("Saving checkpoint (%d min)"%int((timer() - starttime)/60))
                    manager.save()
                    chptime += chpinterval
                loss_gen.reset_states()
                loss_dis.reset_states()
                balance.reset_states()
                logtime = timer()
    except KeyboardInterrupt:
        print("Saving checkpoint (%d min)"%int((timer() - starttime)/60))
        manager.save()


@tf.function()
def _infer(data, tra):
    mask = look_ahead_mask(tf.shape(data)[1])
    tmp, _ = tra(data, False, mask)
    tmp = tf.nn.sigmoid(tmp)[:, -1, :]
    return tmp

def generate(length=SEQUENCE_LENGTH*4):
    """
    Generate midi with the model
    """
    gen = DecodingTransformer(4, 64, 8, 256, 128, 0.1, False)
    checkpoint = tf.train.Checkpoint(gen=gen)
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
        data[:, i, :] = _infer(data[:, (i-SEQUENCE_LENGTH):i, :], gen).numpy()
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
