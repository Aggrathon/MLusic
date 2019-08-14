
import sys
from time import time as timer
import tensorflow as tf
from transformer import Transformer, look_ahead_mask
from lpd import read, write, _infer

SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "./network/transformer_lpd_gan"

@tf.function()
def train_step(data, generator, discriminator, gen_optimiser, dis_optimiser):
    mask = look_ahead_mask(tf.shape(data)[1]-1)
    with tf.GradientTape(persistent=True) as tape:
        fake, _ = generator(data[:, :-1, :], data[:, :-1, :], True, None, mask, None)
        #loss_sup = tf.nn.softmax_cross_entropy_with_logits(data[:, 1:, :], fake)
        fake = tf.nn.sigmoid(fake)
        # fake = fake / tf.maximum(0.5, tf.reduce_max(fake, -1, True))
        fake, _ = discriminator(data[:, :-1, :], fake, True, None, mask, None)
        real, _ = discriminator(data[:, :-1, :], data[:, 1:, :], True, None, mask, None)
        loss_gen = tf.losses.binary_crossentropy(tf.ones_like(fake), fake, True, 0.1)
        loss_fake = tf.losses.binary_crossentropy(tf.zeros_like(fake), fake, True, 0.1)
        loss_real = tf.losses.binary_crossentropy(tf.ones_like(real), real, True, 0.1)
        lsgm = tf.reduce_mean(loss_gen)
        lsrm = tf.reduce_mean(loss_real)
        loss_comb_gen = lsgm# + tf.reduce_mean(loss_sup)
        loss_comb_dis = tf.reduce_mean(loss_fake) + lsrm
    def gen_upd():
        gen_grad = tape.gradient(loss_comb_gen, generator.trainable_variables)
        gen_optimiser.apply_gradients(zip(gen_grad, generator.trainable_variables))
        return 1
    dis_grad = tape.gradient(loss_comb_dis, discriminator.trainable_variables)
    dis_optimiser.apply_gradients(zip(dis_grad, discriminator.trainable_variables))
    balance = tf.cond(lsgm > lsrm, gen_upd, lambda: 0)
    del tape
    return loss_comb_gen, loss_comb_dis, balance

def train():
    gen = Transformer(4, 64, 8, 256, 128, 128, 0.1, False)
    dis = Transformer(4, 64, 8, 256, 128, 1, 0.1, False)
    gen_opt = tf.keras.optimizers.Adam(LEARNING_RATE, 0.5, 0.9)
    dis_opt = tf.keras.optimizers.Adam(LEARNING_RATE, 0.5, 0.9)
    checkpoint = tf.train.Checkpoint(gen=gen, dis=dis, gen_opt=gen_opt, dis_opt=dis_opt)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint)
    loss_gen = tf.keras.metrics.Mean(name='loss_gen')
    loss_dis = tf.keras.metrics.Mean(name='loss_dis')
    def loss(l):
        loss_gen(l[0])
        loss_dis(l[1])
    data = enumerate(read(SEQUENCE_LENGTH+1, BATCH_SIZE))
    try:
        print("Preparing training")
        loss(train_step(next(data)[1], gen, dis, gen_opt, dis_opt))
        print("Starting training")
        start = timer()
        for i, dat in data:
            loss(train_step(dat, gen, dis, gen_opt, dis_opt))
            if i % 50 == 0:
                print("Step: {:6d}      Loss: {:6.4f} {:6.4f}      Time: {:6.2f}".format(i, loss_gen.result(), loss_dis.result(), timer() - start))
                if i % 500 == 0:
                    print("Saving checkpoint")
                    manager.save()
                    loss_gen.reset_states()
                    loss_dis.reset_states()
                start = timer()
    except KeyboardInterrupt:
        print("Saving checkpoint")
        manager.save()


@tf.function()
def _infer(data, tra):
    mask = look_ahead_mask(tf.shape(data)[1])
    tmp, _ = tra(data, data, False, mask, mask, mask)
    tmp = tf.nn.sigmoid(tmp)[:, -1, :]
    # tmp = tmp / tf.maximum(tf.reduce_max(tmp), 0.5)
    return tmp


def generate(length=SEQUENCE_LENGTH*3):
    """
    Generate midi with the model
    """
    tra = Transformer(4, 64, 8, 256, 128, 128, 0.1, False)
    checkpoint = tf.train.Checkpoint(gen=tra)
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
    write(data[0, :, :]*127)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == "generate":
        generate()
    else:
        print("You must specify what you want to do (train / generate)")
