
import sys
from time import time as timer
import tensorflow as tf
from keras_radam.training import RAdamOptimizer
from transformer import DecodingTransformer, look_ahead_mask
from lpd import read, write, _infer

SEQUENCE_LENGTH = 512
BATCH_SIZE = 48
LEARNING_RATE = 1e-6
CHECKPOINT_PATH = "./network/transformer_midi_gan"


class Generator(tf.keras.Model):
    def __init__(self, batch=BATCH_SIZE, seq_len=SEQUENCE_LENGTH, num_layers=4, d_model=64,
                 num_heads=8, dff=256, input_shape=64, output_size=128, rate=0.1, relative=True):
        super(Generator, self).__init__()
        self.seed_shape = (batch, input_shape)
        self.embedding = tf.keras.layers.Dense(d_model//2 * seq_len, activation=tf.nn.leaky_relu)
        self.transformer = DecodingTransformer(num_layers, d_model, num_heads, dff, output_size, rate, relative)
        self.seq_len = seq_len
        self.d_model = d_model

    def seed(self):
        return tf.random.normal(shape=self.seed_shape)

    def call(self, rnd, training, la_mask):
        inp = self.embedding(rnd)
        inp = tf.reshape(inp, (-1, self.seq_len, self.d_model//2))
        return self.transformer(inp, training, la_mask)


@tf.function()
def train_step(data, generator, discriminator, gen_optimiser, dis_optimiser):
    mask = look_ahead_mask(tf.shape(data)[1])
    rnd = generator.seed()
    with tf.GradientTape(persistent=True) as tape:
        fake, _ = generator(rnd, True, mask)
        fake = tf.nn.sigmoid(fake)
        fake, _ = discriminator(fake, True, mask)
        real, _ = discriminator(data, True, mask)
        fake = fake[:, 32:]
        real = real[:, 32:]
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
    dis_grad = tape.gradient(lsdi, discriminator.trainable_variables)
    dis_optimiser.apply_gradients(zip(dis_grad, discriminator.trainable_variables))
    balance = tf.cond(lsgm > lsrm, gen_upd, lambda: 0)
    del tape
    return lsgm, lsdi, balance

def train():
    gen = Generator(BATCH_SIZE, SEQUENCE_LENGTH, 4, 64, 8, 256, 128, 128, 0.1, False)
    dis = DecodingTransformer(4, 64, 8, 256, 1, 0.1, False)
    gen_opt = RAdamOptimizer(LEARNING_RATE, 0.5, 0.9,)
    dis_opt = RAdamOptimizer(LEARNING_RATE, 0.5, 0.9)
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
    data = enumerate(read(SEQUENCE_LENGTH, BATCH_SIZE))
    try:
        print("Preparing training")
        next(data)
        print("Starting training")
        start = timer()
        for i, dat in data:
            loss(train_step(dat, gen, dis, gen_opt, dis_opt))
            if i % 50 == 0:
                print("Step: {:6d}      Loss: {:6.4f} {:6.4f}      Time: {:6.2f}".format(
                    i, loss_gen.result(), loss_dis.result(), timer() - start))
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
def _infer(generator):
    rnd = generator.seed()
    mask = look_ahead_mask(SEQUENCE_LENGTH)
    tmp, _ = generator(rnd, False, mask)
    tmp = tf.nn.sigmoid(tmp)
    return tmp


def generate(num_songs=10):
    """
    Generate midi with the model
    """
    gen = Generator(num_songs, SEQUENCE_LENGTH, 4, 64, 8, 256, 128, 128, 0.1, False)
    checkpoint = tf.train.Checkpoint(gen=gen)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("No checkpoint to load")
        return
    data = _infer(gen).numpy()
    for j in range(num_songs):
        write(data[j, :, :]*127)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == "generate":
        generate()
    else:
        print("You must specify what you want to do (train / generate)")
