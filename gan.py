
import sys
from time import time as timer
import tensorflow as tf
from transformer import Transformer, Encoder, _input_tf, _output_to_int, _input_max_ins, _output

SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE_DIS = 1e-4
LEARNING_RATE_GEN = 1e-4
TIME_MULTIPLIER = 10
CHECKPOINT_PATH = "./network/transformer_gan"


class Generator(tf.keras.Model):
    def __init__(self, batch=BATCH_SIZE, seq_len=SEQUENCE_LENGTH, num_layers=4, d_model=128,
                 num_heads=8, dff=512, input_shape=16, output_size=30, rate=0.1):
        super(Generator, self).__init__()
        self.seed_shape = (batch, seq_len, input_shape)
        self.embedding_in = tf.keras.layers.Dense(d_model//2, activation=tf.nn.leaky_relu)
        self.embedding_out = tf.keras.layers.Dense(d_model//2, activation=tf.nn.leaky_relu)
        self.transformer = Transformer(seq_len, num_layers, d_model, num_heads, dff, output_size, rate)

    def seed(self):
        return tf.random.normal(shape=self.seed_shape)

    def call(self, rnd, training):
        inp = self.embedding_in(rnd)
        tar = self.embedding_out(rnd)
        output, _ = self.transformer(inp, tar, training, None, None, None)
        return output

class Discriminator(tf.keras.Model):
    def __init__(self, seq_len=SEQUENCE_LENGTH, num_layers=6, d_model=128, num_heads=8, dff=512, rate=0.1):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(seq_len, num_layers, d_model, num_heads, dff, rate)
        self.embedding = tf.keras.layers.Dense(d_model)
        self.logit = tf.keras.layers.Dense(1)
        self.flat = tf.keras.layers.Flatten()

    def call(self, inp, training):
        inp = self.embedding(inp)
        enc = self.encoder(inp, training, None)
        out = self.logit(enc)
        out = self.flat(out)
        return tf.reduce_mean(out, -1)

def _clean_gen_out(output):
    time = tf.nn.relu(output[:, :, 0])
    instrument = tf.nn.softmax(tf.nn.leaky_relu(output[:, :, 1:-2]))
    note = output[:, :, -2]
    state = tf.nn.sigmoid(output[:, :, -1])
    return time, instrument, note, state

def _dirty_real_input(time, instrument, note, state, num_ins=20):
    time = time * tf.random.normal(shape=tf.shape(time), mean=TIME_MULTIPLIER, stddev=0.5)
    instrument = tf.one_hot(instrument, num_ins, dtype=tf.float32)
    instrument = instrument + tf.random.normal(shape=tf.shape(instrument), stddev=0.1)
    instrument = tf.nn.softmax(instrument)
    note = note + tf.random.normal(shape=tf.shape(note), stddev=0.1)
    state = tf.nn.softmax((state * 3.0 - 1.5) + tf.random.normal(shape=tf.shape(state), stddev=0.2))
    return time, instrument, note, state

def _concat_inputs(time, instrument, note, state):
    return tf.concat((
        tf.expand_dims(time, -1),
        instrument,
        tf.expand_dims(note, -1),
        tf.expand_dims(state, -1)), -1)

@tf.function()
def _train_step(time, instrument, note, state, gen, dis, gen_opt, dis_opt, max_ins=129):
    real = _concat_inputs(*_dirty_real_input(time, instrument, note, state, max_ins))
    fake = gen.seed()
    with tf.GradientTape(persistent=True) as tape:
        fake = gen(fake, True)
        fake = _concat_inputs(*_clean_gen_out(fake))
        fake = dis(fake, True)
        real = dis(real, True)
        loss1 = tf.losses.binary_crossentropy(tf.ones_like(fake), fake, True, 0.1)
        loss2 = tf.losses.binary_crossentropy(tf.zeros_like(fake), fake, True, 0)
        loss3 = tf.losses.binary_crossentropy(tf.ones_like(real), real, True, 0.1)
        loss4 = loss2 + loss3
    # mult = tf.cond(loss1 + offset < loss4 * 0.5, lambda: 0.1, lambda: 1.0)
    # gen_grad = [g * mult for g in gen_grad]
    # mult = tf.cond(loss1 + offset > loss4, lambda: 0.1, lambda: 1.0)
    # dis_grad = [g * mult for g in dis_grad]
    def gen_upd():
        gen_grad = tape.gradient(loss1, gen.trainable_variables)
        gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))
        return -1
    def dis_upd():
        dis_grad = tape.gradient(loss4, dis.trainable_variables)
        dis_opt.apply_gradients(zip(dis_grad, dis.trainable_variables))
        return 1
    bal1 = tf.cond(loss1 < loss4, dis_upd, lambda: 0)
    bal2 = tf.cond(loss1 > tf.maximum(loss2, loss3), gen_upd, lambda: 0)
    del tape
    return loss1, loss2, loss3, bal1 + bal2


def train():
    """
    Train the model
    """
    print("Setting up training")
    data = _input_tf(sequence=SEQUENCE_LENGTH, batch=BATCH_SIZE)
    ins = _input_max_ins()
    vec_size = 3 + ins
    gen = Generator(output_size=vec_size)
    dis = Discriminator()
    gen_opt = tf.keras.optimizers.Adam(LEARNING_RATE_GEN, 0.8, 0.95)
    dis_opt = tf.keras.optimizers.Adam(LEARNING_RATE_DIS, 0.8, 0.95)
    
    checkpoint = tf.train.Checkpoint(gen=gen, dis=dis, gen_opt=gen_opt, dis_opt=dis_opt)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint)
    gen_loss = tf.keras.metrics.Mean(name='loss_gen')
    opt_loss_r = tf.keras.metrics.Mean(name='loss_opt_r')
    opt_loss_f = tf.keras.metrics.Mean(name='loss_opt_f')
    data = enumerate(data)
    try:
        print("Preparing training")
        _train_step(*(next(data)[1]), gen, dis, gen_opt, dis_opt, ins)
        print("Starting training")
        start = timer()
        for (i, (time, instrument, note, state)) in data:
            loss = _train_step(time, instrument, note, state, gen, dis, gen_opt, dis_opt, ins)
            gen_loss(loss[0])
            opt_loss_r(loss[1])
            opt_loss_f(loss[2])
            if i % 50 == 0:
                print("Batch: {:6d}      Loss: {:4.2f} {:4.2f} {:4.2f}      Time: {:6.2f}".format(
                    i,
                    gen_loss.result(),
                    opt_loss_r.result(),
                    opt_loss_f.result(),
                    timer() - start))
                if i % 500 == 0:
                    print("Saving checkpoint")
                    manager.save()
                    gen_loss.reset_states()
                    opt_loss_r.reset_states()
                    opt_loss_f.reset_states()
                start = timer()
    except KeyboardInterrupt:
        print("Saving checkpoint")
        manager.save()

def _flat_all(*args):
    return tuple(tf.reshape(i, [tf.reduce_prod(tf.shape(i))]) for i in args)

@tf.function()
def _infer(gen):
    out = _flat_all(*_output_to_int(gen(gen.seed(), False)))
    out = (out[0]/TIME_MULTIPLIER, *out[1:])
    return out

def generate():
    """
    Generate midi with the model
    """
    ins = 3 + _input_max_ins()
    gen = Generator(output_size=ins, batch=1)
    checkpoint = tf.train.Checkpoint(gen=gen)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("No checkpoint to load")
        return
    _output(*_infer(gen))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == "generate":
        generate()
    else:
        print("You must specify what you want to do (train / generate)")
