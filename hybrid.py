
import sys
from time import time as timer
import numpy as np
import tensorflow as tf
from transformer import _mask, Transformer, Encoder, SEQUENCE_LENGTH
from gan import BATCH_SIZE
from utils import AddNoiseToInput, CleanOutput

LEARNING_RATE = 1e-5
CHECKPOINT_PATH = "./network/transformer_hybrid"

class Discriminator(tf.keras.Model):
    def __init__(self, seq_len=SEQUENCE_LENGTH, num_layers=6, d_model=128, num_heads=8, dff=512, rate=0.1):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(seq_len, num_layers, d_model, num_heads, dff, rate)
        self.embedding = tf.keras.layers.Dense(d_model)
        self.logit = tf.keras.layers.Dense(1)

    def call(self, inp, training, mask):
        inp = self.embedding(inp)
        enc = self.encoder(inp, training, mask)
        return self.logit(enc)

@tf.function()
def _train_step(data, inp, out, transformer, dis, tra_opt, dis_opt):
    clean = inp.no_noise(*data)
    dirty = inp(*data)
    mask = 1 - _mask(1, tf.shape(clean)[-2])
    with tf.GradientTape(persistent=True) as tape:
        unclean, _ = transformer(clean[:, :-1, :], clean[:, :-1, :], True, mask, mask, mask)
        fake = out(unclean)
        fake = dis(fake, True, mask)
        real = dis(dirty[:, 1:, :], True, mask)
        loss1 = tf.losses.binary_crossentropy(tf.ones_like(fake), fake, True, 0.1)
        loss2 = tf.losses.binary_crossentropy(tf.zeros_like(fake), fake, True, 0)
        loss3 = tf.losses.binary_crossentropy(tf.ones_like(real), real, True, 0.1)
        loss4 = tf.reduce_mean(loss2 + loss3)
        loss5 = tf.losses.binary_crossentropy(clean[:, 1:, 3], unclean[:, :, -1], from_logits=True, label_smoothing=0.1)
        loss6 = tf.losses.sparse_categorical_crossentropy(data[2][:, 1:], unclean[:, :, (out.instruments + 1):-1], from_logits=True)
        loss7 = tf.losses.sparse_categorical_crossentropy(data[1][:, 1:], unclean[:, :, 1:(out.instruments + 1)], from_logits=True)
        loss8 = tf.losses.logcosh(clean[:, 1:, 0], fake[:, :, 0])
        loss9 = tf.reduce_mean(loss5) + tf.reduce_mean(loss6) + tf.reduce_mean(loss7) +  tf.reduce_mean(loss8)
        loss10 = tf.reduce_mean(loss1)
        loss11 = loss9 * 0.5 + loss10
    tra_grad = tape.gradient(loss11, transformer.trainable_variables)
    tra_opt.apply_gradients(zip(tra_grad, transformer.trainable_variables))
    def dis_upd():
        dis_grad = tape.gradient(loss4, dis.trainable_variables)
        dis_opt.apply_gradients(zip(dis_grad, dis.trainable_variables))
        return 1
    tf.cond(loss10 < loss4, dis_upd, lambda: 0)
    del tape
    return loss1, loss2, loss3, loss9

def train():
    """
    Train the model
    """
    print("Setting up training")
    inp = AddNoiseToInput(time_multiplier=10, relative=True)
    out = CleanOutput(time_multiplier=10, relative=True, as_output=False)
    tra = Transformer(SEQUENCE_LENGTH-1, output_size=2 + out.instruments - out.minnote + out.maxnote + 1)
    dis = Discriminator(SEQUENCE_LENGTH-1)
    tra_opt = tf.keras.optimizers.Adam(LEARNING_RATE, 0.8, 0.95)
    dis_opt = tf.keras.optimizers.Adam(LEARNING_RATE, 0.8, 0.95)
    
    checkpoint = tf.train.Checkpoint(tra=tra, dis=dis, tra_opt=tra_opt, dis_opt=dis_opt)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint)
    tra_loss = tf.keras.metrics.Mean(name='loss_tra')
    opt_loss_r = tf.keras.metrics.Mean(name='loss_opt_r')
    opt_loss_f = tf.keras.metrics.Mean(name='loss_opt_f')
    sup_loss = tf.keras.metrics.Mean(name='loss_sup')
    data = enumerate(inp.read_dataset(batch=BATCH_SIZE, sequence=SEQUENCE_LENGTH))
    try:
        print("Preparing training")
        _train_step(next(data)[1], inp, out, tra, dis, tra_opt, dis_opt)
        print("Starting training")
        start = timer()
        for i, dat in data:
            loss = _train_step(dat, inp, out, tra, dis, tra_opt, dis_opt)
            tra_loss(loss[0])
            opt_loss_r(loss[1])
            opt_loss_f(loss[2])
            sup_loss(loss[3])
            if i % 50 == 0:
                print("Batch: {:6d}      Loss: {:4.2f} {:4.2f} {:4.2f} {:4.2f}      Time: {:6.2f}".format(
                    i,
                    tra_loss.result(),
                    opt_loss_r.result(),
                    opt_loss_f.result(),
                    sup_loss.result(),
                    timer() - start))
                if i % 500 == 0:
                    print("Saving checkpoint")
                    manager.save()
                    tra_loss.reset_states()
                    opt_loss_r.reset_states()
                    opt_loss_f.reset_states()
                    sup_loss.reset_states()
                start = timer()
    except KeyboardInterrupt:
        print("Saving checkpoint")
        manager.save()

@tf.function()
def _infer(data, inp, gen, out):
    data = inp(*data)
    mask = 1 - _mask(1, tf.shape(data)[1])
    tmp, _ = gen(data[:, :-1, :], data[:, :-1, :], False, mask, mask, mask)
    return out(tmp)

@tf.function()
def decode(inp):
    return inp

def generate():
    """
    Generate midi with the model
    """
    inp = AddNoiseToInput(time_multiplier=10, relative=True)
    out = CleanOutput(time_multiplier=10, relative=True, as_output=True)
    tra = Transformer(SEQUENCE_LENGTH-1, output_size=2 + out.instruments - out.minnote + out.maxnote + 1)
    checkpoint = tf.train.Checkpoint(tra=tra)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("No checkpoint to load")
        return
    data = next(iter(inp.read_dataset(batch=1, sequence=SEQUENCE_LENGTH)))
    data = [np.copy(decode(i)) for i in data]
    for i in range(20, SEQUENCE_LENGTH-1):
        tmp = _infer(data, inp, tra, out)
        for j, t in enumerate(tmp):
            data[j][:, i+1] = t[:, i]
    out.write_to_file(*data)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == "generate":
        generate()
    else:
        print("You must specify what you want to do (train / generate)")
