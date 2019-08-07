"""
    Transformer for predicting the next "pseudo" midi events

    Transformer architechture from https://www.tensorflow.org/beta/tutorials/text/transformer
    Relative time embeddings from https://arxiv.org/abs/1809.04281
"""
import sys
from time import time as timer
from uuid import uuid4
import tensorflow as tf
import numpy as np
from convert_input import DATA_FILE, META_FILE


SEQUENCE_LENGTH = 257
BATCH_SIZE = 64
LEARNING_RATE = 1e-6
CHECKPOINT_PATH = "./network/transformer_midi"


def _input(file=DATA_FILE, batch: int = BATCH_SIZE, sequence: int = SEQUENCE_LENGTH, relative: bool = True):
    """
    Read the input dataset

    Keyword Arguments:
        input {} -- input file name (default: {DATA_FILE})
        batch {int} -- batch size (default: {BATCH_SIZE})
        sequence {int} -- sequence length (default: {SEQUENCE_LENGTH})

    Returns:
        PrefetchDataset -- (time, instrument, tone, state)
        int -- num_instruments
    """
    data = tf.data.experimental.make_csv_dataset(
        file_pattern=str(file),
        batch_size=sequence,
        column_names=["time", "instrument", "note", "state"],
        column_defaults=[0, 0, 0, 0],
        shuffle=False,
        header=False)
    if relative:
        data = data.map(lambda row: (
            tf.concat(([0.0], tf.cast(tf.cast(row["time"][1:] - row["time"][:-1], tf.float64)/100_000, tf.float32)), -1),
            row["instrument"],
            tf.cast(row["note"], tf.float32),
            tf.cast(row["state"], tf.float32)))
    else:
        data = data.map(lambda row: (
            tf.cast(tf.cast(row["time"] - tf.reduce_min(row["time"]), tf.float64)/100_000, tf.float32),
            row["instrument"],
            tf.cast(row["note"], tf.float32),
            tf.cast(row["state"], tf.float32)))
    data = data.shuffle(batch*80).batch(batch)
    num_instruments = 129
    with open(META_FILE) as file:
        num_instruments = len(file.readlines())
    return data.prefetch(tf.data.experimental.AUTOTUNE), num_instruments


def _mask(start: int, total: int = SEQUENCE_LENGTH):
    """Create triangular mask

    Arguments:
        start {int} -- trim too few unmasked

    Keyword Arguments:
        total {int} -- The maximum sequence length (default: {SEQUENCE_LENGTH})

    Returns:
        Tensor(total-start, total) -- mask
    """
    return tf.linalg.band_part(tf.ones((total-1, total-1)), -1, 0)[(start-1):, :]


def attention(q, k, v, mask=None):
    """Attention (scaled dot)

    Arguments:
        q {Tensor(..., seq_len_q, depth)} -- query
        k {Tensor(..., seq_len_k, depth)} -- key
        v {Tensor(..., seq_len_k, depth_v)} -- value

    Keyword Arguments:
        mask {Tensor(..., seq_len_q, seq_len_k)} -- mask (default: None)

    Returns:
        Tensor(..., seq_len_q, depth_v) -- Output
        Tensor(..., seq_len_q, seq_len_k) -- Attention Weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    sq_dk = tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
    scaled_attention_logits = matmul_qk / sq_dk
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class AttentionRelative(tf.keras.layers.Layer):
    """
        Attention (scaled dot) with relative distances
    """
    def __init__(self):
        super(AttentionRelative, self).__init__()
    
    def build(self, input_shape):
        #This assumes that t.shape(q) == tf.shape(k)
        init_val = tf.keras.initializers.RandomNormal(stddev=tf.cast(input_shape[-1], tf.float32)**-0.5)
        self.key_relative_embeddings = self.add_weight("key_relative_embeddings", input_shape[-2:], tf.float32, init_val, trainable = True)
        self.value_relative_embeddings = self.add_weight("value_relative_embeddings", input_shape[-2:], tf.float32, init_val, trainable = True)

    def call(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        # Relative Embedding
        matmul_rel = tf.einsum("bhld,md->bhlm", q, self.key_relative_embeddings)
        matmul_rel = tf.slice(tf.reshape(tf.pad(matmul_rel, [[0, 0], [0, 0], [0, 0], [1, 0]]), \
            tf.shape(matmul_rel) + [0, 0, 1, 0]), [0, 0, 1, 0], [-1, -1, -1, -1])
        # Continue Normally
        scaled_attention_logits = matmul_qk + matmul_rel #TODO: add matmul_time == rel with q[:,:,0]? only for first layer?
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        # Add relative to output
        relative_weights = tf.slice(tf.reshape(tf.pad(attention_weights, [[0, 0], [0, 0], [1, 0], [0, 0]]), \
            tf.shape(attention_weights) + [0, 0, 0, 1]), [0, 0, 0, 1], tf.shape(attention_weights))
        output += tf.einsum("bhlm,md->bhld", relative_weights, self.value_relative_embeddings)
        return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        self.attention = AttentionRelative()

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights # (batch_size, seq_len_q, d_model), (batch_size, num_heads, seq_len_q, seq_len_k)


def _feed_forward(d_model, dff):
    """A short feed forward network

    Arguments:
        d_model {int} -- output layers
        dff {int} -- internal layers

    Returns:
        Keras(batch_size, seq_len, d_model) -- Keras sequential model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(seq_len, d_model, num_heads)
        self.ffn = _feed_forward(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2 # (batch_size, input_seq_len, d_model)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, seq_len, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(seq_len, d_model, num_heads, dff, rate) for _ in range(num_layers)]

    def call(self, x, training, mask):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(seq_len, d_model, num_heads)
        self.mha2 = MultiHeadAttention(seq_len, d_model, num_heads)
        self.ffn = _feed_forward(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2 # (batch_size, target_seq_len, d_model), ?, ?


class Decoder(tf.keras.layers.Layer):
    def __init__(self, seq_len, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.dec_layers = [DecoderLayer(seq_len, d_model, num_heads, dff, rate) for _ in range(num_layers)]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights # (batch_size, target_seq_len, d_model), {}


class Transformer(tf.keras.Model):
    def __init__(self, seq_len, num_layers, d_model, num_heads, dff, output_size, rate=0.1):
        super(Transformer, self).__init__()
        self.embedding = tf.keras.layers.Dense(d_model)
        self.encoder = Encoder(seq_len, num_layers, d_model, num_heads, dff, rate)
        self.decoder = Decoder(seq_len, num_layers, d_model, num_heads, dff, rate)
        self.final_layer = tf.keras.layers.Dense(output_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        inp = self.embedding(inp)
        tar = self.embedding(tar)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights # (batch_size, tar_seq_len, output_size), ?


def _loss(vec, time, instrument, note, state):
    loss_state = tf.reduce_mean(tf.losses.binary_crossentropy(state, vec[:, :, -1], from_logits=True, label_smoothing=0.1))
    loss_note = tf.reduce_mean(tf.losses.MSE(note, vec[:, :, -2])) / 10
    loss_time = tf.reduce_mean(tf.losses.logcosh(time, vec[:, :, 0])) * 10
    loss_inst = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(instrument, vec[:, :, 1:-2], from_logits=True))
    return loss_time + loss_inst + loss_note + loss_state, loss_time, loss_inst, loss_note, loss_state


@tf.function()
def _train_step(time, instrument, note, state, transformer, optimiser, max_ins=129):
    mask = 1 - _mask(1, tf.shape(instrument)[-1])
    data = tf.concat((
        tf.expand_dims(time, -1),
        tf.one_hot(instrument, max_ins, dtype=tf.float32),
        tf.expand_dims(note, -1),
        tf.expand_dims(state, -1)), -1)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(data[:, :-1, :], data[:, :-1, :], True, mask, mask, mask)
        loss = _loss(predictions, time[:, 1:], instrument[:, 1:], note[:, 1:], state[:, 1:])
    gradients = tape.gradient(loss[0], transformer.trainable_variables)
    optimiser.apply_gradients(zip(gradients, transformer.trainable_variables))
    return loss


def train():
    """
    Train the model
    """
    data, ins = _input(sequence=SEQUENCE_LENGTH)
    optimiser = tf.keras.optimizers.Adam(LEARNING_RATE, 0.9, 0.98, 1e-9)
    vec_size = 3 + ins
    transformer = Transformer(SEQUENCE_LENGTH-1, 4, 128, 8, 512, vec_size, 0.1)
    checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimiser)
    manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_PATH, max_to_keep=5)
    if manager.latest_checkpoint:
        print('Loading latest checkpoint')
        checkpoint.restore(manager.latest_checkpoint)
    loss_total = tf.keras.metrics.Mean(name='loss_total')
    loss_state = tf.keras.metrics.Mean(name='loss_state')
    loss_note = tf.keras.metrics.Mean(name='loss_note')
    loss_time = tf.keras.metrics.Mean(name='loss_time')
    loss_inst = tf.keras.metrics.Mean(name='loss_inst')
    data = enumerate(data)
    next(data)
    try:
        start = timer()
        print("Starting training")
        for (i, (time, instrument, note, state)) in data:
            loss = _train_step(time, instrument, note, state, transformer, optimiser, ins)
            loss_time(loss[1])
            loss_inst(loss[2])
            loss_note(loss[3])
            loss_state(loss[4])
            loss_total(loss[0])
            if i % 50 == 0:
                print("Batch: {:6d}      Loss: {:5.3f} ({:4.2f} {:4.2f} {:4.2f} {:4.2f})      Time: {:6.2f}".format(
                    i,
                    loss_total.result(),
                    loss_time.result(),
                    loss_inst.result(),
                    loss_note.result(),
                    loss_state.result(),
                    timer() - start))
                if i % 500 == 0:
                    print("Saving checkpoint")
                    manager.save()
                    loss_total.reset_states()
                    loss_state.reset_states()
                    loss_note.reset_states()
                    loss_time.reset_states()
                    loss_inst.reset_states()
                start = timer()
    except KeyboardInterrupt:
        print("Saving checkpoint")
        manager.save()


def generate(songs=1):
    """
    Generate midi with the model
    """
    pass
    # TODO: Update this to use "pseudo" midi:s
    # s = Song().read_data(DATA_FILE)
    # def _gen():
    #     for _ in range(songs):
    #         start = np.random.randint(0, s.times.shape[0]-50)
    #         end = start + 40
    #         times = s.times[start:end, :]
    #         instruments = s.notes[start:end, 0]
    #         notes = s.notes[start:end, 1]
    #         yield times, instruments, notes
    # def _inp():
    #     data = tf.data.Dataset.from_generator(
    #         _gen, (tf.float32, tf.int32, tf.int32), ((40, 3), (40,), (40,)))
    #     tim, ins, ton = data.batch(1).make_one_shot_iterator().get_next()
    #     return {"times": tim, "instrument": ins, "tone": ton}, {}
    # est = tf.estimator.Estimator(_model, "network/predictor")
    # for out in est.predict(_inp, yield_single_examples=False):
    #     times = out["times"]
    #     instruments = out["instrument"]
    #     tones = out["tone"]
    #     times.shape = times.shape[1:]
    #     tones.shape = tones.shape[1]
    #     instruments.shape = instruments.shape[1]
    #     s.set_data(times, np.stack((instruments, tones), 1))
    #     save_and_convert_song(s, "output/song_"+str(uuid4())+".csv", False)


def _handle_start():
    if len(sys.argv) == 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == "generate":
        generate(5)
    else:
        print("You must specify what you want to do (train / generate)")


if __name__ == "__main__":
    _handle_start()
