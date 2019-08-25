"""
    Transformer architechture

    Transformer architechture from https://www.tensorflow.org/beta/tutorials/text/transformer
    Relative time embeddings from https://arxiv.org/abs/1809.04281
"""
import tensorflow as tf

def positional_encoding(seq_len, d_model):
    """
        Get a positional encoding matrix
    """
    positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    models = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2.0 * tf.floor(models / 2.0)) / tf.cast(d_model, tf.float32))
    angle_rads = positions * angle_rates
    rads_even = tf.sin(angle_rads[:, 0::2])
    rads_odd = tf.cos(angle_rads[:, 1::2])
    return tf.concat((rads_even, rads_odd), -1)[tf.newaxis, :, :]

def padding_mask(seq):
    """
        Masks out padding elements
        (Not needed here since the length is fixed)
    """
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

def look_ahead_mask(size):
    """
        Masks future tokens
    """
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

class Attention(tf.keras.layers.Layer):
    """
        Attention (scaled dot)
    """
    def __init__(self):
        super(Attention, self).__init__()

    def call(self, q, k, v, mask=None):
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
        self.key_relative_embeddings = self.add_weight("key_relative_embeddings", input_shape[-2:], tf.float32, init_val, trainable=True)
        self.value_relative_embeddings = self.add_weight("value_relative_embeddings", input_shape[-2:], tf.float32, init_val, trainable=True)

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
    def __init__(self, d_model, num_heads, relative=True):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        self.attention = AttentionRelative() if relative else Attention()

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


def feed_forward(d_model, dff):
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
    def __init__(self, d_model, num_heads, dff, rate=0.1, relative=True):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, relative)
        self.ffn = feed_forward(d_model, dff)
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
    def __init__(self, num_layers, d_model, num_heads, dff, input_size, rate=0.1, relative=True):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = feed_forward(d_model, d_model)
        self.relative = relative
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, relative) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)    

        def build(self, input_shape):
            if not self.relative:
                self.pos_enc = positional_encoding(input_shape[1], self.d_model)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        if not self.relative:
            x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x = x + self.pos_enc
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, relative=True):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, relative)
        self.mha2 = MultiHeadAttention(d_model, num_heads, relative)
        self.ffn = feed_forward(d_model, dff)
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
    def __init__(self, num_layers, d_model, num_heads, dff, output_size, rate=0.1, relative=True):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = feed_forward(d_model, d_model)
        self.relative = relative
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, relative) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        if not self.relative:
            self.pos_enc = positional_encoding(input_shape[1], self.d_model)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]    
        attention_weights = {}
        x = self.embedding(x)
        if not self.relative:
            x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x = x + self.pos_enc
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights # (batch_size, target_seq_len, d_model), {}


class Transformer(tf.keras.Model):
    def __init__(self, num_layers=4, d_model=128, num_heads=8, dff=512, input_size=30, output_size=30, rate=0.1, relative=True):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_size, rate, relative)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, output_size, rate, relative)
        self.final_layer = tf.keras.layers.Dense(output_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights # (batch_size, tar_seq_len, output_size), ?


class DecodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, relative=True):
        super(DecodingLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, relative)
        self.ffn = feed_forward(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask):
        attn, attn_weights_block = self.mha(x, x, x, look_ahead_mask)
        attn = self.dropout1(attn, training=training)
        out = self.layernorm1(attn + x)
        ffn_output = self.ffn(out)
        ffn_output = self.dropout2(ffn_output, training=training)
        final = self.layernorm2(ffn_output + out)
        return final, attn_weights_block


class DecodingTransformer(tf.keras.Model):
    def __init__(self, num_layers=4, d_model=128, num_heads=8, dff=512, output_size=30, rate=0.1, relative=True):
        super(DecodingTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = feed_forward(d_model, d_model)
        self.relative = relative
        self.dec_layers = [DecodingLayer(d_model, num_heads, dff, rate, relative) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(output_size)

    def build(self, input_shape):
        if not self.relative:
            self.pos_enc = positional_encoding(input_shape[1], self.d_model)

    def call(self, x, training, la_mask):
        attention_weights = {}
        x = self.embedding(x)
        if not self.relative:
            x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x = x + self.pos_enc
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block = self.dec_layers[i](x, training, la_mask)
            attention_weights['decoder_layer{}_block'.format(i+1)] = block
        final_output = self.final_layer(x)
        return final_output, attention_weights # (batch_size, tar_seq_len, output_size), ?
