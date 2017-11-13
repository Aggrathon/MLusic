
import tensorflow as tf
from song import get_all_vectors
from model import network


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    def input_fn():
        vectors = get_all_vectors()
        data = tf.convert_to_tensor(vectors, tf.float32)
        data = tf.reshape(data, vectors.shape)
        batch = tf.stack([tf.random_crop(data, (32, vectors.shape[-1])) for _ in range(32)])
        batch = tf.reshape(batch, (32, 32, vectors.shape[-1]))
        return {'input': batch[:, :-1, :]}, {'output': batch[:, -1:, :]}
    nn = network()
    nn.train(input_fn)
