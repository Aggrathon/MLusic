
import tensorflow as tf
from song import get_all_vectors
from model import network
from config import BATCH_SIZE, SEQUENCE_LENGTH


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    def input_fn():
        vectors = get_all_vectors()
        data = tf.convert_to_tensor(vectors, tf.float32)
        data = tf.reshape(data, vectors.shape)
        batch = tf.stack([tf.random_crop(data, (SEQUENCE_LENGTH+1, vectors.shape[-1])) for _ in range(BATCH_SIZE)])
        batch = tf.reshape(batch, (BATCH_SIZE, SEQUENCE_LENGTH+1, vectors.shape[-1]))
        return {'input': batch[:, :-1, :]}, {'output': batch[:, -1:, :]}
    nn = network()
    nn.train(input_fn)
