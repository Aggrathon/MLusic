
import tensorflow as tf
from song import get_all_vectors
from model import network
from config import BATCH_SIZE, SEQUENCE_LENGTH


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    def input_fn():
        vectors = get_all_vectors()
        data = tf.data.Dataset.from_tensor_slices(vectors)
        batch = data.repeat().batch(SEQUENCE_LENGTH+1).shuffle(buffer_size=1000).batch(BATCH_SIZE).make_one_shot_iterator().get_next()
        return {'input': batch[:, :-1, :]}, {'output': batch[:, -1:, :]}
    nn = network()
    nn.train(input_fn)
