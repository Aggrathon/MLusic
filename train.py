
import tensorflow as tf
from song import get_all_vectors
from model import network
from config import BATCH_SIZE, SEQUENCE_LENGTH


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    def input_fn():
        vec = get_all_vectors()
        def gen():
            for v in vec:
                yield v
        data = tf.data.Dataset.from_generator(gen, tf.float32, (vec.shape[1]))
        batch = data.repeat().batch(SEQUENCE_LENGTH+1).shuffle(buffer_size=800).batch(BATCH_SIZE).make_one_shot_iterator().get_next()
        return {'input': batch[:, :-1, :]}, {'output': batch[:, -1:, :]}
    nn = network(BATCH_SIZE)
    nn.train(input_fn)
