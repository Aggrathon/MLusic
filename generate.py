
import random
import numpy as np
import tensorflow as tf
from model import network
from song import Song, Note, get_all_vectors
from convert_outputs import save_and_convert_song
from config import SEQUENCE_LENGTH


if __name__ == "__main__":
    vectors = get_all_vectors()
    rnd = random.randrange(0, len(vectors)-SEQUENCE_LENGTH)
    sequence = [vectors[i+rnd, :] for i in range(SEQUENCE_LENGTH)]
    ph = None
    def input():
        global ph
        ph = tf.placeholder(tf.float32, [1, SEQUENCE_LENGTH, vectors.shape[1]])
        return {'input': ph}, None
    def feed():
        global ph
        return {ph: [sequence[-SEQUENCE_LENGTH:]]}
    nn = network()
    for i, vec in enumerate(nn.predict(input, None, [tf.train.FeedFnHook(feed)])):
        sequence.append(vec['output'])
        if i > 1000:
            break
    song = Song('test')
    song.notes = [Note.from_vector(v) for v in sequence[SEQUENCE_LENGTH:]]
    song.export_cleanup()
    save_and_convert_song(song, True)
