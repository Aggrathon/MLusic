
import tflearn
from song import *
import numpy


def learn():
    max_len = 200
    learning_rate = 0.001

    songs = read_all_inputs()
    max_len = min(max_len, min(songs, key=lambda s: s.length).length-1)
    names = {s.name : i for i, s in enumerate(songs)}
    matrices = [s.generate_tone_matrix(max_len+1) for s in songs if s.cleanup()]

    x = [numpy.delete(m, (m.shape[0]-1), axis=0) for m in matrices]
    y = [numpy.delete(m, (0), axis=0) for m in matrices]

    songs.clear()
    matrices.clear()
    print("Data gathered and formatted (sequence length = {}), starting to build the network".format(max_len))

    rnn = tflearn.input_data(shape=[None, max_len, 128])
    rnn = tflearn.lstm(rnn, 512, return_seq=True)
    rnn = tflearn.dropout(rnn, 0.5)
    rnn = tflearn.lstm(rnn, 512, return_seq=True)
    rnn = tflearn.dropout(rnn, 0.5)
    rnn = tflearn.lstm(rnn, 512)
    rnn = tflearn.dropout(rnn, 0.5)
    rnn = tflearn.fully_connected(rnn, 128, activation='softmax')
    rnn = tflearn.regression(rnn, optimizer='adam', loss='binary_crossentropy', learning_rate=learning_rate)
    gen = tflearn.SequenceGenerator(rnn, names, seq_maxlen=max_len, checkpoint_path=OUTPUT_FOLDER)

    print("Neural network created, starting the learning process")
    gen.fit(x, y, n_epoch=1, batch_size=58, show_metric=True, snapshot_epoch=True) # , validation_set=0.1


if __name__ == "__main__":
    learn()
