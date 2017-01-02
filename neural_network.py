
import random
import sys
import numpy
import tflearn
from song import *

SEQUENCE_LENGTH = 128


def get_data(sequence_length=SEQUENCE_LENGTH, scramble_sequences=True):
    songs = read_all_inputs()
    dictionary = { i : i for i in range(128) }
    matrices = [s.generate_tone_matrix() for s in songs if s.cleanup()]
    x = []
    y = []
    for m in matrices:
        for i in range(0, m.shape[0]-sequence_length-1):
            x.append(m[i:i+sequence_length, :])
            y.append(m[i+sequence_length:i+sequence_length+1, :].flatten())
    if scramble_sequences:
        for i in reversed(range(0, len(x))):
            j = random.randint(0, i)
            tmp = x[j]
            x[j] = x[i]
            x[i] = tmp
            tmp = y[j]
            y[j] = y[i]
            y[i] = tmp
    return dictionary, x, y

def build_network(sequence_length=SEQUENCE_LENGTH, dictionary=None):
    learning_rate = 0.001

    rnn = tflearn.input_data(shape=[None, sequence_length, 128])
    rnn = tflearn.lstm(rnn, 512, return_seq=True)
    rnn = tflearn.dropout(rnn, 0.5)
    rnn = tflearn.lstm(rnn, 512, return_seq=True)
    rnn = tflearn.dropout(rnn, 0.5)
    rnn = tflearn.lstm(rnn, 512)
    rnn = tflearn.dropout(rnn, 0.5)
    rnn = tflearn.fully_connected(rnn, 128, activation='softmax')
    rnn = tflearn.regression(rnn, optimizer='adam', loss='binary_crossentropy', learning_rate=learning_rate)
    gen = tflearn.SequenceGenerator(rnn, dictionary, seq_maxlen=sequence_length, checkpoint_path=OUTPUT_FOLDER+"checkpoints", tensorboard_dir=OUTPUT_FOLDER+"logs")
    return gen

def train():
    seq_len = SEQUENCE_LENGTH

    print("Gathering data")
    dictionary, x, y = get_data(seq_len)
    print("Building the neural network")
    network = build_network(seq_len, dictionary)
    print("Starting the learning process")
    network.fit(x, y, n_epoch=2, show_metric=True, snapshot_epoch=True, validation_set=0.1)
    network.save(OUTPUT_FOLDER+"third.tflearn")
    print("Trained Network Configuration saved to {}third.tflearn".format(OUTPUT_FOLDER))

def generate():
    seq_len = SEQUENCE_LENGTH
    song_len = 1000

    print("Gathering data")
    dictionary, x, y = get_data(seq_len, False)
    print("Building the neural network")
    network = build_network(seq_len, dictionary)
    print("Loading the configuration")
    network.load(OUTPUT_FOLDER+"third.tflearn")

    print("Generating a new sequence")
    sequence = x[random.randint(0, len(x)-1)]
    generated = sequence
    for i in range(song_len):
        pred = numpy.array(network._predict([sequence]))
        array_to_boolean(pred)
        combine = numpy.zeros((seq_len+i+1, 128))
        combine[:-1, :] = generated
        combine[-1:, :] = pred
        generated = combine
        sequence = combine[-seq_len:, :]
        if i % 200 == 0 and i != 0:
            print("{} / {} timesteps generated".format(i,song_len))
    song = Song.convert_tone_matrix(generated)
    song.save_to_file()
    print("Generated song saved to "+OUTPUT_FOLDER+song.name+".csv", song.length)

def array_to_boolean(matrix):
    mean = numpy.mean(matrix)
    shx, shy = matrix.shape
    for i in range(shx):
        for j in range(shy):
            if matrix[i, j] > mean:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            generate()
        else:
            train()
    else:
        train()
