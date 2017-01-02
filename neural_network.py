
import datetime
import random
import sys
import numpy
import tflearn
from tflearn.data_utils import shuffle
from song import *
from platform_dependent import save_and_convert_song

NOTE_RANGE = HIGHEST_NOTE - LOWEST_NOTE
if ADD_META_TO_MATRIX:
    NOTE_RANGE += 2

def get_data(scramble_sequences: bool=True) -> ([], []):
    if ADD_META_TO_MATRIX:
        matrices = [add_time_signature(s.generate_tone_matrix(), s.length)
                    for s in read_all_inputs() if s.cleanup()]
    else:
        matrices = [s.generate_tone_matrix() for s in read_all_inputs() if s.cleanup()]
    x = []
    y = []
    for m in matrices:
        for i in range(0, m.shape[0]-SEQUENCE_LENGTH-1):
            x.append(m[i:i+SEQUENCE_LENGTH, :])
            y.append(m[i+SEQUENCE_LENGTH:i+SEQUENCE_LENGTH+1, :].flatten())
    if scramble_sequences:
        return shuffle(x, y)
    else:
        return x, y

def add_time_signature(matrix: numpy.ndarray, length: int) -> numpy.ndarray:
    m_len, _ = matrix.shape
    rel_time = [[float(i/length)] for i in range(m_len)]
    bar_time = [[i%(TIME_RESOLUTION*4)] for i in range(m_len)]
    return numpy.hstack((matrix, rel_time, bar_time))

def build_network(name: str=None):
    rnn = tflearn.input_data(shape=[None, SEQUENCE_LENGTH, NOTE_RANGE])
    for i in range(NETWORK_DEPTH):
        rnn = tflearn.lstm(rnn, NETWORK_WIDTH, return_seq=(i < NETWORK_DEPTH/2))
        rnn = tflearn.dropout(rnn, DROPOUT)
    rnn = tflearn.fully_connected(rnn, NOTE_RANGE, activation='softmax')
    rnn = tflearn.regression(rnn, optimizer='adam', loss='binary_crossentropy', learning_rate=LEARNING_RATE)
    sqgen = tflearn.SequenceGenerator(rnn, {i: i for i in range(NOTE_RANGE)}, seq_maxlen=SEQUENCE_LENGTH, \
            checkpoint_path=os.path.join(NETWORK_FOLDER, "checkpoints"), tensorboard_dir=os.path.join(NETWORK_FOLDER, "logs"))
    network_path = check_network(name)
    if network_path != '':
        sqgen.load(network_path)
    return sqgen

def check_network(name: str=None) -> str:
    if name is not None:
        path = os.path.join(NETWORK_FOLDER, name)
        if os.path.isfile(path+".meta"):
            return path
        else:
            print("Network ", name, "was not found, creating a new network")
            return ''
    else:
        files = os.listdir(NETWORK_FOLDER)
        meta = [os.path.join(NETWORK_FOLDER, f) for f in files if f.endswith(".meta")]
        if len(meta) == 0:
            print("No network was not found, creating a new network")
            return ''
        else:
            meta.sort(key=lambda value: os.path.getmtime(value))
            return meta[0][:-5]

def get_network_path(name: str) -> str:
    if name is None:
        name = 'ai-{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
    return os.path.join(NETWORK_FOLDER, name)


def train(network_name: str=None):
    print("Gathering data")
    x, y = get_data()
    print("Constructing the neural network")
    network = build_network(network_name)
    print("Starting the learning process")
    network.fit(x, y, n_epoch=TRAINING_EPOCHS, show_metric=True, snapshot_epoch=True, validation_set=VALIDATION_SIZE)
    path = get_network_path(network_name)
    network.save(path)
    print("Trained network configuration saved to "+path)

def generate(network_name: str=None):
    if check_network(network_name) == '':
        print("Cannot generate a song with an empty network")
        return
    print("Constructing the neural network")
    network = build_network(network_name)

    print("Gathering data")
    songs = read_all_inputs()
    if ADD_META_TO_MATRIX:
        sequence = add_time_signature(random.choice(songs).generate_tone_matrix()[:SEQUENCE_LENGTH], SONG_LENGTH)
    else:
        sequence = random.choice(songs)[:SEQUENCE_LENGTH]
    songs.clear()
    generated = sequence
    print("Generating a new sequence")
    for i in range(SONG_LENGTH):
        pred = numpy.array(network._predict([sequence]))
        array_to_boolean(pred, SEQUENCE_LENGTH+i, SONG_LENGTH)
        combine = numpy.zeros((SEQUENCE_LENGTH+i+1, NOTE_RANGE))
        combine[:-1, :] = generated
        combine[-1:, :] = pred
        generated = combine
        sequence = combine[-SEQUENCE_LENGTH:, :]
        if i % 200 == 0 and i != 0:
            print("{} / {} timesteps generated".format(i, SONG_LENGTH))
    file_name = save_and_convert_song(Song.convert_tone_matrix(generated))
    print("Generated song saved to "+file_name)

def array_to_boolean(matrix: numpy.ndarray, index: int, length: int):
    mean = numpy.mean(matrix[:, :-2])
    shx, shy = matrix.shape
    for i in range(shx):
        for j in range(shy-2):
            if matrix[i, j] > mean:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    matrix[0, -2] = float(index/length)
    matrix[0, -1] = index%(TIME_RESOLUTION*4)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            generate()
        else:
            train()
    else:
        train()
