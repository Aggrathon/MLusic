
import datetime
import random
import sys
import numpy
from numpy.random import rand as nprand
import tflearn
from tflearn.data_utils import shuffle
from song import *
from config import *
from platform_dependent import save_and_convert_song, copy_file

NOTE_RANGE = HIGHEST_NOTE - LOWEST_NOTE + META_TO_MATRIX

def get_data(scramble_sequences: bool=True) -> ([], []):
    matrices = [add_meta_to_matrix(s.generate_tone_matrix(), s.length)
                for s in read_all_inputs() if s.cleanup()]
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

def add_meta_to_matrix(matrix: numpy.ndarray, length: int) -> numpy.ndarray:
    if META_TO_MATRIX == 0:
        return matrix
    elif META_TO_MATRIX == 1:
        m_len, _ = matrix.shape
        bar_time = [[i%(TIME_RESOLUTION*4)] for i in range(m_len)]
        return numpy.hstack((matrix, bar_time))
    elif  META_TO_MATRIX == 2:
        m_len, _ = matrix.shape
        bar_time = [[i%(TIME_RESOLUTION*4)] for i in range(m_len)]
        rel_time = [[float(i/length)] for i in range(m_len)]
        return numpy.hstack((matrix, bar_time, rel_time))

def build_network(name: str=None):
    rnn = tflearn.input_data(shape=[None, SEQUENCE_LENGTH, NOTE_RANGE])
    for i in range(NETWORK_DEPTH):
        if i < DOUBLE_WIDTH_LAYERS:
            rnn = tflearn.lstm(rnn, NETWORK_WIDTH*2, return_seq=(i != NETWORK_DEPTH-1))
        else:
            rnn = tflearn.lstm(rnn, NETWORK_WIDTH*2, return_seq=(i != NETWORK_DEPTH-1))
        rnn = tflearn.dropout(rnn, DROPOUT)
    rnn = tflearn.fully_connected(rnn, NOTE_RANGE, activation='softmax')
    rnn = tflearn.regression(rnn, optimizer='adadelta', loss='binary_crossentropy', learning_rate=LEARNING_RATE)
    sqgen = tflearn.SequenceGenerator(rnn, {i: i for i in range(NOTE_RANGE)}, seq_maxlen=SEQUENCE_LENGTH, \
            checkpoint_path=os.path.join(NETWORK_FOLDER, "checkpoints", ""), tensorboard_dir=os.path.join(NETWORK_FOLDER, "logs"))
    network_path = check_network(name)
    if network_path != '':
        sqgen.load(network_path)
    return sqgen

def check_network(name: str=None, log: bool=True) -> str:
    if name is not None:
        path = os.path.join(NETWORK_FOLDER, name)
        if os.path.isfile(path+".meta"):
            return path
        else:
            if log:
                print("Network ", name, "was not found, creating a new network")
            return ''
    else:
        files = os.listdir(NETWORK_FOLDER)
        meta = [os.path.join(NETWORK_FOLDER, f) for f in files if f.endswith(".meta")]
        if len(meta) == 0:
            if log:
                print("No network was not found, creating a new network")
            return ''
        else:
            meta.sort(key=lambda value: os.path.getmtime(value), reverse=True)
            if log:
                print("Using network configuration:", meta[0][meta[0].rfind(os.path.sep)+1:-5])
            return meta[0][:-5]

def get_network_path(name: str) -> str:
    if name is None:
        name = 'ai-{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
    return os.path.join(NETWORK_FOLDER, name)


def train(network_name: str=None):
    path = get_network_path(network_name)
    print("Gathering data")
    x, y = get_data()
    print("Constructing the neural network")
    network = build_network(network_name)
    print("Saving a copy of the current config")
    copy_file("config.py", path+"-config.py")
    print("Starting the learning process")
    network.fit(x, y, n_epoch=TRAINING_EPOCHS, show_metric=True, snapshot_epoch=True, validation_set=VALIDATION_SIZE)
    network.save(path)
    print("Trained network configuration saved to "+path)

def generate(network_name: str=None):
    if check_network(network_name, False) == '':
        print("Cannot generate a new song without any network configuration (train the network first)")
        return
    print("Constructing the neural network")
    network = build_network(network_name)
    print("Gathering Seed")
    song = random.choice(read_all_inputs())
    start = 0
    if not SEED_ONLY_BEGINNING:
        start = random.randint(0, song.length-SEQUENCE_LENGTH-1)
    sequence = add_meta_to_matrix(song.generate_tone_matrix()[start:start+SEQUENCE_LENGTH], SONG_LENGTH)
    generated = sequence
    print("Song {} used as Seed (starting from {} %)".format(song.name, round(start*100.0/song.length)))
    print("Generating a new sequence")
    for i in range(SEQUENCE_LENGTH, SONG_LENGTH):
        pred = numpy.array(network._predict([sequence]))
        prediction_to_timestep(pred, i, SONG_LENGTH)
        generated = numpy.vstack((generated, pred))
        sequence = generated[-SEQUENCE_LENGTH:, :]
        if i % 100 == 0 and i != 0:
            print("{} / {} timesteps generated".format(i, SONG_LENGTH))
    file_name = save_and_convert_song(Song.convert_tone_matrix(generated))
    print("Generated song saved to "+file_name)

def prediction_to_timestep(matrix: numpy.ndarray, index: int, length: int):
    notes = matrix[0][:-META_TO_MATRIX]
    mean = numpy.mean(notes)
    std = numpy.std(notes)
    for i in range(notes.shape[0]):
        matrix[0, i] = 0
    if std != 0:
        notes *= (nprand(*notes.shape) - 0.5) * (2*RANDOMNESS)
        over_mean = [i for i in range(notes.shape[0]) if notes[i] > mean]
        if len(over_mean) > AVERAGE_TONE_DENSITY / 2 and len(over_mean) < AVERAGE_TONE_DENSITY * 2:
            for i in over_mean:
                matrix[0, i] = 1
        else:
            sort_notes = [(num, i) for i, num in enumerate(notes)]
            sort_notes.sort(reverse=True)
            for _, i in over_mean[:AVERAGE_TONE_DENSITY]:
                matrix[0, i] = 1
    if META_TO_MATRIX == 1:
        matrix[0, -1] = index%(TIME_RESOLUTION*4)
    elif META_TO_MATRIX == 2:
        matrix[0, -2] = index%(TIME_RESOLUTION*4)
        matrix[0, -1] = float(index/length)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            generate()
        else:
            train()
    else:
        train()
