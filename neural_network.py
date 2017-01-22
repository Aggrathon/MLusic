import gc
import datetime
import random
import sys
import numpy
from numpy.random import rand as nprand
import tflearn
from song import *
from config import *
from platform_dependent import save_and_convert_song, copy_file

NOTE_RANGE = numpy.sum([td['highest_note'] - td['lowest_note'] for td in TRACK_TO_DATA]) + META_TO_MATRIX

def get_data(scramble_sequences: bool=True) -> ([], []):
    matrices = [add_meta_to_matrix(s.generate_tone_matrix(), s.length, s.bar_length)
                for s in read_all_inputs() if s.cleanup(False)]
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

def shuffle(arr1, arr2):
    assert len(arr1) == len(arr2)
    for i in range(len(arr1)):
        rnd = random.randrange(i, len(arr1))
        tmp = arr1[i]
        arr1[i] = arr1[rnd]
        arr1[rnd] = tmp
        tmp = arr2[i]
        arr2[i] = arr2[rnd]
        arr2[rnd] = tmp
    return arr1, arr2

def add_meta_to_matrix(matrix: numpy.ndarray, length: int, bar_length: int=4) -> numpy.ndarray:
    if META_TO_MATRIX == 0:
        return matrix
    elif META_TO_MATRIX == 1:
        m_len, _ = matrix.shape
        bar_time = [[i%(TIME_RESOLUTION*4)] for i in range(m_len)]
        return numpy.hstack((matrix, bar_time))
    elif  META_TO_MATRIX == 2:
        m_len, _ = matrix.shape
        bar_time = [[i%(TIME_RESOLUTION*bar_length)/(TIME_RESOLUTION*bar_length)] for i in range(m_len)]
        rel_time = [[float(i/length)] for i in range(m_len)]
        return numpy.hstack((matrix, bar_time, rel_time))

def build_network(name: str=None):
    rnn = tflearn.input_data(shape=[None, SEQUENCE_LENGTH, NOTE_RANGE])
    rnn = tflearn.dropout(rnn, DROPOUT, name="Dropout_pre")
    for i in range(NETWORK_DEPTH):
        if i < DOUBLE_WIDTH_LAYERS or i >= NETWORK_DEPTH+DOUBLE_WIDTH_LAYERS:
            rnn = tflearn.lstm(rnn, NETWORK_WIDTH*2, return_seq=(i != NETWORK_DEPTH-1), name="LSTM"+str(i))
        else:
            rnn = tflearn.lstm(rnn, NETWORK_WIDTH, return_seq=(i != NETWORK_DEPTH-1), name="LSTM"+str(i))
        rnn = tflearn.dropout(rnn, DROPOUT, name="Dropout"+str(i))
    rnn = tflearn.fully_connected(rnn, NOTE_RANGE, activation='softmax', name="FullyConnected")
    rnn = tflearn.regression(rnn, optimizer='adadelta', loss='mean_square', learning_rate=LEARNING_RATE)
    sqgen = tflearn.SequenceGenerator(rnn, {i: i for i in range(NOTE_RANGE)}, seq_maxlen=SEQUENCE_LENGTH, \
            checkpoint_path=os.path.join(NETWORK_FOLDER, "checkpoint", ""), tensorboard_dir=os.path.join(NETWORK_FOLDER, "logs"))
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
    gc.collect()
    network.fit(x, y, n_epoch=TRAINING_EPOCHS, show_metric=True, snapshot_epoch=True, validation_set=VALIDATION_SIZE)
    network.save(path)
    print("Trained network configuration saved to "+path)

def generate(nr_songs: int = 1, network_name: str=None):
    if check_network(network_name, False) == '':
        print("Cannot generate a new song without any network configuration (train the network first)")
        return
    print("Constructing the neural network")
    network = build_network(network_name)
    print("Gathering Random Seeds")
    songs = [s for s in read_all_inputs() if s.cleanup()]
    for _ in range(nr_songs):
        song = random.choice(songs)
        sequence = song.generate_tone_matrix()
        start = 0
        if not SEED_ONLY_BEGINNING:
            length, _ = sequence.shape
            start = random.randint(0, length-SEQUENCE_LENGTH-1)
        sequence = add_meta_to_matrix(sequence[start:start+SEQUENCE_LENGTH, :], SONG_LENGTH, BAR_LENGTH)
        if SEED_PROCESS:    # Processing seed in order to avoid replication
            x, y = sequence.shape
            for i in range(y):
                for j in range(x):
                    if sequence[j, i] > 0:
                        if (j > 0 and sequence[j-1, i] == 0) and (j < x-1 and sequence[j+1, i] == 0):
                            sequence[j, i] = 0
            sequence = numpy.flipud(sequence)
        generated = sequence
        print("Song {} used as Seed (starting from {} %)".format(song.name, round(start*100.0/song.length)))
        print("Generating a new sequence")
        for i in range(SEQUENCE_LENGTH, SONG_LENGTH):
            pred = numpy.array(network._predict([sequence]))
            prediction_to_timestep(pred, i, SONG_LENGTH)
            generated = numpy.vstack((generated, pred))
            sequence = generated[-SEQUENCE_LENGTH:, :]
            if i % 200 == 0 and i != 0:
                print("{} / {} timesteps generated".format(i, SONG_LENGTH))
        song = Song.convert_tone_matrix(generated, BAR_LENGTH)
        if network_name is not None:
            song.name = network_name+"-"+song.name
        file_name = save_and_convert_song(song)
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
        if len(over_mean) >= 1 / 2 and len(over_mean) < AVERAGE_TONE_DENSITY * 2:
            for i in over_mean:
                matrix[0, i] = 1
        else:
            sort_notes = [(num, i) for i, num in enumerate(notes)]
            sort_notes.sort(reverse=True)
            for _, i in over_mean[:AVERAGE_TONE_DENSITY]:
                matrix[0, i] = 1
    if META_TO_MATRIX == 1:
        matrix[0, -1] = index%(TIME_RESOLUTION*BAR_LENGTH)/(TIME_RESOLUTION*BAR_LENGTH)
    elif META_TO_MATRIX == 2:
        matrix[0, -2] = index%(TIME_RESOLUTION*BAR_LENGTH)/(TIME_RESOLUTION*BAR_LENGTH)
        matrix[0, -1] = float(index/length)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            generate()
        else:
            train()
    else:
        train()
