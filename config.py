
"""
Song Cleaning Config
    - These configs will change how the learning data is prepared for training
"""
# Discard short songs
MIN_SONG_LENGTH = 50
# Discard songs with tempos much larger than the standard 500000
MAX_SONG_TEMPO = 1000000
# Discard tracks that covers only short portions of the song
MIN_TRACK_COVERAGE = 0.3
# Discard tones that lasts forever
MAX_TONE_LENGTH = 128
# Discard any song that is not in common time (Common time == 4/4)
ENFORCE_COMMON_TIME = True


"""
Song Generation
    - These configs will change how new songs are generated
"""
SONG_LENGTH = 6000
INSTRUMENT = 0      # 48
AVERAGE_TONE_DENSITY = 4
RANDOMNESS = 0.15    # 0.0 - 0.5
SEED_ONLY_BEGINNING = False


"""
Neural Network Training
    - These configs change how the network is trained
"""
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 2
VALIDATION_SIZE = 0.12


"""
Neural Network Layout
    - Changing these will require you to recreate and retrain any existing network
"""
# Number of nodes per hidden LSTM layer
NETWORK_WIDTH = 128
# Number of hidden LSTM layers
NETWORK_DEPTH = 6
# Number of hidden LSTM layers that have their width doubled (doubled < depth)
DOUBLE_WIDTH_LAYERS = 2
# Combat overfitting by randomly turning off nodes during training [0.0...1.0[
DROPOUT = 0.7       # 0.0 - 0.9
# How many past timesteps should be used to predict the next one
SEQUENCE_LENGTH = 192
# The size of the timesteps, in note notation this would be 1/resolution:th notes
TIME_RESOLUTION = 16
# Midi Note Range
LOWEST_NOTE = 24
HIGHEST_NOTE = 104
# Should additional metadata be added to the data?
#       0   :   No metadata
#       1   :   Position in bar
#       2   :   Position in bar and relative time of the whole song
META_TO_MATRIX = 1


"""
File Location
    - These will change were the software is looking for and saving files
"""
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
NETWORK_FOLDER = "network"
