
"""
Song Cleaning Config
    - These configs will change how the learning data is prepared for training
"""
# Discard short songs
MIN_SONG_LENGTH = 400
# Discard songs with tempos much larger than the standard 500000
MAX_SONG_TEMPO = 1000000
# Discard tracks that covers only short portions of the song
MIN_TRACK_COVERAGE = 0.3
# Discard tones that lasts forever
MAX_TONE_LENGTH = 64
# Discard any song that is not in common time (Common time == 4/4)
ENFORCE_COMMON_TIME = True
# Break overlapping notes of the same tone by adding silence between
ADD_SILENCE_BEFORE = False
# Instead of a binary value for notes allow a note to scale in the tone matrix
ALLOW_NOTE_SCALING = False


"""
Song Generation
    - These configs will change how new songs are generated
"""
# How long should the generated sample be (in ticks, TIME_RESOLUTION == ticks per quarter)
SONG_LENGTH = 1000
# The generated song use these instruments (match the instruments to the generated tracks, see below)
#       0   :   Piano
#       48  :   String Ensamble
#       33  :   Electric Bass
#       30  :   Distortion Guitar
#       29  :   Overdriven Guitar
INSTRUMENT = [30,33]
# The generated track width will be between 1 and DENSITY*2
AVERAGE_TONE_DENSITY = 3
# Shift the likelyhood that a tone will play by a relative amount
RANDOMNESS = 0.15    # 0.0 - 0.5
# Bar length of the generated song
BAR_LENGTH = 4
# If the generated song contains notes much longer than a bar
#  split it up into smaller, bar-sized, chunks
BREAK_LONG_NOTES = True
# The generation seeds will only start at the beginning of the input tracks
SEED_ONLY_BEGINNING = False
# Process Seed in order to avoid replication,
#  the seed will be inverted and too short notes will be removed
SEED_PROCESS = True


"""
Neural Network Training
    - These configs change how the network is trained
"""
LEARNING_RATE = 0.1
# How many times will the training process iterate through all inputs
TRAINING_EPOCHS = 3
VALIDATION_SIZE = 0.12


"""
Neural Network Layout
    - Changing these will require you to recreate and retrain any existing network
"""
# Number of nodes per hidden LSTM layer
# (base this number on the sum of track widths, see below)
NETWORK_WIDTH = 100
# Number of hidden LSTM layers
NETWORK_DEPTH = 3
# Number of hidden LSTM layers that have their width doubled (doubled < depth, negative for reverse selection)
DOUBLE_WIDTH_LAYERS = 0
# Combat overfitting by randomly turning off nodes during training [0.0...1.0[
DROPOUT = 0.7       # 0.0 - 0.9
# How many past timesteps should be used to predict the next one
SEQUENCE_LENGTH = 128
# The size of the timesteps, in note notation this would be 1/resolution:th notes
TIME_RESOLUTION = 8
# You can configure the network to generate multiple tracks (with different instruments, see above)
# Each track has a different tone range
# The selector function determines if a track in the input belongs to the output track
# The last output track should have a selector that always returns true
import numpy as np
TRACK_TO_DATA = [
    {"lowest_note":24, "highest_note":64, "selector": lambda track: np.mean([n[1] for n in track]) < 46.5}, # acccomp
    {"lowest_note":35, "highest_note":95, "selector": lambda track: True}                                   # melody
]
# Should additional metadata be added to the data?
#       0   :   No metadata
#       1   :   Position in bar
#       2   :   Position in bar and relative time of the whole song
META_TO_MATRIX = 2


"""
File Location
    - These will change were the software is looking for and saving files
"""
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
NETWORK_FOLDER = "network"
