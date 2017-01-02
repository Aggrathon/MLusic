
# Song Cleaning Config
#   - These configs will change how the learning data is prepared for training
MIN_SONG_LENGTH = 50
MAX_SONG_TEMPO = 1000000
MIN_TRACK_COVERAGE = 0.3
MAX_TONE_LENGTH = 128
ENFORCE_COMMON_TIME = True  #Common time == 4/4

# Song Generation
#   - These configs will change how new songs are generated
SONG_LENGTH = 600

# Neural Network Training
#   - These configs change how the network is trained
LEARNING_RATE = 0.005
TRAINING_EPOCHS = 1
VALIDATION_SIZE = 0.1

# Neural Network Layout
#   - Changing these will require you to recreate and retrain any existing network
NETWORK_WIDTH = 512
NETWORK_DEPTH = 3
DROPOUT = 0.5
SEQUENCE_LENGTH = 128
LOWEST_NOTE = 24
HIGHEST_NOTE = 104
TIME_RESOLUTION = 16
ADD_META_TO_MATRIX = True

# File Location
#   - These will change were the software is looking for and saving files
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
NETWORK_FOLDER = "network"
