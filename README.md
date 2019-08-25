# MLusic - Music Generation with Deep Learning

This branch explores using the transformer architecture for generating music.
The other branches contains attempts with other techniques.
Currently I don't yet have a convincing example (WIP).


## Dependencies

- Python 3
- Tensorflow (2.0)
- Numpy
- Pypianoroll


## Data

Both the learning data and the generated results are midi-files.
Pypianoroll is used to both parse and write the midi-files.
As the dataset you can use e.g. [LPD](https://salu133445.github.io/lakh-pianoroll-dataset/dataset).


## Usage

1. Install all the requirements in requirements.txt
1. Download and unzip the dataset to the `./input` directory.
1. Prepare the dataset using `python lpd.py convert`
1. Train a model using `python pianoroll_MODEL.py train` where MODEL is one of predict/gan/hybrid
   - The training can safely be stopped using `Control + C`
1. Generate music using `python pianoroll_MODEL.py generate`
1. Use, e.g., VLC to play the generated music
