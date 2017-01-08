# MLusic - Music Generation with Deep Learning
Create a LSTM neural network that can be trained to generate music.
MIDI files are used both as input and output. Since there might be no way of separating 
melody from "decorations" and accompaniment some cleaning is done before learning (like removing percussion).
This means the final result is  highly dependent on the learning material.
And overfitting with a large network is a real threat.

## Examples
The examples folder contains a sample trained network and some generated songs.

## Dependencies
 - Python 3
 - Tensorflow
 - TFLearn &nbsp; ([http://tflearn.org/](http://tflearn.org/))
 - MidiCsv &nbsp; ([http://www.fourmilab.ch/webtools/midicsv/](http://www.fourmilab.ch/webtools/midicsv/))
 - Numpy

## Data
Both the learning data and the generated results are midi-files.
As a bridge between python and midi the tool from 
[http://www.fourmilab.ch/webtools/midicsv/](http://www.fourmilab.ch/webtools/midicsv/)
is used (the windows binaries are included, for other platforms the tool must be compiled from source).
Before being used for learning the songs are stripped of any percussion, short decorative tracks or really short notes.

## Usage
 1. Put your learning midi files in the input folder
 2. Convert them to csv-files using: &nbsp; &nbsp; `python main.py convert input`  
    - If not on Windows you must compile the midi-csv utility first
 3. (Optionally) edit the config.py file to modify the training and generation procedures
 4. Train the neural network using: &nbsp; &nbsp; `python main.py train`
 5. Generate a new song using: &nbsp; &nbsp; `python main.py generate`
 6. To see all available options just run: &nbsp; &nbsp; `python main.py help`