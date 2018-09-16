# MLusic - Music Generation with Deep Learning

## Dependencies
 - Python 3
 - Tensorflow
 - MidiCsv &nbsp; ([http://www.fourmilab.ch/webtools/midicsv/](http://www.fourmilab.ch/webtools/midicsv/))
 - Numpy

## Data
Both the learning data and the generated results are midi-files.
As a bridge between python and midi the tool from 
[http://www.fourmilab.ch/webtools/midicsv/](http://www.fourmilab.ch/webtools/midicsv/)
Before being used for learning the songs are stripped of any percussion, short decorative tracks or really short notes.

## Usage
 1. Put your midi files in the input folder
 2. Convert them to csv-files using: &nbsp; &nbsp; `python convert_input.py`  
    - If not on Windows you must compile the midi-csv utility first