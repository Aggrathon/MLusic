# MLusic - Music Generation with Deep Learning

## Dependencies
 - Python 3
 - Tensorflow
 - TFLearn ([http://tflearn.org/](http://tflearn.org/))
 - MidiCsv ([http://www.fourmilab.ch/webtools/midicsv/](http://www.fourmilab.ch/webtools/midicsv/))
 - Numpy

## Data
Both the input and the output is in the form of midicsv as formatted by the tool 
[http://www.fourmilab.ch/webtools/midicsv/](http://www.fourmilab.ch/webtools/midicsv/). 
The program will use all  *.csv files in the input folder as learning data, expecting them to be already converted midi files.
It outputs .csv files to the output folder ready to be converted to midi.

## Usage
 1. Put your learning midi files in the input folder and convert them to .csv  
   \- Optionally analyze the song composition using: `python input_analyzer.py`
 2. Train the neural network using: `python learn.py`
 3. Generate a new song using: `python learn.py generate`
 4. The new song will appear in the output folder as a csv (convert it to midi to listen to it)