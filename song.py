import os

INPUT_FOLDER = "input/"
OUTPUT_FOLDER = "output/"


class Song(object):
    name = ""
    bar_length = 4
    beat_unit = 4
    tempo = 0
    ticks_per_quarter = 0
    instruments = []
    track_instrument = []
    tracks = []


    def __init__(self, file_name):
        file = open(file_name, "r")
        line = file.readline()
        self.ticks_per_quarter = int(line[line.rfind(",")+2:])
        self.name = file_name[file_name.find("/")+1:file_name.rfind(".")]
        self.instruments = []
        self.track_instrument = []
        track_key = dict()
        instrument_key = dict()
        self.tracks = []

        line = file.readline()
        while line != "":
            split = line.split(", ")

            if split[2] == "Note_on_c": #creating a note with zero length
                track = int(split[0])
                time = int(split[1])
                instrument = instrument_key[int(split[3])]
                note = int(split[4])
                velocity = int(split[5])
                key = track*256+instrument
                try:
                    track = track_key[key]
                except (KeyError, TypeError):
                    track = len(self.tracks)
                    self.tracks.append([])
                    track_key[key] = track
                    self.track_instrument.append(instrument)
                self.tracks[track].append([time, note, velocity, 0])

            elif split[2] == "Note_off_c": #setting the length of a note
                track = int(split[0])
                time = int(split[1])
                instrument = instrument_key[int(split[3])]
                note = int(split[4])
                key = track*256+instrument
                track = track_key[key]
                index = len(self.tracks[track])-1
                while self.tracks[track][index][1] != note:
                    index -= 1
                    if index < 0:
                        print(self.tracks[track])
                self.tracks[track][index][-1] = time - self.tracks[track][index][0]

            elif split[2] == "Tempo":
                self.tempo = int(split[3])

            elif split[2] == "Time_signature":
                self.bar_length = int(split[3])
                self.beat_unit = 2**int(split[4])

            elif split[2] == "Program_c":
                channel = int(split[3])
                instrument = int(split[4])
                try:
                    index = self.instruments.index(instrument)
                    instrument_key[channel] = index
                except ValueError:
                    instrument_key[channel] = len(self.instruments)
                    self.instruments.append(instrument)

            line = file.readline()
        file.close()


    def __repr__(self):
        return "<Song '{}'>".format(self.name)


    def save_to_file(self, file_name=None):
        if file_name is None:
            file_name = OUTPUT_FOLDER+self.name+".csv"
        file = open(file_name, "w")
        #generate csv
        file.close()


def read_all_inputs():
    return [Song(INPUT_FOLDER+s) for s in os.listdir(INPUT_FOLDER) if s.endswith(".csv")]
