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
        lines = file.readlines()
        self.ticks_per_quarter = int(lines[0][lines[0].rfind(",")+2:])
        self.name = file_name[file_name.find("/")+1:file_name.rfind(".")]
        self.instruments = []
        self.track_instrument = []
        track_key = dict()
        instrument_key = dict()
        self.tracks = []

        for line in lines:
            split = line.split(", ")
            if split[2] == "Note_on_c": #creating a note with zero length
                track_nr = int(split[0])
                time = int(split[1])
                channel = int(split[3])
                note = int(split[4])
                velocity = int(split[5])
                key = track_nr*256+channel
                try:
                    track_key[key].append([time, note, velocity, 0])
                except (KeyError, TypeError):
                    track_key[key] = [[time, note, velocity, 0]]
            elif split[2] == "Note_off_c": #setting the length of a note
                track_nr = int(split[0])
                time = int(split[1])
                channel = int(split[3])
                note = int(split[4])
                key = track_nr*256+channel
                track = track_key[key]
                index = len(track)-1
                while track[index][1] != note:
                    index -= 1
                track[index][-1] = time - track[index][0]
            elif split[2] == "Tempo":
                self.tempo = int(split[3])
            elif split[2] == "Time_signature":
                self.bar_length = int(split[3])
                self.beat_unit = 2**int(split[4])
            elif split[2] == "Program_c":
                channel = int(split[3])
                instrument = int(split[4])
                instrument_key[channel] = instrument

        for channel, instrument in instrument_key.items():
            if not instrument in self.instruments:
                self.instruments.append(instrument)
            instrument_key[channel] = self.instruments.index(instrument)
        for key, track in track_key.items():
            if len(track) > 0:
                self.track_instrument.append(int(instrument_key.get(key&255, 0)))
                self.tracks.append(track)

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
