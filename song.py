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
    tracks = []

    def __init__(self, file_name):
        file = open(file_name, "r")
        line = file.readline()
        self.ticks_per_quarter = int(line[line.rfind(",")+2:])
        self.name = file_name[file_name.find("/")+1:file_name.rfind(".")]
        self.instruments = []

        line = file.readline()
        while line != "":
            split = line.split(", ")
            if split[2] == "Note_on_c":
                pass
            elif split[2] == "Note_off_c":
                pass
            elif split[2] == "Tempo":
                self.tempo = int(split[3])
            elif split[2] == "Time_signature":
                self.bar_length = int(split[3])
                self.beat_unit = 2**int(split[4])
            elif split[2] == "Program_c":
                channel = int(split[3])
                while len(self.instruments) <= channel:
                    self.instruments.append(-1)
                self.instruments[channel] = int(split[4])
            elif split[2] == "Key_signature":
                pass
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
