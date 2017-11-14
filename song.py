
import os
from math import log2
import numpy as np
from config import INPUT_FOLDER, OUTPUT_FOLDER, MAX_INSTRUMENTS, MAX_TONE


def reduce_instrument(instr):
    """
        Reduces the range of instuments
    """
    if instr > 32 and instr < 41: #Bass
        return 2
    if instr > 24 and instr < 33:  # Guitar
        return 3
    if instr == 128:
        return 0
    if instr < 9:
        return 4
    return 1

def expand_instument(instr):
    """
        Opposite of reduce_instuments
    """
    if instr == 2:
        return 40
    if instr == 3:
        return 32
    if instr == 1:
        return 53
    if instr == 4:
        return 1
    return instr


class Note():
    """
        Class for containing a note in a sequence
    """
    def __init__(self, instrument=0, tone=0, length=0, delay=0, channel=0):
        self.instrument = instrument
        self.tone = tone
        self.length = length
        self.delay = delay
        self.channel = channel
    
    def to_vector(self, max_instr=MAX_INSTRUMENTS, max_tone=MAX_TONE):
        """
            Get the vector representing the note
        """
        vec = np.zeros(max_instr+max_tone+2, np.float)
        vec[-2] = float(self.length)
        vec[-1] = float(self.delay)
        vec[self.instrument] = 1.0
        vec[max_instr+self.tone] = 1.0
        return vec

    @classmethod
    def from_vector(cls, vector, max_instr=MAX_INSTRUMENTS, max_tone=MAX_TONE):
        """
            Convert a matrix into a note
        """
        return Note(
            np.argmax(vector[:max_instr]),
            np.argmax(vector[max_instr:max_instr+max_tone]),
            int(vector[-2]),
            int(vector[-1])
        )


class Song(object):

    def __init__(self, name):
        self.bar_length = 4
        self.beat_unit = 4
        self.tempo = 500000
        self.ticks_per_quarter = 4
        self.name = name
        self.length = 0
        self.notes = []
        self.instruments = set()

    def __repr__(self):
        return "<Song '{}'>".format(self.name)

    @staticmethod
    def read_csv_file(file_name):
        self = Song(file_name[file_name.rfind(os.path.sep)+1:file_name.rfind(".")])
        with open(file_name, "r") as file:
            lines = file.readlines()
            self.ticks_per_quarter = int(lines[0][lines[0].rfind(",")+2:])

            notes = []
            instruments = list(-1 for _ in range(100))
            instruments[9] = 128
            instruments[10] = 128

            for line in lines:
                split = line.split(", ")
                if split[2] == "Note_on_c": #creating a note with zero length
                    time = int(split[1])
                    channel = int(split[3])
                    note = int(split[4])
                    notes.append(Note(instruments[channel], note, 1000, time, channel))
                elif split[2] == "Note_off_c": #setting the length of a note
                    time = int(split[1])
                    channel = int(split[3])
                    tone = int(split[4])
                    for note in reversed(notes):
                        if note.channel == channel and note.tone == tone:
                            note.length = time - note.delay
                            break
                elif split[2] == "Tempo":
                    if split[0] == "0":
                        self.tempo = int(split[3])
                elif split[2] == "Time_signature":
                    self.bar_length = int(split[3])
                    self.beat_unit = 2**int(split[4])
                elif split[2] == "Program_c":
                    channel = int(split[3])
                    instrument = int(split[4])
                    instruments[channel] = instrument
            
            notes.sort(key=lambda n: n.delay)
            time = notes[0].delay
            for note in notes:
                t = note.delay - time
                time = note.delay
                note.delay = t  #Convert time to delay since last note
            
            self.instruments = set(n.instrument for n in notes)
            self.notes = notes
            time = 0
            for note in notes:
                time += note.delay
            self.length = time + notes[-1].length
        return self

    def save_to_file(self, file_name=None) -> str:
        instument_set = set()
        for note in self.notes:
            instument_set.add(note.instrument)
        if 0 in instument_set:
            instument_set.remove(0)
        instruments = {ins: i for i, ins in enumerate(instument_set)}
        if file_name is None:
            file_name = os.path.join(OUTPUT_FOLDER, self.name+".csv")
        with open(file_name, "w") as file:
            file.write("0, 0, Header, 1, {}, {}\n".format(2, self.ticks_per_quarter))
            file.write("1, 0, Start_track\n")
            file.write("1, 0, Time_signature, {}, {}, 24, 8\n".format(self.bar_length, int(log2(self.beat_unit))))
            file.write("1, 0, Tempo, {}\n".format(self.tempo))
            for ins, i in instruments.items():
                file.write("1, 0, Program_c, {}, {}\n".format(i, ins))
            file.write("1, 0, End_track\n")
            file.write("2, 0, Start_track\n")
            notelist=[]
            time = 0
            for note in self.notes:
                time += note.delay
                if note.instrument != 0:
                    notelist.append((instruments[note.instrument], time, note.tone, True))
                    notelist.append((instruments[note.instrument], time+note.length, note.tone, False))
            notelist.sort(key=lambda n: n[1])
            for note in notelist:
                if note[-1]:
                    file.write("2, {}, Note_on_c, {}, {}, 80\n".format(note[1], note[0], note[2]))
                else:
                    file.write("2, {}, Note_off_c, {}, {}, 0\n".format(note[1], note[0], note[2]))
            file.write("2, {}, End_track\n".format(notelist[-1][1]))
            file.write("0, {}, End_of_file\n".format(notelist[-1][1]+1))
        return file_name

    def import_cleanup(self, reduce_instruments=True, remove_percussion=True, minimum_note_length=1):
        if reduce_instruments:
            for note in self.notes:
                note.instrument = reduce_instrument(note.instrument)
        self.notes = [n for n in self.notes if \
            not (remove_percussion and n.instrument == 128) and \
            not n.length < minimum_note_length]
        self.instruments = set(n.instrument for n in self.notes)

    def export_cleanup(self, reduced_instruments=True):
        time = 0
        if reduced_instruments:
            for note in self.notes:
                note.instrument = expand_instument(note.instrument)
            self.instruments = set(n.instrument for n in self.notes)
        for note in self.notes:
            time += note.delay
        self.length = time + self.notes[-1].length


def read_all_inputs():
    return [Song.read_csv_file(os.path.join(INPUT_FOLDER, s))
            for s in os.listdir(INPUT_FOLDER) if s.endswith(".csv")]

def get_all_vectors():
    print("Converting csvs to songs")
    songs = read_all_inputs()
    for s in songs:
        s.import_cleanup()
    print("Converting notes to vectors")
    return np.asarray([note.to_vector() for song in songs for note in song.notes], np.float)
