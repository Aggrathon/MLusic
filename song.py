
import os
import datetime
from math import log2, floor
import numpy
from config import *

def reduce_instrument(instr):
    if instr > 31 and instr < 33: #Bass
        return 40
    if instr > 24 and instr < 29:  # Guitar
        return 32
    if instr == 10:
        return 10
    return instr

def expand_instument(instr):
    return instr


class Note():
    def __init__(self, instrument=0, tone=0, length=0, delay=0):
        self.instrument = instrument
        self.tone = tone
        self.length = length
        self.delay = delay


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
            instruments = list(0 for _ in range(100))

            for line in lines:
                split = line.split(", ")
                if split[2] == "Note_on_c": #creating a note with zero length
                    time = int(split[1])
                    channel = int(split[3])
                    note = int(split[4])
                    notes.append(Note(channel, note, 0, time))
                elif split[2] == "Note_off_c": #setting the length of a note
                    time = int(split[1])
                    channel = int(split[3])
                    tone = int(split[4])
                    for note in reversed(notes):
                        if note.instrument == channel and note.tone == tone:
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
                note.instrument = instruments[note.instrument]  #Convert channels to instruments
                t = note.delay - time
                time = note.delay
                note.delay = t  #Convert time to delay since last note
            
            self.instruments = set(i for i in instruments if i > 0)
            self.notes = notes
            self.length = time + notes[-1].length
        return self

    def save_to_file(self, file_name=None) -> str:
        instument_set = set()
        for note in self.notes:
            instument_set.add(note.instrument)
        instuments = {ins: i for i, ins in enumerate(instument_set)}
        if file_name is None:
            file_name = os.path.join(OUTPUT_FOLDER, self.name+".csv")
        with open(file_name, "w") as file:
            file.write("0, 0, Header, 1, {}, {}\n".format(len(self.tracks)+1, self.ticks_per_quarter))
            file.write("1, 0, Start_track\n")
            file.write("1, 0, Time_signature, {}, {}, 24, 8\n".format(self.bar_length, int(log2(self.beat_unit))))
            file.write("1, 0, Tempo, {}\n".format(self.tempo))
            for ins, i in instruments:
                file.write("1, 0, Program_c, {}, {}\n".format(i, ins))
            file.write("1, 0, End_track\n")
            file.write("2, 0, Start_track\n")
            notelist=[]
            time = 0
            for note in self.notes:
                time += note.delay
                notelist.append((instuments[note.instrument], time, note.tone, True))
                notelist.append((instuments[note.instrument], time+note.length, note.tone, False))
            notelist.sort(key=lambda n: n[1])
            for note in notelist:
                if note[-1]:
                    file.write("2, {}, Note_on_c, {}, {}, 80\n".format(note[1], note[0], note[2]))
                else:
                    file.write("2, {}, Note_off_c, {}, {}, 0\n".format(note[1], note[0], note[2]))
            file.write("2, {}, End_track\n".format(self.length))
            file.write("0, 0, End_of_file\n")
        return file_name

    def import_cleanup(self, reduce_instruments=True, remove_percussion=True, minimum_note_length=1):
        if reduce_instruments:
            for note in self.notes:
                note.instrument = reduce_instrument(note.instrument)
        self.notes = [n for n in self.notes if \
            not (remove_percussion and n.instrument == 10) and \
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
