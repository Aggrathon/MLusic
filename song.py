
import os
import datetime
from math import log2
from path import Path
import numpy as np

PERCUSSION_INSTRUMENT = 128

class Song(object):

    def __init__(self):
        self.bar_length = 4
        self.beat_unit = 4
        self.tempo = 500000
        self.ticks_per_quarter = 192
        self.instruments = []
        self.notes = [] #lists of onehot values [[instrument, note]]
        self.times = [] #lists of continous values [[time, duration, velocity]]

    def __start_note(self, index, time, track, channel, note, velocity):
        instrument = self.instruments[track*16+channel]
        self.notes[index, 0] = instrument
        self.notes[index, 1] = note
        self.times[index, 0] = time
        self.times[index, 2] = velocity / 127
        return index + 1

    def __end_note(self, index, time, track, channel, note):
        instrument = self.instruments[track*16+channel]
        for i in reversed(range(index)):
            if self.notes[i, 0] == instrument and self.notes[i, 1] == note and self.times[i, 0] < time:
                self.times[i, 1] = time - self.times[i, 0]
                return

    def read_csv_file(self, file):
        self.instruments = [0]*16
        self.instruments[9] = PERCUSSION_INSTRUMENT
        with open(file, "r", encoding="utf8", errors="ignore") as f:
            lines = f.readlines()
            if not lines or len(lines) == 0:
                print("Error reading file:", file)
                return self
        self.notes = np.zeros((len(lines), 2), dtype=np.int32)
        self.times = np.zeros((len(lines), 3), dtype=np.float32)
        header = lines[0].split(", ")
        self.instruments = self.instruments * (int(header[-2])+1)
        self.ticks_per_quarter = int(header[-1])
        index = 0
        for line in lines:
            split = line.split(", ")
            if split[2] == "Note_on_c": #creating a note with zero length
                velocity = int(split[-1])
                if velocity == 0:
                    self.__end_note(index, int(split[1]), int(split[0]), int(split[3]), int(split[4]))
                else:
                    index = self.__start_note(index, int(split[1]), int(split[0]), int(split[3]), int(split[4]), velocity)
            elif split[2] == "Note_off_c": #setting the length of a note
                self.__end_note(index, int(split[1]), int(split[0]), int(split[3]), int(split[4]))
            elif split[2] == "Tempo":
                self.tempo = int(split[3])
            elif split[2] == "Time_signature":
                self.bar_length = int(split[3])
                self.beat_unit = 2**int(split[4])
            elif split[2] == "Program_c":
                self.instruments[int(split[0])*16 + int(split[3])] = int(split[4])
        sort = np.argsort(self.times[:index, 0], axis=None)[self.times[:index, 1] > 0.001]
        self.notes = self.notes[sort, :]
        self.times = self.times[sort, :]
        to_ms = (60000 / (self.tempo * self.ticks_per_quarter)) #Convert to ms
        self.times[1:, 0] = (self.times[1:, 0] - self.times[:-1, 0]) * to_ms # Relative times
        self.times[:, 1] = self.times[:, 1] * to_ms
        self.times[0, 0] = 100
        self.instruments = sorted(list(set(self.instruments)))
        if len(self.instruments) > 10:
            i = self.instruments.index(PERCUSSION_INSTRUMENT)
            self.instruments[i] = self.instruments[9]
            self.instruments[9] = PERCUSSION_INSTRUMENT
            assert len(self.instruments) <= 16
        return self

    def save_to_file(self, file_name):
        instruments = {i: c for c, i in enumerate(self.instruments)}
        instruments[PERCUSSION_INSTRUMENT] = 9
        with open(file_name, "w") as file:
            file.write("0, 0, Header, 1, {}, {}\n".format(2, self.ticks_per_quarter))
            file.write("1, 0, Start_track\n")
            file.write("1, 0, Time_signature, {}, {}, 24, 8\n".format(self.bar_length, int(log2(self.beat_unit))))
            file.write("1, 0, Tempo, {}\n".format(self.tempo))
            file.write("1, 0, End_track\n")
            file.write("2, 0, Start_track\n")
            for i, ins in enumerate(self.instruments):
                if ins != PERCUSSION_INSTRUMENT:
                    file.write("2, 0, Program_c, {}, {}\n".format(i, ins))
            notes = []
            tot = -self.times[0, 0]
            to_tick = (self.tempo * self.ticks_per_quarter) / 60000
            for note, time in zip(self.notes, self.times):
                tot += time[0]
                tick = int(tot * to_tick)
                notes.append((tick, "2, {}, Note_on_c, {}, {}, {}\n".format(tick, instruments[note[0]], note[1], int(time[2]*127))))
                tick = int((tot+time[1]) * to_tick)
                notes.append((tick, "2, {}, Note_off_c, {}, {}, 0\n".format(tick, instruments[note[0]], note[1])))
            notes.sort()
            for _, line in notes:
                file.write(line)
            end = int(notes[-1][1].split(", ")[1])
            file.write("2, {}, End_track\n".format(end+1))
            file.write("0, {}, End_of_file\n".format(end+2))


if __name__ == "__main__":
    s = Song().read_csv_file("input/2_Minutes_To_Midnight.csv")
    s.save_to_file("output/example.csv")
