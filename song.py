"""
    Script for processing files
"""

from math import log2
from path import Path
import numpy as np

PERCUSSION_INSTRUMENT = 128
RESOLUTION = 40

class FileFormatException(Exception):
    """
        Exception for when a file is not in the expected format
    """
    pass

class Song():
    """
        Class for processing songs between midi and data formats
    """

    def __init__(self):
        self.bar_length = 4
        self.beat_unit = 4
        self.tempo = 500000
        self.ticks_per_quarter = 192
        self.instruments = []
        self.notes = np.zeros((0, 2), dtype=np.int32) #array of onehot values [[instrument, note]]
        self.times = np.zeros((0, 3), dtype=np.float32) #array of continous values [[time, duration, velocity]]

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

    def read_csv(self, filename):
        """
        Read a midi csv and process the notes

        Arguments:
            filename {str} -- The input midi csv

        Returns:
            Song -- self for chaining
        """
        self.instruments = [0]*16
        self.instruments[9] = PERCUSSION_INSTRUMENT
        with open(filename, "r", encoding="utf8", errors="ignore") as file:
            lines = file.readlines()
            if not lines:
                raise FileFormatException(filename)
        self.notes = np.zeros((len(lines), 2), dtype=np.int32)
        self.times = np.zeros((len(lines), 3), dtype=np.float32)
        header = lines[0].split(", ")
        if len(header) != 6 or header[2] != "Header":
            raise FileFormatException(filename)
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
        self.times[0, 0] = 0.1
        self.times[:, 0] = np.round(self.times[:, 0]*RESOLUTION)/RESOLUTION
        self.times[:, 1] = np.round(self.times[:, 1]*RESOLUTION + 0.2)/RESOLUTION
        self.instruments = sorted(list(set(self.instruments)))
        if len(self.instruments) > 10:
            i = self.instruments.index(PERCUSSION_INSTRUMENT)
            self.instruments[i] = self.instruments[9]
            self.instruments[9] = PERCUSSION_INSTRUMENT
            assert len(self.instruments) <= 16
        return self

    def save_midi(self, file_name="song.csv"):
        """
        Save the song as a csv midi file

        Arguments:
            file_name {str} -- The output file

        Returns:
            Song -- self for chaining
        """
        instruments = {i: c for c, i in enumerate(self.instruments)}
        instruments[PERCUSSION_INSTRUMENT] = 9
        max_index = len(self.instruments)
        if max_index > 25:
            index = 25
            while index < len(self.instruments):
                instruments[self.instruments[index]] = max_index
                max_index += 1
                index += 16
        tracks = dict()
        for i in self.instruments:
            tracks[i] = instruments[i] // 16 + 2
            instruments[i] = instruments[i] % 16
        num_tracks = (max_index-1)//16+1
        with open(file_name, "w") as file:
            file.write("0, 0, Header, 1, {}, {}\n".format(2, self.ticks_per_quarter))
            file.write("1, 0, Start_track\n")
            file.write("1, 0, Time_signature, {}, {}, 24, 8\n".format(self.bar_length, int(log2(self.beat_unit))))
            file.write("1, 0, Tempo, {}\n".format(self.tempo))
            file.write("1, 0, End_track\n")
            end = 0
            for i in range(num_tracks):
                file.write("{}, 0, Start_track\n".format(i + 2))
                for  ins in self.instruments:
                    j = tracks[ins]
                    if ins != PERCUSSION_INSTRUMENT and j == i+2:
                        file.write("{}, 0, Program_c, {}, {}\n".format(j, instruments[ins], ins))
                notes = []
                tot = -self.times[0, 0]
                to_tick = (self.tempo * self.ticks_per_quarter) / 60000
                for note, time in zip(self.notes, self.times):
                    tot += time[0]
                    track = tracks[note[0]]
                    if track != i+2:
                        continue
                    tick = int(tot * to_tick)
                    instrument = instruments[note[0]]
                    notes.append((tick, "{}, {}, Note_on_c, {}, {}, {}\n"
                                .format(track, tick, instrument, note[1], int(time[2]*127))))
                    tick = int((tot+time[1]) * to_tick)
                    notes.append((tick, "{}, {}, Note_off_c, {}, {}, 0\n".format(track, tick, instrument, note[1])))
                notes.sort()
                for _, line in notes:
                    file.write(line)
                end = max(end, int(notes[-1][1].split(", ")[1]))
                file.write("{}, {}, End_track\n".format(i + 2, end+1))
            file.write("0, {}, End_of_file\n".format(end+2))
        return self

    def save_data(self, filename="data.csv"):
        """
        Save the processed notes to a data file (.csv format)

        Arguments:
            filename {str} -- The output file

        Returns:
            Song -- self for chaining
        """
        with open(filename, "w", encoding="utf8") as file:
            #file.write('"deltaTime","duration","instrument","tone","velocity"')
            for note, time in zip(self.notes, self.times):
                file.write("%f,%f,%d,%d,%f\n"%(time[0], time[1], note[0], note[1], time[2]))
        return self

    def read_data(self, filename):
        """
        Read the notes from a processed data file

        Arguments:
            filename {str} -- The file to read

        Returns:
            Song -- self for chaining
        """
        with open(filename, "r", encoding="utf8") as file:
            lines = file.readlines()[1:]
        notes = np.zeros((len(lines), 2), dtype=np.int32)
        times = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            split = line.split(",")
            times[i, 0] = float(split[0])
            times[i, 1] = float(split[1])
            notes[i, 0] = int(split[2])
            notes[i, 1] = int(split[3])
            times[i, 2] = float(split[4])
        self.set_data(times, notes)
        return self
    
    def set_data(self, times, notes):
        """
        Set the notes directly

        Arguments:
            times {np.array} -- Numpy array with the note times
            notes {np.array} -- Numpy array with the note ton

        Returns:
            Song -- self for chaining
        """
        self.times = times
        self.notes = notes
        self.instruments = list(np.unique(self.notes[:, 0]))
        perc = self.instruments.index(PERCUSSION_INSTRUMENT)
        if perc > 9:
            self.instruments[perc] = self.instruments[9]
            self.instruments[9] = PERCUSSION_INSTRUMENT
        return self


    def combine(self, other):
        """
        Add the notes from another song at the end

        Arguments:
            other {Song} -- The song with the notes to add

        Returns:
            Song -- self for chaining
        """
        self.set_data(np.concatenate((self.times, other.times), 0), np.concatenate((self.notes, other.notes), 0))
        return self

    @staticmethod
    def read_folder(folder):
        """
        Read all midi csvs in afolderand combine them

        Arguments:
            folder {str} -- The folder to read from

        Returns:
            Song -- A song with the combined folder content
        """
        folder = Path(folder)
        song = Song()
        for file in folder.files("*.csv"):
            try:
                song.combine(Song().read_csv(file))
            except FileFormatException:
                pass
        return song
