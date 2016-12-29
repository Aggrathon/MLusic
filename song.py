
import os
from math import log2
from collections import Counter
import numpy

INPUT_FOLDER = "input/"
OUTPUT_FOLDER = "output/"
PERCUSSION_INSTRUMENT = -10


class Song(object):
    name = ""
    bar_length = 4
    beat_unit = 4
    tempo = 0
    ticks_per_quarter = 0
    instruments = []
    track_instrument = []
    tracks = []
    track_volume = []
    length = 0


    def __init__(self, file_name):
        file = open(file_name, "r")
        lines = file.readlines()
        self.ticks_per_quarter = int(lines[0][lines[0].rfind(",")+2:])
        self.name = file_name[file_name.find("/")+1:file_name.rfind(".")]
        self.instruments = []
        self.track_instrument = []
        track_key = dict()
        instrument_key = dict()
        instrument_key[10] = PERCUSSION_INSTRUMENT
        self.tracks = []
        self.track_volume = []
        self.length = 1
        volumes = dict()

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
            elif split[2] == "Control_c" and split[4] == "7":
                channel = int(split[3])
                vol1 = int(split[5])
                vol2 = volumes.get(channel, 0)
                if vol1 > vol2:
                    volumes[channel] = vol1

        for channel, instrument in instrument_key.items():
            if not instrument in self.instruments:
                self.instruments.append(instrument)
            instrument_key[channel] = self.instruments.index(instrument)
        for key, track in track_key.items():
            if len(track) > 0:
                length = track[-1][0]+track[-1][-1]+1
                if length > 0:
                    try:
                        self.track_instrument.append(instrument_key[key&255])
                    except KeyError:
                        self.track_instrument.append(instrument_key.get(10, 0))
                    self.track_volume.append(volumes.get(key&255, 127))
                    self.tracks.append(track)
                    if self.length < length:
                        self.length = length
        file.close()


    def __repr__(self):
        return "<Song '{}'>".format(self.name)


    def save_to_file(self, file_name=None):
        if file_name is None:
            file_name = OUTPUT_FOLDER+self.name+".csv"
        file = open(file_name, "w")
        file.write("0, 0, Header, 1, {}, {}\n".format(len(self.tracks), self.ticks_per_quarter))
        file.write("1, 0, Start_track\n")
        file.write("1, 0, Time_signature, {}, {}, 24, 8\n".format(self.bar_length, int(log2(self.beat_unit)) ))
        file.write("1, 0, Tempo, {}\n".format(self.tempo))
        for i, ins in enumerate(self.instruments):
            if ins != PERCUSSION_INSTRUMENT:
                file.write("1, 0, Program_c, {}, {}\n".format(i, ins))
        file.write("1, 0, End_track\n")
        for i, track in enumerate(self.tracks):
            track_nr = i+2
            file.write("{}, 0, Start_track\n".format(track_nr))
            notes = []
            for note in track:
                notes.append((note[0], "{}, {}, Note_on_c, {}, {}, {}\n".format(track_nr, note[0], self.track_instrument[i], note[1], note[2])))
                notes.append((note[0]+note[-1], "{}, {}, Note_off_c, {}, {}, 0\n".format(track_nr, note[0]+note[-1], self.track_instrument[i], note[1])))
            notes.sort()
            for _, line in notes:
                file.write(line)
            file.write("{}, {}, End_track\n".format(track_nr, notes[-1][0]+1))
        file.write("0, 0, End_of_file\n")
        file.close()

    def cleanup(self, piano_only=True, smallest_note = 8):
        #Remove percussion
        self.tracks = [t for i, t in enumerate(self.tracks) if \
            self.instruments[self.track_instrument[i]] < 97 and \
            self.instruments[self.track_instrument[i]] != PERCUSSION_INSTRUMENT]
        #Remove short notes and tracks
        for track in self.tracks:
            track = [t for t in track if \
                t[2] > 0 and t[3] > 0 and  \
                int(self.ticks_per_quarter / t[3] * 4) <= smallest_note]
            conc_over_sound, conc_over_length = __track_concurrency__(track)
            if conc_over_sound < 0.3 or conc_over_length < 0.3:
                track.clear()
        #Remove quiet tracks
        specified_volumes = [v for v in self.track_volume if v != 127]
        min_volume = numpy.mean(specified_volumes)-numpy.std(specified_volumes)
        for i, track in enumerate(self.tracks):
            if self.track_volume[i] < min_volume:
                track.clear()
        #Remove empty tracks
        self.tracks = [t for t in self.tracks if len(t) > 0]
        #Decrease the resolution to 1/(2*smallest_note) notes
        self.length = 1
        smallest_note = 2*smallest_note
        for track in self.tracks:
            for t in track:
                t[0] = round(t[0]*smallest_note/self.ticks_per_quarter)
                t[-1] = round(t[-1]*smallest_note/self.ticks_per_quarter)
            end = track[-1][0]+1+track[-1][-1]
            if self.length < end:
                self.length = end
        self.ticks_per_quarter = smallest_note
        #Optionally remove instruments
        if piano_only:
            self.instruments = [0]
            self.track_instrument = [0] * len(self.tracks)


def __track_length__(track):
    return track[-1][0]+track[-1][-1]-track[0][0]

def __track_concurrency__(track, length = 0):
    if len(track) == 0:
        return 0, 0
    if length == 0:
        length = __track_length__(track)
        if length == 0:
            length = 1
    start = 0
    end = 0
    coverage = 0
    tone_length = 0
    for n in track:
        if n[0] > end:
            coverage += end-start
            start = n[0]
            end = n[0] + n[-1]
        else:
            new_end = n[0] + n[-1]
            if new_end > end:
                end = new_end
        tone_length += n[-1]
    coverage += end-start
    if coverage == 0:
        coverage = 1
    return tone_length/coverage, tone_length/length


def read_all_inputs():
    return [Song(INPUT_FOLDER+s) for s in os.listdir(INPUT_FOLDER) if s.endswith(".csv")]
