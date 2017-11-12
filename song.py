
import os
import datetime
from math import log2, floor
import numpy
from config import *
from multiprocessing import Pool

PERCUSSION_INSTRUMENT = -10


class Song(object):
    name = ""
    bar_length = 4
    beat_unit = 4
    tempo = 500000
    ticks_per_quarter = 4
    instruments = []
    track_instrument = []
    tracks = []
    track_volume = []
    length = 1


    def __init__(self, name):
        self.bar_length = 4
        self.beat_unit = 4
        self.tempo = 500000
        self.ticks_per_quarter = 4
        self.name = name
        self.instruments = []
        self.track_instrument = []
        self.tracks = [] #lists of notes, note = [time, note, velocity, duration]
        self.track_volume = []
        self.length = 1

    def __repr__(self):
        return "<Song '{}'>".format(self.name)

    @staticmethod
    def read_csv_file(file_name):
        self = Song(file_name[file_name.rfind(os.path.sep)+1:file_name.rfind(".")])
        file = open(file_name, "r")
        lines = file.readlines()
        self.ticks_per_quarter = int(lines[0][lines[0].rfind(",")+2:])

        track_key = dict()
        instrument_key = dict()
        instrument_key[10] = PERCUSSION_INSTRUMENT
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
                    track_key[key].append([time, note, velocity, self.ticks_per_quarter])
                except (KeyError, TypeError):
                    track_key[key] = [[time, note, velocity, self.ticks_per_quarter]]
            elif split[2] == "Note_off_c": #setting the length of a note
                track_nr = int(split[0])
                time = int(split[1])
                channel = int(split[3])
                tone = int(split[4])
                key = track_nr*256+channel
                track = track_key[key]
                for note in reversed(track):
                    if note[1] == tone:
                        note[-1] = time - note[0]
            elif split[2] == "Tempo":
                if split[0] == "0":
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
        return self

    def save_to_file(self, file_name=None) -> str:
        if file_name is None:
            file_name = os.path.join(OUTPUT_FOLDER, self.name+".csv")
        file = open(file_name, "w")
        file.write("0, 0, Header, 1, {}, {}\n".format(len(self.tracks)+1, self.ticks_per_quarter))
        file.write("1, 0, Start_track\n")
        file.write("1, 0, Time_signature, {}, {}, 24, 8\n".format(self.bar_length, int(log2(self.beat_unit))))
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
            if len(notes) > 0:
                notes.sort()
                for _, line in notes:
                    file.write(line)
                file.write("{}, {}, End_track\n".format(track_nr, notes[-1][0]+1))
            else:
                file.write("{}, {}, End_track\n".format(track_nr, 1))
        file.write("0, 0, End_of_file\n")
        file.close()
        return file_name

    def cleanup(self, one_instrument=False, smallest_note=8):
        #Remove percussion
        for i, track in enumerate(self.tracks):
            instrument = self.instruments[self.track_instrument[i]]
            if instrument > 96 or instrument == PERCUSSION_INSTRUMENT:
                track.clear()
        #Remove quiet tracks
        specified_volumes = [v for i, v in enumerate(self.track_volume) if v != 127 and len(self.tracks[i]) > 0]
        if len(specified_volumes) > 0:
            min_volume = numpy.mean(specified_volumes)-numpy.std(specified_volumes)
            for i, track in enumerate(self.tracks):
                if self.track_volume[i] < min_volume:
                    track.clear()
        #Decrease the resolution to 1/(2*smallest_note) notes
        smallest_note = smallest_note/2 #relative to quarter notes
        if self.ticks_per_quarter > smallest_note:
            for track in self.tracks:
                for t in track:
                    t[0] = round(t[0]*smallest_note/self.ticks_per_quarter)
                    t[-1] = round(t[-1]*smallest_note/self.ticks_per_quarter)
                # Remove too short or too long notes
                track = [n for n in track if n[2] > 0 and n[-1] > 0 and n[-1] < MAX_TONE_LENGTH]
            self.ticks_per_quarter = smallest_note
        #Calculate new length
        self.length = 1
        for track in self.tracks:
            end = 0
            for n in track:
                new_end = n[0]+1+n[-1]
                if new_end > end:
                    end = new_end
            if self.length < end:
                self.length = end
        #Remove short tracks
        for track in self.tracks:
            conc_over_sound, conc_over_length = track_concurrency(track)
            if conc_over_sound < MIN_TRACK_COVERAGE or conc_over_length < MIN_TRACK_COVERAGE:
                track.clear()
        #Remove empty tracks
        new_tracks = []
        new_volumes = []
        new_instruments = []
        for i, track in enumerate(self.tracks):
            if len(track) > 0:
                new_tracks.append(track)
                new_volumes.append(self.track_volume[i])
                new_instruments.append(self.track_instrument[i])
        self.tracks = new_tracks
        self.track_volume = new_volumes
        self.track_instrument = new_instruments
        #Optionally remove instruments
        if one_instrument:
            self.instruments = INSTRUMENT[:1]
            self.track_instrument = [0] * len(self.tracks)
        else:
            ninst = []
            for i, ti in enumerate(self.track_instrument):
                if self.instruments[ti] in ninst:
                    self.track_instrument[i] = ninst.index(self.instruments[ti])
                else:
                    self.track_instrument[i] = len(ninst)
                    ninst.append(self.instruments[ti])
            self.instruments = ninst
        #Remove silence in the beginning
        start = self.length
        for t in self.tracks:
            if start > t[0][0]:
                start = t[0][0]
        if ADD_SILENCE_BEFORE:
            start -= 1
        if start > 0:
            self.length -= start
            for track in self.tracks:
                for n in track:
                    n[0] -= start
        #Return wether this song is usable
        return \
            self.tempo < MAX_SONG_TEMPO and \
            len(self.tracks) > 0 and \
            self.length/self.ticks_per_quarter > MIN_SONG_LENGTH and \
            (not ENFORCE_COMMON_TIME or (self.bar_length == 4 and self.beat_unit == 4))

    def generate_tone_matrix(self):
        self.cleanup(False, TIME_RESOLUTION)
        matrix = numpy.zeros(shape=(self.length, numpy.sum([td['highest_note'] - td['lowest_note'] for td in TRACK_TO_DATA])))
        for i, track in enumerate(self.tracks):
            offset, lowest_note, note_range = Song.__get_track_matrix_numbers__(track, self.instruments[self.track_instrument[i]])
            if ALLOW_NOTE_SCALING:
                scale, width, _ = track_lengths(track)
                scale = scale * scale / self.length / width
            for note in track:
                tone = note[1] - lowest_note
                if tone >= 0 and tone < note_range:
                    for i in range(0, note[-1]):
                        if ALLOW_NOTE_SCALING:
                            matrix[note[0]+i, tone+offset] += scale
                        else:
                            matrix[note[0]+i, tone+offset] = 1
        if ADD_SILENCE_BEFORE:
            for i, track in enumerate(self.tracks):
                offset, lowest_note, note_range = Song.__get_track_matrix_numbers__(track, self.instruments[self.track_instrument[i]])
                for note in track:
                    tone = note[1] - lowest_note
                    if tone >= 0 and tone < note_range:
                        matrix[note[0]-1, tone+offset] = 0
        return matrix

    @staticmethod
    def __get_track_matrix_numbers__(track, instrument):
        offset = 0
        lowest_note = 0
        note_range = 128
        for td in TRACK_TO_DATA:
            if td['selector'](track, instrument):
                lowest_note = td['lowest_note']
                note_range = td['highest_note'] - lowest_note
                break
            else:
                offset += td['highest_note'] - td['lowest_note']
        return offset, lowest_note, note_range


    @staticmethod
    def convert_tone_matrix(matrix: numpy.ndarray, bar_length: int=4):
        song = Song('song-{:%Y%m%d%H%M%S}'.format(datetime.datetime.now()))
        song.bar_length = bar_length
        song.length, _ = matrix.shape
        song.track_volume = [127]
        song.track_instrument = [0 for t in TRACK_TO_DATA]
        song.instruments = INSTRUMENT[:len(TRACK_TO_DATA)]
        song.tracks = [[] for t in TRACK_TO_DATA]
        offset = 0
        for i, td in enumerate(TRACK_TO_DATA):
            lowest_note = td['lowest_note']
            note_range = td['highest_note'] - lowest_note
            for note in range(note_range):
                last_on = 0
                for time in range(0, song.length):
                    state = matrix[time, offset+note]
                    if state == 0:
                        if last_on != time:
                            length = time-last_on
                            if BREAK_LONG_NOTES:
                                long_note = song.bar_length*song.beat_unit*song.ticks_per_quarter
                                while length > long_note*2:
                                    song.tracks[i].append([last_on, note+lowest_note, 60, long_note])
                                    last_on += long_note
                                    length -= long_note
                            song.tracks[i].append([last_on, note+lowest_note, 90, length])
                        last_on = time
            offset += note_range
        for track in song.tracks:
            track.sort(key=lambda n: n[0])
        return song


def track_length(track):
    return track[-1][0]+track[-1][-1]-track[0][0]

def track_lengths(track):
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
    return coverage, tone_length, end - track[0][0]


def track_concurrency(track, length=0):
    if len(track) == 0:
        return 0, 0
    coverage, tone_length, actual_lengt = track_lengths(track)
    if coverage == 0:
        coverage = 1
    if length == 0:
        length = actual_lengt
        if length == 0:
            length = 1
    return tone_length/coverage, tone_length/length

def __read_all_inputs__(song):
    return Song.read_csv_file(song)
def read_all_inputs():
    songs = Pool().map(__read_all_inputs__, [os.path.join(INPUT_FOLDER, s)
            for s in os.listdir(INPUT_FOLDER) if s.endswith(".csv")])
    print("Number of songs:", len(songs))
    return songs
