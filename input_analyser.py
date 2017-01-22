
import sys
import os
import tkinter
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from song import *
from platform_dependent import save_and_convert_song
from config import *


def histogram(data, title, bins="auto", show_mean=True):
    plt.clf()
    plt.hist(data, bins)
    if show_mean and len(data) > 1:
        mean = np.mean(data)
        std = np.std(data)
        plt.axvline(mean, color="r")
        plt.axvline(mean+std, color="y")
        plt.axvline(mean-std, color="y")
    plt.title(title)
    plt.show()


class Analyzer(object):
    songs = []

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.songs = read_all_inputs()

    def cleanup(self):
        print("Cleaning input")
        self.songs = [s for s in self.songs if s.cleanup(False)]


    def graph_song_length(self):
        histogram([s.length/s.ticks_per_quarter/4*TIME_RESOLUTION for s in self.songs], "Song Lengths")

    def graph_song_length_raw(self):
        histogram([s.length for s in self.songs], "Song Raw Lengths")

    def graph_bar_length(self):
        histogram([s.bar_length for s in self.songs], "Bar Lengths", range(2, 9))

    def graph_beat_unit(self):
        histogram([s.beat_unit for s in self.songs], "Beat Lengths", range(2, 13))

    def graph_tempo(self):
        histogram([s.tempo for s in self.songs], "Tempo")

    def graph_ticks(self):
        histogram([s.ticks_per_quarter for s in self.songs], "ticks_per_quarter")

    def graph_instrument_num(self):
        histogram([len(s.instruments) for s in self.songs], "Number of Instruments", range(1, 16))

    def graph_instruments(self):
        histogram([ins for s in self.songs for ins in s.instruments], "Instruments", range(0, 128))

    def graph_notes(self):
        histogram([n[1] for s in self.songs for t in s.tracks for n in t], "Notes", range(0, 128))

    def graph_notes_length_ticks(self):
        histogram([n[-1] for s in self.songs for t in s.tracks for n in t], "Note Lengths (ticks)")

    def graph_notes_length_beats(self):
        histogram([(s.ticks_per_quarter*4/n[-1]) for s in self.songs for t in s.tracks for n in t if n[-1] > 0], "Note Lengths (1/Length)", range(0, 40))

    def graph_note_velocity(self):
        histogram([n[2] for s in self.songs for t in s.tracks for n in t], "Note Velocity", range(0, 128, 5))

    def graph_track_num(self):
        histogram(list(np.array([len(s.tracks) for s in self.songs]).flat), "Tracks", range(1,20))

    def graph_track_len(self):
        histogram([track_length(t)*100/s.length for s in self.songs for t in s.tracks], "Track Lengths (song length %)", range(0, 100, 5))

    def graph_track_concurrency(self):
        plt.clf()
        tracks_len = []
        tracks_conc = []
        for s in self.songs:
            for t in s.tracks:
                conc, length = track_concurrency(t)
                tracks_conc.append(conc*100)
                tracks_len.append(length*100)
        plt.hist(tracks_conc, bins=range(0, 400, 10), label="sound length", alpha=0.5)
        plt.hist(tracks_len, bins=range(0, 400, 10), label="track length", alpha=0.5)
        plt.legend(loc='upper right')
        plt.title("Track Concurrency (%)")
        plt.show()

    def graph_track_width_max(self):
        tracks_max_width = []
        for s in self.songs:
            for t in s.tracks:
                i = 0
                max_concurrent = 1
                while i < len(t):
                    end = t[i][0]+t[i][-1]
                    j = i
                    while j < len(t):
                        if t[j][0] > end:
                            break
                        j += 1
                    if j-i > max_concurrent:
                        max_concurrent = j-i
                    i += 1
                tracks_max_width.append(max_concurrent)
        histogram(tracks_max_width, "Track Max Width", range(0,30), False)

    def graph_track_width_mean(self):
        tracks_mean_width = []
        for s in self.songs:
            for t in s.tracks:
                i = 0
                max_concurrent = 1
                mean_concurrent = 0
                while i < len(t):
                    end = t[i][0]+t[i][-1]
                    j = i
                    while j < len(t):
                        if t[j][0] > end:
                            break
                        j += 1
                    mean_concurrent += j-i
                    i += 1
                tracks_mean_width.append(mean_concurrent/len(t))
        histogram(tracks_mean_width, "Track Mean Width", [i*0.5 for i in range(0, 12)], False)

    def graph_track_mean_note(self):
        mean = [np.mean([n[1] for n in t]) for s in self.songs for t in s.tracks]
        histogram(mean, "Track Mean Pitch", range(20,100,1))

    def graph_matrix(self):
        matrix = self.songs[0].generate_tone_matrix()
        plt.clf()
        plt.imshow(matrix.T, cmap='Greys', aspect='auto', interpolation='none')
        plt.title("Note Matrix for "+self.songs[0].name)
        plt.show()

    def graph_matrix_tone_width(self):
        means = []
        for song in self.songs:
            matrix = song.generate_tone_matrix()
            x, y = matrix.shape
            for i in range(x):
                means.append(numpy.sum(matrix[i, :]))
        histogram(means, "Tone Width in Matrices", range(1, 30))

    def interactive_plot(self, cleanup=False):
        def select_data(value):
            self.songs.clear()
            if value == "All songs":
                print("Showing graphs for all songs")
                self.reset()
            else:
                print("Showing graphs for "+value)
                self.songs.append(Song.read_csv_file(os.path.join(INPUT_FOLDER, value+".csv")))
        def keep_first_output():
            print("Keeping only thise tracks that leads to the first output track")
            for song_index, song in enumerate(self.songs):
                nt, ti, tv = [], [], []
                for i, t in enumerate(song.tracks):
                    if TRACK_TO_DATA[0]['selector'](t, song.instruments[song.track_instrument[i]]):
                        nt.append(t)
                        ti.append(song.track_instrument[i])
                        tv.append(song.track_volume[i])
                self.songs[song_index].tracks = nt
                self.songs[song_index].track_instrument = ti
                self.songs[song_index].track_volume = tv
        def keep_second_output():
            print("Keeping only thise tracks that leads to the second output track")
            for song_index, song in enumerate(self.songs):
                nt, ti, tv = [], [], []
                for i, t in enumerate(song.tracks):
                    if not TRACK_TO_DATA[0]['selector'](t, song.instruments[song.track_instrument[i]]) \
                            and TRACK_TO_DATA[1]['selector'](t, song.instruments[song.track_instrument[i]]):
                        nt.append(t)
                        ti.append(song.track_instrument[i])
                        tv.append(song.track_volume[i])
                self.songs[song_index].tracks = nt
                self.songs[song_index].track_instrument = ti
                self.songs[song_index].track_volume = tv
        def keep_third_output():
            print("Keeping only thise tracks that leads to the third output track")
            for song_index, song in enumerate(self.songs):
                nt, ti, tv = [], [], []
                for i, t in enumerate(song.tracks):
                    if not TRACK_TO_DATA[0]['selector'](t, song.instruments[song.track_instrument[i]]) \
                            and not TRACK_TO_DATA[1]['selector'](t, song.instruments[song.track_instrument[i]]) \
                            and TRACK_TO_DATA[2]['selector'](t, song.instruments[song.track_instrument[i]]):
                        nt.append(t)
                        ti.append(song.track_instrument[i])
                        tv.append(song.track_volume[i])
                self.songs[song_index].tracks = nt
                self.songs[song_index].track_instrument = ti
                self.songs[song_index].track_volume = tv
        if cleanup:
            self.cleanup()

        window = tkinter.Tk()
        window.title("Input Analyser")
        ttk.Label(window, text="Select Data:").pack(fill='x')
        ttk.OptionMenu(window, tkinter.StringVar(), *(["All songs", "All songs"]+[s.name for s in self.songs]), command=select_data).pack(fill='x')
        ttk.Button(window, text="Cleanup songs", command=self.cleanup).pack(fill='x')
        if len(TRACK_TO_DATA) > 1:
            ttk.Button(window, text="Keep only first output tracks", command=keep_first_output).pack(fill='x')
            ttk.Button(window, text="Keep only second output tracks", command=keep_second_output).pack(fill='x')
            if len(TRACK_TO_DATA) > 2:
                ttk.Button(window, text="Keep only third output tracks", command=keep_third_output).pack(fill='x')
        else:
            ttk.Button(window, text="Keep only output tracks", command=keep_first_output).pack(fill='x')
        ttk.Label(window, text="Export: (Selected/First Song)").pack(fill='x')
        ttk.Button(window, text="Export Cleaned", command=lambda: file_export(self.songs[0])).pack(fill='x')
        ttk.Button(window, text="Export Matrixed", command=lambda: file_export(self.songs[0], True)).pack(fill='x')
        ttk.Button(window, text="Show Matrix", command=self.graph_matrix).pack(fill='x')

        ttk.Label(window, text="Select Graph:").pack(fill='x')
        graphs = [
            ("Song Length", self.graph_song_length),
            ("Song Raw Length", self.graph_song_length_raw),
            ("Tempo", self.graph_tempo),
            ("Bar Length", self.graph_bar_length),
            ("Beat Unit", self.graph_beat_unit),
            ("Instruments", self.graph_instruments),
            ("Instrument Counts", self.graph_instrument_num),
            ("Notes", self.graph_notes),
            ("Note Lengths", self.graph_notes_length_beats),
            ("Note Lengths in Ticks", self.graph_notes_length_ticks),
            ("Note Velocities", self.graph_note_velocity),
            ("Track Counts", self.graph_track_num),
            ("Track Mean Pitch", self.graph_track_mean_note),
            ("Track Lengths", self.graph_track_len),
            ("Track Max Widths", self.graph_track_width_max),
            ("Track Mean Widths", self.graph_track_width_mean),
            ("Track Concurrency", self.graph_track_concurrency),
            ("Matrix Tone Width", self.graph_matrix_tone_width)
        ]
        for label, action in graphs:
            ttk.Button(window, text=label, command=action).pack(fill='x')
        window.mainloop()


def file_export(song, matrix_conversion=False, discard_instruments=True):
    song.cleanup(discard_instruments)
    if matrix_conversion:
        matrix = song.generate_tone_matrix()
        song = Song.convert_tone_matrix(matrix)
    save_and_convert_song(song, True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "clean":
            Analyzer().interactive_plot(cleanup=True)
        else:
            Analyzer().interactive_plot()
    else:
        Analyzer().interactive_plot()





