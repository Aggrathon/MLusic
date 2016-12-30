
import sys
import subprocess
import os
import tkinter
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from song import *


SONG_NAME = "sarajevo" # avemar~1 to_town rudolph carolbel sarajevo

def histogram(data, title, bins="auto"):
    plt.clf()
    plt.hist(data, bins)
    if len(data) > 1:
        mean = np.mean(data)
        std = np.std(data)
        plt.axvline(mean, color="r")
        plt.axvline(mean+std, color="y")
        plt.axvline(mean-std, color="y")
    plt.title(title)
    plt.show()

def graph_song_length(songs):
    histogram([s.length/s.ticks_per_quarter for s in songs], "Song Lengths")

def graph_song_length_raw(songs):
    histogram([s.length for s in songs], "Song Raw Lengths")

def graph_bar_length(songs):
    histogram([s.bar_length for s in songs], "Bar Lengths", range(2, 9))

def graph_beat_unit(songs):
    histogram([s.beat_unit for s in songs], "Beat Lengths", range(2, 13))

def graph_tempo(songs):
    histogram([s.tempo for s in songs], "Tempo")

def graph_ticks(songs):
    histogram([s.ticks_per_quarter for s in songs], "ticks_per_quarter")

def graph_instrument_num(songs):
    histogram([len(s.instruments) for s in songs], "Number of Instruments", range(1, 16))

def graph_instruments(songs):
    histogram([ins for s in songs for ins in s.instruments], "Instruments", range(0, 128))

def graph_notes(songs):
    histogram([n[1] for s in songs for t in s.tracks for n in t], "Notes", range(0, 128))

def graph_notes_length_ticks(songs):
    histogram([n[-1] for s in songs for t in s.tracks for n in t], "Note Lengths (ticks)")

def graph_notes_length_beats(songs):
    histogram([(s.ticks_per_quarter*4/n[-1]) for s in songs for t in s.tracks for n in t if n[-1] > 0], "Note Lengths (1/Length)", range(0, 40))

def graph_note_velocity(songs):
    histogram([n[2] for s in songs for t in s.tracks for n in t], "Note Velocity", range(0, 128, 5))

def graph_track_num(songs):
    histogram(list(np.array([len(s.tracks) for s in songs]).flat), "Tracks", range(1,20))

def graph_track_len(songs):
    histogram([__track_length__(t)*100/s.length for s in songs for t in s.tracks], "Track Lengths (song length %)", range(0, 100, 5))

def graph_track_concurrency(songs):
    plt.clf()
    tracks_len = []
    tracks_conc = []
    for s in songs:
        for t in s.tracks:
            conc, length = __track_concurrency__(t)
            tracks_conc.append(conc*100)
            tracks_len.append(length*100)
    plt.hist(tracks_conc, bins=range(0, 400, 10), label="sound length", alpha=0.5)
    plt.hist(tracks_len, bins=range(0, 400, 10), label="track length", alpha=0.5)
    plt.legend(loc='upper right')
    plt.title("Track Concurrency (%)")
    plt.show()

def graph_track_width_max(songs):
    tracks_max_width = []
    for s in songs:
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
    histogram(tracks_max_width, "Track Max Width", range(0,30))

def graph_track_width_mean(songs):
    tracks_mean_width = []
    for s in songs:
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
    histogram(tracks_mean_width, "Track Mean Width", [i*0.5 for i in range(0, 12)])

def graph_matrix(song):
    matrix = song.generate_tone_matrix()
    plt.clf()
    plt.imshow(matrix.T, cmap='Greys', aspect='auto', interpolation='none')
    plt.title("Note Matrix for "+song.name)
    plt.show()


def interactive_plot(cleanup=False):
    songs = read_all_inputs()
    def select_data(value):
        songs.clear()
        if value == "All songs":
            for s in read_all_inputs():
                songs.append(s)
        else:
            songs.append(Song.read_csv_file(INPUT_FOLDER+value+".csv"))
    def clean_data():
        for s in songs:
            s.cleanup(False)
    if cleanup:
        clean_data()

    window = tkinter.Tk()
    window.title("Input Analyser")
    ttk.Label(window, text="Select Data:").pack(fill='x')
    ttk.OptionMenu(window, tkinter.StringVar(), *(["All songs"]+[s.name for s in songs]), command=select_data).pack(fill='x')
    ttk.Button(window, text="Cleanup songs", command=clean_data).pack(fill='x')
    ttk.Label(window, text="Export: (Selected/First Song)").pack(fill='x')
    ttk.Button(window, text="Export Cleaned", command=lambda: file_export(False, songs[0])).pack(fill='x')
    ttk.Button(window, text="Export Matrixed", command=lambda: file_export(True, songs[0])).pack(fill='x')
    ttk.Button(window, text="Show Matrix", command=lambda: graph_matrix(songs[0])).pack(fill='x')

    ttk.Label(window, text="Select Graph:").pack(fill='x')
    graphs = [
        ("Song Length", graph_song_length),
        ("Song Raw Length", graph_song_length_raw),
        ("Tempo", graph_tempo),
        ("Bar Length", graph_bar_length),
        ("Beat Unit", graph_beat_unit),
        ("Instruments", graph_instruments),
        ("Instrument Counts", graph_instrument_num),
        ("Notes", graph_notes),
        ("Note Lengths", graph_notes_length_beats),
        ("Note Lengths in Ticks", graph_notes_length_ticks),
        ("Note Velocities", graph_note_velocity),
        ("Track Counts", graph_track_num),
        ("Track Lengths", graph_track_len),
        ("Track Max Widths", graph_track_width_max),
        ("Track Mean Widths", graph_track_width_mean),
        ("Track Concurrency", graph_track_concurrency)
    ]
    for label, action in graphs:
        ttk.Button(window, text=label, command=lambda a=action: a(songs)).pack(fill='x')
    window.mainloop()


def file_export(matrix_conversion=False,song=None):
    if song is None:
        song = Song.read_csv_file(INPUT_FOLDER+SONG_NAME+".csv")
    song.cleanup()
    if matrix_conversion:
        matrix = song.generate_tone_matrix()
        song = Song.convert_tone_matrix(matrix)
    song.save_to_file()
    folder = ".\\"+OUTPUT_FOLDER[:-1]+"\\"
    subprocess.run([folder+"Csvmidi.exe", "-v", "{}{}.csv".format(folder, song.name), "{}{}.mid".format(folder, song.name)], shell=False)
    os.startfile("{}{}.mid".format(folder, song.name))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            file_export()
        if sys.argv[1] == "test2":
            file_export(True)
        elif sys.argv[1] == "clean":
            interactive_plot(cleanup=True)
    else:
        interactive_plot()





