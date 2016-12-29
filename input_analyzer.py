
import numpy as np
import matplotlib.pyplot as plt
from song import read_all_inputs, Song, __track_length__, __track_concurrency__
import sys
import subprocess
import os


SONG_NAME = "avemar~1" # avemar~1 to_town rudolph carolbel


def plot_input_graphs(cleanup=False, one_song=False):
    songs = []
    if one_song:
        songs = [Song("input/"+SONG_NAME+".csv")]
    else:
        songs = read_all_inputs()
        if cleanup:
            for s in songs:
                s.cleanup(False)
        
    plt.hist([s.length/s.ticks_per_quarter/4*s.beat_unit/s.bar_length for s in songs], bins=range(0,210,10))
    plt.title("Song Lengths")
    plt.show()
    plt.hist([s.bar_length for s in songs], bins=range(2,9))
    plt.title("Bar Lengths")
    plt.show()
    plt.hist([s.beat_unit for s in songs], bins=range(2,13))
    plt.title("Beat Lengths")
    plt.show()

    plt.hist([s.tempo for s in songs], bins="auto")
    plt.title("Tempo")
    plt.show()
    plt.hist([s.ticks_per_quarter for s in songs], bins="auto")
    plt.title("ticks_per_quarter")
    plt.show()

    plt.hist([len(s.instruments) for s in songs], bins=range(1,16))
    plt.title("Number of Instruments")
    plt.show()
    inst = []
    for s in songs:
        for i in s.instruments:
            inst.append(i)
    plt.hist(inst, bins=range(-10,128))
    plt.title("Instruments")
    plt.show()

    notes = []
    for s in songs:
        for t in s.tracks:
            for n in t:
                notes.append(n[1])
    plt.hist(notes, bins=range(35,86))
    plt.title("Notes")
    plt.show()

    """
    notes_len = []
    for s in songs:
        for t in s.tracks:
            for n in t:
                if n[-1] > 0:
                    notes_len.append(n[-1])
    if cleanup:
        plt.hist(notes_len, bins=range(0,80))
    else:
        plt.hist(notes_len, bins=range(0,800,10))
    plt.title("Note Lengths (ticks)")
    plt.show()
    """
    notes_len = []
    for s in songs:
        for t in s.tracks:
            for n in t:
                if n[-1] > 0:
                    notes_len.append(float(s.ticks_per_quarter)/float(n[-1])*4)
    plt.hist(notes_len, bins=range(0,40))
    plt.title("Note Lengths (1/Length)")
    plt.show()
    
    notes_vel = []
    for s in songs:
        for t in s.tracks:
            for n in t:
                notes_vel.append(n[1])
    plt.hist(notes_vel, bins=range(0,130,5))
    plt.title("Note Velocity")
    plt.show()
    
    if cleanup:
        plt.hist(list(np.array([len(s.tracks) for s in songs]).flat), bins=range(1,16))
    else:
        plt.hist(list(np.array([len(s.tracks) for s in songs]).flat), bins="auto")
    plt.title("Tracks")
    plt.show()

    tracks_len = []
    for s in songs:
        for t in s.tracks:
            tracks_len.append(__track_length__(t)*100/s.length)
    plt.hist(tracks_len, bins=range(0, 100, 5))
    plt.title("Track Lengths (song length %)")
    plt.show()

    tracks_len = []
    tracks_conc = []
    for s in songs:
        for t in s.tracks:
            conc, length = __track_concurrency__(t)
            tracks_conc.append(conc*100)
            tracks_len.append(length*100)
    plt.hist(tracks_conc, bins=range(0, 400, 10))
    plt.title("Track Concurrency (sound length %)")
    plt.show()
    plt.hist(tracks_len, bins=range(0, 400, 10))
    plt.title("Track Concurrency (track length %)")
    plt.show()

    tracks_max_width = []
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
                if j-i > max_concurrent:
                    max_concurrent = j-i
                mean_concurrent += j-i
                i += 1
            tracks_mean_width.append(mean_concurrent/len(t))
            tracks_max_width.append(max_concurrent)
    plt.hist(tracks_max_width, bins=range(0,30))
    plt.title("Track Max Width")
    plt.show()
    plt.hist(tracks_mean_width, bins=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    plt.title("Track Mean Width")
    plt.show()


def test_file_generation():
    s = Song("input/"+SONG_NAME+".csv")
    s.cleanup()
    s.save_to_file()
    subprocess.run([".\output\Csvmidi.exe", "-v", ".\output\\"+SONG_NAME+".csv", ".\output\\"+SONG_NAME+".mid"], shell=True)
    os.startfile(".\output\\"+SONG_NAME+".mid")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_file_generation()
        elif sys.argv[1] == "clean":
            plot_input_graphs(cleanup=True)
        elif sys.argv[1] == "one":
            plot_input_graphs(one_song=True)
    else:
        plot_input_graphs()





