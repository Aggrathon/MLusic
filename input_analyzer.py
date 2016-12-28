
import numpy as np
import matplotlib.pyplot as plt
from song import read_all_inputs




if __name__ == "__main__":
    songs = read_all_inputs()
    
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
    plt.hist(list(np.array([s.instruments for s in songs]).flat), bins="auto")
    plt.title("Instruments")
    plt.show()
    plt.hist(list(np.array([len(s.tracks) for s in songs]).flat), bins="auto")
    plt.title("Tracks")
    plt.show()

    notes = []
    for s in songs:
        for t in s.tracks:
            for n in t:
                notes.append(n[1])
    plt.hist(notes, bins=range(35,86))
    plt.title("Notes")
    plt.show()

    notes_len = []
    for s in songs:
        for t in s.tracks:
            for n in t:
                if n[-1] > 0:
                    notes_len.append(n[-1])
    plt.hist(notes_len, bins=range(0,800,10))
    plt.title("Note Lengths (ticks)")
    plt.show()

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

    tracks_len = []
    for s in songs:
        for t in s.tracks:
            t_len = 0
            for n in t:
                t_len += n[-1]
            tracks_len.append(t_len/(t[-1][0]+t[-1][-1])*100)
    plt.hist(tracks_len, bins=range(0, 400, 10))
    plt.title("Track Lengths (%)")
    plt.show()

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
    plt.hist(tracks_max_width, bins=range(0,30))
    plt.title("Track Max Width")
    plt.show()





