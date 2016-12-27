
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

