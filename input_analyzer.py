
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
    plt.hist([len([i for i in s.instruments if i != -1]) for s in songs], bins=range(1,16))
    plt.title("Number of Instruments")
    plt.show()
    plt.hist(list(np.array([[i for i in s.instruments if i != -1] for s in songs]).flat), bins="auto")
    plt.title("Instruments")
    plt.show()

