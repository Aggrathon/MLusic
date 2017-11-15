
import numpy as np
from matplotlib import pyplot as plt
from song import read_all_inputs

def instruments():
    songs = read_all_inputs()
    for s in songs:
        s.import_cleanup()
    plt.hist([i for s in songs for i in s.instruments], np.arange(129))
    plt.show()


if __name__ == "__main__":
    instruments()
