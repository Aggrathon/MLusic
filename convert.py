"""
    Script for converting between "csv" midi and "pseudo" midi
"""
from pathlib import Path

PERCUSSION_INSTRUMENT = 128


class FileFormatException(Exception):
    """
        Exception for when a file is not in the expected format
    """


def read_csv(filename):
    """
    Read a midi csv and process the notes

    Arguments:
        filename {str} -- The input midi csv

    Returns:
        list((time, instument, note, status),)
    """
    instruments = [0]*100
    instruments[9] = PERCUSSION_INSTRUMENT
    with open(filename, "r", encoding="utf8", errors="ignore") as file:
        lines = file.readlines()
        if not lines:
            raise FileFormatException(filename)
    output = []
    header = lines[0].split(", ")
    if len(header) != 6 or header[2] != "Header":
        raise FileFormatException(filename)
    instruments = instruments * (int(header[-2]) + 1)
    ticks_per_quarter = int(header[-1])
    tempo = 500_000
    last_tempo_tick = 0
    last_tempo_time = 0
    lines = [(int(l[1]), int(l[0]), *l[2:]) for l in (line.split(", ") for line in lines)]
    lines.sort(key=lambda a: (a[0], a[1], a[2].startswith("n")))
    for line in lines:
        if line[2] == "Note_on_c":
            time = ((line[0] - last_tempo_tick) * tempo) // ticks_per_quarter + last_tempo_time
            if int(line[-1]) == 0:  # Velocity == 0
                output.append((time, instruments[line[1]*16 + int(line[3])], int(line[-2]), 0))
            else:
                output.append((time, instruments[line[1]*16 + int(line[3])], int(line[-2]), 1))
        elif line[2] == "Note_off_c":
            time = ((line[0] - last_tempo_tick) * tempo) // ticks_per_quarter + last_tempo_time
            output.append((time, instruments[line[1]*16 + int(line[3])], int(line[-2]), 0))
        elif line[2] == "Program_c":
            instruments[line[1]*16 + int(line[3])] = int(line[4])
        elif line[2] == "Tempo":
            last_tempo_time += ((line[0] - last_tempo_tick) * tempo) // ticks_per_quarter
            last_tempo_tick = line[0]
            tempo = int(line[3])
    output.sort()
    return output

def write_csv(filename, instument_csv, notes):
    instruments = dict()
    with open(instument_csv) as file:
        for line in file.readlines():
            split = line.split(", ")
            key = int(split[0])
            track = key // 15 + 2
            channel = key % 15
            if channel == 10:
                channel = 15
            ins = int(split[1])
            if ins == PERCUSSION_INSTRUMENT:
                channel = 10
            instruments[int(split[0])] = (track, channel, ins)
    tracks = len(instruments)//15+2
    with open(filename, "w", encoding="utf8") as file:
        file.write("0, 0, Header, 1, %d, 192\n"%tracks)
        file.write("1, 0, Start_track\n")
        file.write("1, 0, Time_signature, 4, 2, 24, 8\n")
        file.write("1, 0, Tempo, 500000\n")
        end = notes[-1][0] * 192 // 500_000 + 100
        file.write("1, %d, End_track\n"%end)
        for i in range(2, tracks):
            file.write("%d, 0, Start_track\n"%i)
            for j, k, l in instruments.values():
                if j == i:
                    file.write("%d, 0, Program_c, %d, %d\n"%(j, k, l))
            for note in notes:
                track, chan, ins = instruments[note[1]]
                if track != i:
                    continue
                if note[-1] == 1:
                    file.write("%d, %d, Note_on_c, %d, %d, 110\n"%(track, note[0]*192//500_000, chan, note[2]))
                else:
                    file.write("%d, %d, Note_off_c, %d, 0, 0\n"%(track, note[0]*192//500_000, chan))
            file.write("%d, %d, End_track\n"%(i, end))
        file.write("0, 0, End_of_file\n")


def read_folder(folder: Path, time_precision: int = 10):
    """
    Read all midi csvs in a folder and combine them

    Arguments:
        folder {Path} -- The folder to read from
        time_precision {int} -- Reduce the time precision from 1 microseconds when combining multiple files

    Returns:
        list((time, instument, note, status),)
        list((instrument_id, instrument),)
    """
    folder = Path(folder)
    output = []
    time = 0
    instruments = [-1] * (PERCUSSION_INSTRUMENT + 1)
    inst_ind = 0
    for file in filter_folder_files(folder, ".csv"):
        try:
            out = read_csv(file)
            if out:
                for note in out:
                    if instruments[note[1]] == -1:
                        instruments[note[1]] = inst_ind
                        inst_ind += 1
                    output.append((note[0] // time_precision + time, instruments[note[1]], note[2], note[3]))
                time = output[-1][0]
        except FileFormatException:
            pass
    instruments = sorted([(ins, i) for i, ins in enumerate(instruments) if ins != -1])
    return output, instruments


def filter_folder_files(folder: Path, extension: str):
    """Get an iterator of all files in the folder with a specific extension

    Arguments:
        folder {Path} -- The folder to iterate over
        extension {str} -- The extension of the files to return

    Returns:
        A generator of files
    """
    if not extension[0] == ".":
        extension = "."+extension
    for file in folder.iterdir():
        if file.is_file() and extension in file.suffixes:
            yield file
