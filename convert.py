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
    instruments = instruments * (int(header[-2])+1)
    lines = [(int(l[1]), int(l[0]), *l[2:]) for l in (line.split(", ") for line in lines)]
    lines.sort(key=lambda a: (a[0], a[1], a[2].startswith("n")))
    for line in lines:
        if line[2] == "Note_on_c":
            if int(line[-1]) == 0:  # Velocity == 0
                output.append((line[0], instruments[line[1]*16 + int(line[3])], int(line[-2]), 0))
            else:
                output.append((line[0], instruments[line[1]*16 + int(line[3])], int(line[-2]), 1))
        elif line[2] == "Note_off_c":
            output.append((line[0], instruments[line[1]*16 + int(line[3])], int(line[-2]), 0))
        elif line[2] == "Program_c":
            instruments[line[1]*16 + int(line[3])] = int(line[4])
    output.sort()
    return output


def read_folder(folder: Path):
    """
    Read all midi csvs in a folder and combine them

    Arguments:
        folder {str} -- The folder to read from

    Returns:
        list((time, instument, note, status),)
    """
    folder = Path(folder)
    output = []
    time = 0
    for file in filter_folder_files(folder, ".csv"):
        try:
            out = read_csv(file)
            if out:
                for note in out:
                    output.append((note[0] + time, *note[1:]))
                time = output[-1][0]
        except FileFormatException:
            pass
    return output


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
