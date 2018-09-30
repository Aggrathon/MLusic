"""
    Script for turning output csv:s into midi:s
"""
import platform
import subprocess
import os
from path import Path
from song import Song

OUTPUT_FOLDER = Path("output")


def check_output_converter():
    """
    Check if the csv to midi executable exists

    Returns:
        [Path] -- The executable (None if not found)
    """
    system = platform.system()
    if system == 'Windows':
        path = OUTPUT_FOLDER / "Csvmidi.exe"
        if path.isfile():
            return path
        else:
            print("Could not find the executable for turning csvs into midis")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Then put the Csvmidi.exe file in the "+OUTPUT_FOLDER+" folder")
    else:
        path = OUTPUT_FOLDER / "csvmidi"
        if path.isfile():
            return path
        else:
            print("Could not find the executable for turning csvs into midis")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Compile it and put the csvmidi file in the "+OUTPUT_FOLDER+" folder")
    return None

def convert_outputs(reconvert=False):
    """
    Convert all csvs in the ouput directory to midis

    Keyword Arguments:
        reconvert {bool} -- Should existing midis be recreated (default: {False})
    """
    exe = check_output_converter()
    if exe is None:
        return
    print("Converting all csvs in the", OUTPUT_FOLDER, "folder to midis")
    csvs = OUTPUT_FOLDER.files("*.csv")
    midis = OUTPUT_FOLDER.files("*.mid*")
    if csvs:
        for csv in csvs:
            midi = csv.replace(".csv", ".midi")
            if midi in midis and not reconvert:
                continue
            print("Converting", csv, "to midi")
            subprocess.run([exe, str(csv), str(midi)], shell=False)
        print("All songs converted")
    else:
        print("No csvs found")

def save_and_convert_song(song: Song, filename: str, play_on_finished: bool = True):
    """
    Convert a Song to midi

    Arguments:
        song {Song} -- The song to convert
        filename {str} -- The filename for the saved midi csv (also used for the actual midi file)

    Keyword Arguments:
        play_on_finished {bool} -- Play the resulting midi file with the default program (default: {True}, only on Windows)
    """
    song.save_midi(filename)
    exe = check_output_converter()
    output = filename[:-4]+".midi"
    if exe is not None:
        subprocess.run([exe, "-v", filename, output], shell=False)
        if play_on_finished:
            os.startfile(output.replace('\\', '\\\\'))

if __name__ == "__main__":
    convert_outputs()
