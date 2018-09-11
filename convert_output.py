"""
    Script for turning output csv:s into midi:s
"""
import platform
import subprocess
import os
from path import Path
from song import Song

OUTPUT_FOLDER = Path("output")


def check_output_converter() -> (bool, str):
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
            return  path
        else:
            print("Could not find the executable for turning csvs into midis")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Compile it and put the csvmidi file in the "+OUTPUT_FOLDER+" folder")
    return None

def convert_outputs(reconvert=False):
    exe = check_output_converter()
    if exe is None:
        return
    print("Converting all csvs in the", OUTPUT_FOLDER, "folder to midis")
    csvs = OUTPUT_FOLDER.files("*.csv")
    midis = OUTPUT_FOLDER.files("*.mid*")
    if len(csvs) > 0:
        for csv in csvs:
            midi = csv.replace(".csv", ".midi")
            if midi in midis and not reconvert:
                continue
            print("Converting", csv, "to midi")
            subprocess.run([exe, str(csv), str(midi)], shell=False)
        print("All songs converted")
    else:
        print("No csvs found")

def save_and_convert_song(song: Song, play_on_finished=False):
    file_name = song.save_to_file()
    exe = check_output_converter()
    if exe is not None:
        subprocess.run([exe, "-v", file_name, file_name[:-4]+".midi"], shell=False)
        os.startfile(file_name[:-4]+".midi")

if __name__ == "__main__":
    convert_outputs()
