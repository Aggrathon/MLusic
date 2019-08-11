"""
    Script for turning output csv:s into midi:s
"""
import platform
import subprocess
import os
from pathlib import Path
from convert import filter_folder_files

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
        if path.is_file():
            return path
        else:
            print("Could not find the executable for turning csvs into midis")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Then put the Csvmidi.exe file in the "+OUTPUT_FOLDER+" folder")
    else:
        path = OUTPUT_FOLDER / "csvmidi"
        if path.is_file():
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
    csvs = list(filter_folder_files(OUTPUT_FOLDER, "csv"))
    midis = list(filter_folder_files(OUTPUT_FOLDER, "mid")) + list(filter_folder_files(OUTPUT_FOLDER, "midi"))
    if csvs:
        for csv in csvs:
            midi = str(csv).replace(".csv", ".midi")
            if not reconvert and midi in midis:
                continue
            print("Converting", csv, "to midi")
            subprocess.run([str(exe), str(csv), str(midi)], shell=False)
        print("All songs converted")
    else:
        print("No csvs found")

if __name__ == "__main__":
    convert_outputs()
