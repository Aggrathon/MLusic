"""
    Script for converting all input midis to csvs
"""

import platform
import os
import subprocess
from config import INPUT_FOLDER


def check_input_converter() -> (bool, str):
    system = platform.system()
    if system == 'Windows':
        path = os.path.join(INPUT_FOLDER, "Midicsv.exe")
        if os.path.isfile(path):
            return True, path
        else:
            print("Could not find the executable for turning midis into csvs")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Then put the Midicsv.exe file in the "+INPUT_FOLDER+" folder")
    else:
        path = os.path.join(INPUT_FOLDER, "midicsv")
        if os.path.isfile(path):
            return True, path
        else:
            print("Could not find the executable for turning midis into csvs")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Compile it and put the midicsv file in the "+INPUT_FOLDER+" folder")
    return False, ""

def convert_inputs(reconvert=False):
    converter_exists, exe = check_input_converter()
    if converter_exists:
        print("Converting all midis in the", INPUT_FOLDER, "folder to csvs")
        files = os.listdir(INPUT_FOLDER)
        csvs = [f[:-4] for f in files if f.endswith(".csv")]
        midis = [f[:-4] for f in files if f.endswith(".mid")]+[f[:-5] for f in files if f.endswith(".midi")]
        if len(midis) > 0:
            for song in midis:
                if song in csvs and not reconvert:
                    continue
                print("Converting", song, "to csv")
                path = os.path.join(INPUT_FOLDER, song)
                subprocess.run([exe, path+".mid", path+".csv"], shell=False)
            print("All songs converted")
        else:
            print("No midis found")

if __name__ == "__main__":
    convert_inputs()