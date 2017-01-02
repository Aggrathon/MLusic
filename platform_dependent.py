
import platform
import os
import subprocess
from config import *
from song import Song


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

def check_output_converter() -> (bool, str):
    system = platform.system()
    if system == 'Windows':
        path = os.path.join(OUTPUT_FOLDER, "Csvmidi.exe")
        if os.path.isfile(path):
            return True, path
        else:
            print("Could not find the executable for turning csvs into midis")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Then put the Csvmidi.exe file in the "+OUTPUT_FOLDER+" folder")
    else:
        path = os.path.join(OUTPUT_FOLDER, "csvmidi")
        if os.path.isfile(path):
            return True, path
        else:
            print("Could not find the executable for turning csvs into midis")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Compile it and put the csvmidi file in the "+OUTPUT_FOLDER+" folder")
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

def convert_outputs(reconvert=False):
    converter_exists, exe = check_output_converter()
    if converter_exists:
        print("Converting all csvs in the", OUTPUT_FOLDER, "folder to midis")
        files = os.listdir(OUTPUT_FOLDER)
        csvs = [f[:-4] for f in files if f.endswith(".csv")]
        midis = [f[:-4] for f in files if f.endswith(".mid")]+[f[:-5] for f in files if f.endswith(".midi")]
        if len(csvs) > 0:
            for song in csvs:
                if song in midis and not reconvert:
                    continue
                print("Converting", song, "to midi")
                path = os.path.join(OUTPUT_FOLDER, song)
                subprocess.run([exe, path+".csv", path+".mid"], shell=False)
            print("All songs converted")
        else:
            print("No csvs found")

def save_and_convert_song(song: Song, play_on_finished=False) -> str:
    file_name = song.save_to_file()
    converter_exists, exe = check_output_converter()
    if converter_exists:
        subprocess.run([exe, "-v", file_name, file_name[:-4]+".mid"], shell=False)
        os.startfile(file_name[:-4]+".mid")
    return file_name
