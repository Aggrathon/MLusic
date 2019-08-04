"""
    Script for converting input midi:s to csv:s
"""
import platform
import subprocess
from pathlib import Path
from convert import read_folder, filter_folder_files

INPUT_FOLDER = Path("input")
DATA_FILE = INPUT_FOLDER / "data.csv"
META_FILE = INPUT_FOLDER / "meta.csv"

def check_input_converter() -> Path:
    """
    Check if the midi to csv executable exists

    Returns:
        Path -- The executable (None if not found)
    """
    system = platform.system()
    if system == 'Windows':
        path = INPUT_FOLDER / "Midicsv.exe"
        if path.is_file():
            return path
        else:
            print("Could not find the executable for turning midis into csvs")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Then put the Midicsv.exe file in the "+INPUT_FOLDER+" folder")
    else:
        path = INPUT_FOLDER / "midicsv"
        if path.is_file():
            return path
        else:
            print("Could not find the executable for turning midis into csvs")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Compile it and put the midicsv file in the "+INPUT_FOLDER+" folder")
    return None

def convert_inputs(reconvert=False):
    """
        Convert all input midi files
        Args:
            reconvert: If true then all files are reprocessed
    """
    exe = check_input_converter()
    if exe is None:
        return
    print("Converting all midis in the", INPUT_FOLDER, "folder to csvs")
    midis = list(filter_folder_files(INPUT_FOLDER, "mid")) + list(filter_folder_files(INPUT_FOLDER, "midi"))
    csvs = list(filter_folder_files(INPUT_FOLDER, "csv"))
    if midis:
        for midi in midis:
            csv = midi.replace(".midi", ".csv").replace(".mid", ".csv")
            if not reconvert and csv in csvs:
                continue
            print("Converting", midi.basename(), "to csv")
            subprocess.run([exe, str(midi), str(csv)], shell=False)
        print("All songs converted")
    else:
        print("No midis found")

def process_inputs(recombine=False):
    """
    Convert all the input midi csvs into the data format and combine them
    """
    print("Converting all csv-midi to pseudo-midi and combining them")
    if not DATA_FILE.exists() or not META_FILE.exists() or recombine:
        if DATA_FILE.exists():
            DATA_FILE.unlink()
        if META_FILE.exists():
            META_FILE.unlink()
        output, instruments = read_folder(INPUT_FOLDER)
        with open(DATA_FILE, "w", encoding="utf8") as file:
            for note in output:
                file.write("%d, %d, %d, %d\n"%note)
        with open(META_FILE, "w", encoding="utf8") as file:
            for ins in instruments:
                file.write("%d, %d\n"%ins)

if __name__ == "__main__":
    convert_inputs()
    process_inputs()
