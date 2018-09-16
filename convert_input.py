"""
    Script for converting input midi:s to csv:s
"""
import platform
import subprocess
from path import Path
from song import Song

INPUT_FOLDER = Path("input")
DATA_FILE = INPUT_FOLDER / "data.csv"

def check_input_converter() -> Path:
    """
    Check if the midi to csv executable exists

    Returns:
        Path -- The executable (None if not found)
    """
    system = platform.system()
    if system == 'Windows':
        path = INPUT_FOLDER / "Midicsv.exe"
        if path.isfile():
            return path
        else:
            print("Could not find the executable for turning midis into csvs")
            print("Please download it from here: http://www.fourmilab.ch/webtools/midicsv/")
            print("Then put the Midicsv.exe file in the "+INPUT_FOLDER+" folder")
    else:
        path = INPUT_FOLDER / "midicsv"
        if path.isfile():
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
    midis = INPUT_FOLDER.files("*.mid*")
    csvs = INPUT_FOLDER.files("*.csv")
    if midis:
        for midi in midis:
            csv = midi.replace(".midi", ".csv").replace(".mid", ".csv")
            if csv in csvs and not reconvert:
                continue
            print("Converting", midi.basename(), "to csv")
            subprocess.run([exe, str(midi), str(csv)], shell=False)
        print("All songs converted")
    else:
        print("No midis found")

def process_inputs(overwrite=False):
    """
    Convert all the input midi csvs into the data format and combine them

    Keyword Arguments:
        overwrite {bool} -- If the data file exists should it be recreated (default: {False})

    Returns:
        Song -- the combined midis
    """
    if DATA_FILE.exists() and not overwrite:
        print("Data file already exists")
        return
    print("Processing the csvs")
    return Song.read_folder(INPUT_FOLDER).save_data(DATA_FILE)

if __name__ == "__main__":
    convert_inputs()
    process_inputs()
