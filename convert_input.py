"""
    Script for converting input midi:s to csv:s
"""
import platform
from path import Path
import subprocess

INPUT_FOLDER = Path("input")

def check_input_converter():
    """
        Check if the executable exists
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
    if len(midis) > 0:
        for midi in midis:
            csv = midi.replace(".midi", ".csv").replace(".mid", ".csv")
            if csv in csvs and not reconvert:
                continue
            print("Converting", midi.basename(), "to csv")
            subprocess.run([exe, str(midi), str(csv)], shell=False)
        print("All songs converted")
    else:
        print("No midis found")


if __name__ == "__main__":
    convert_inputs()
