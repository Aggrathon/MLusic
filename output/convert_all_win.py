import os
import subprocess

files = os.listdir()
csvs = [f[:-4] for f in files if f.endswith(".csv")]
midis = [f[:-4] for f in files if f.endswith(".mid")]+[f[:-5] for f in files if f.endswith(".midi")]

if "Csvmidi.exe" not in files:
    print("You must download the Csvmidi.exe program first")
    exit()

if len(csvs) > 0:
    for song in csvs:
        if song in midis:
            continue
        print("Converting",song)
        subprocess.run(["Csvmidi.exe", song+".csv", song+".mid"], shell=False)
    print("All songs converted")
else:
    print("No csvs found")