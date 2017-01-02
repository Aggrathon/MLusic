
from sys import argv
from config import *
from platform_dependent import convert_inputs, convert_outputs
from input_analyser import interactive_plot
from neural_network import generate, train


if __name__ == "__main__":
    if len(argv) < 2:
        argv.append("help")

    if argv[1] == "convert":
        if len(argv[2]) > 2:
            if argv[2] == "output":
                convert_outputs()
            elif argv[2] == "input":
                convert_inputs()
            else:
                print("Unknown argument:", argv[2])
        else:
            convert_inputs()
            convert_outputs()

    elif argv[1] == "analyse":
        interactive_plot()

    elif argv[1] == "train":
        if len(argv) > 2:
            train(argv[2])
        else:
            train()

    elif argv[1] == "generate":
        if len(argv) > 2:
            generate(argv[2])
        else:
            generate()

    else:
        print(""" Usage:
    python main.py [argument]
 Arguments:
    convert output   :   convert all csv-files in the {0} folder to midis
    convert input    :   convert all midi-files in the {1} folder to csvs
    convert          :   convert both outputs and inputs
    analyse          :   Analyse songs from the input folder using graphs
    train            :   Train the latest neural network on the csvs in the {1} folder
    train [name]     :   Train a neural network with the specified name
    generate         :   Generate a song using the latest neural network
    generate [name]  :   Generate a song using the specified network
    help             :   Display this text"""
              .format(OUTPUT_FOLDER, INPUT_FOLDER))

