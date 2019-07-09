"""
Converter of flat ROOT::TTree to hdf5. Low overhead version of convertTree.py

K. Schweiger, 2019
"""
import os
import logging

from collections import namedtuple

from preprocessing.dataset import Dataset
from utils.utils import initLogging, checkNcreateFolder


def convertTree(inputs, outFolder, name, treeName, indexVars):
    logging.info("Starting conversion")

    checkNcreateFolder(outFolder)

    dataset = Dataset(name, outFolder, treeName)

    files = []
    for _input in inputs:
        with open(_input, "r") as f:
            data = f.read()
            for line in data.split("\n"):
                if ".root" in line:
                    files.append(line)

    logging.info("Setting files")
    dataset.addFiles(files)
    logging.info("Setting output branches")
    dataset.setOutputBranches("*")

    logging.debug("Setting indexing branches: %s", indexVars)
    dataset.outputIndex = indexVars

    logging.info("Starting processing dataset")
    dataset.process(999999999999999999999)
    logging.info("Finished processing")
    
def parseArgs(args):
    import argparse
    argumentparser = argparse.ArgumentParser(
        description='Training script for autoencoders',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argumentparser.add_argument(
        "--log",
        action="store",
        type=int,
        help="Define logging level: CRITICAL - 50, ERROR - 40, WARNING - 30, INFO - 20, DEBUG - 10, \
        NOTSET - 0 \nSet to 0 to activate ROOT root messages",
        default=20
    )
    argumentparser.add_argument(
        "--inputs",
        action="store",
        type=str,
        nargs="+",
        help="List of input dataset txt file",
        required=True
    )
    argumentparser.add_argument(
        "--output",
        action="store",
        type=str,
        help="Output folder (will be created if not present of fs",
        required=True
    )
    argumentparser.add_argument(
        "--name",
        action="store",
        type=str,
        help="This will be the name of the output file (.h5 will be appended by script)",
        required=True,
    )
    argumentparser.add_argument(
        "--treeName",
        action="store",
        type=str,
        help="Name of the TTree",
        default="tree"
    )
    argumentparser.add_argument(
        "--indexVariables",
        action="store",
        nargs="+",
        type=str,
        help="Variables that are used for indexing",
        default=["evt", "run", "lumi"]
    )
    return argumentparser.parse_args(args)


if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])
    initLogging(args.log)

    convertTree(args.inputs, args,output, args.name, args.treeName, args.indexVariables)
