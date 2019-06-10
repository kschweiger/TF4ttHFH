"""
Converter of flat ROOT::TTree to hdf5 file that can be used by Keras

K. Schweiger, 2019
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
import uproot as root

from preprocessing.dataset import Dataset
from preprocessing.utils import initLogging
#sys.path.insert(0, os.path.abspath('../utils/'))
from utils.ConfigReader import ConfigReaderBase



class Config(ConfigReaderBase):
    def __init__(self, **kwargs):
        # Parse kwargs
        pathToConfig = kwargs["path"]

        self.addVariables = kwargs["addVars"]
        self.indexVariables = kwargs["indexVars"]
        self.outputFolder = kwargs["output"]
        
        super(Config, self).__init__(pathToConfig)
        
        self.sampleName = self.readConfig.get("Sample", "name")
        cwd = os.getcwd()
        self.fileList = cwd+"/"+self.readConfig.get("Sample", "path")
        self.files = []
        with open(self.fileList, "r") as f:
            data = f.read()
            for line in data.split("\n"):
                if ".root" in line:
                    self.files.append(line)

        
        self.sampleSelection = self.readConfig.get("Sample", "selection")

        self.outputPrefix = self.readConfig.get("General", "outputPrefix")
        self.maxEvents = self.readConfig.getint("General", "maxEvents")

        self.outputVariables = self.getList(self.readConfig.get("General", "outputVariables"))
        self.outputVariables += self.addVariables
        self.outputVariables += self.indexVariables

        self.outputVariables = list(set(self.outputVariables)) #remove duplicates
        
        logging.debug("------ Config ------")
        logging.debug("output folder: %s", self.outputFolder)
        logging.debug("output prefix: %s", self.outputPrefix)
        logging.debug("output variables: %s", self.outputVariables)
        logging.debug("maxEvents: %s", self.maxEvents)
        logging.debug("Sample:")
        logging.debug("  name: %s", self.sampleName)
        logging.debug("  filelist: %s",  self.fileList)
        logging.debug("  num Files: %s", len(self.files))
        logging.debug("  selection: %s", self.sampleSelection)

    def __repr__(self):
        confRepr = "------ Config ------\n"
        confRepr += "output folder: %s\n"%(self.outputFolder)
        confRepr += "output prefix: %s\n"%(self.outputPrefix)
        confRepr += "output variables: %s\n"%(self.outputVariables)
        confRepr += "maxEvents: %s\n"%(self.maxEvents)
        confRepr += "Sample:\n"
        confRepr += "  name: %s\n"%(self.sampleName)
        confRepr += "  filelist: %s\n"%( self.fileList)
        confRepr += "  num Files: %s\n"%(len(self.files))
        for i, f in enumerate(self.files):
            confRepr += "    File %s: %s\n"%(i,f)
        confRepr += "  selection: %s\n"%(self.sampleSelection)
        
        return confRepr


def convertTree(config, treeName):
    logging.info("Starting conversion")
    
    dataset = Dataset(config.outputPrefix+"_"+config.sampleName, config.outputFolder, treeName)

    logging.info("Setting files")
    dataset.addFiles(config.files)

    logging.info("Setting output branches")
    dataset.setOutputBranches(config.ouputVariables)

    logging.debug("Setting indexing branches: %s", config.indexVariables)
    dataset.outputIndex = config.indexVariables
    
    logging.info("Starting processing dataset")
    dataset.process(config.maxEvents)
    
    logging.info("Finished processing")
    
if __name__ == "__main__":
    import argparse
    ##############################################################################################################
    ##############################################################################################################
    # Argument parser definitions:
    argumentparser = argparse.ArgumentParser(
        description='HDf5 converter for flat ROOT::TTree',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argumentparser.add_argument(
        "--log",
        action = "store",
        help = "Define logging level: CRITICAL - 50, ERROR - 40, WARNING - 30, INFO - 20, DEBUG - 10, NOTSET - 0 \nSet to 0 to activate ROOT root messages",
        type=int,
        default=20
    )
    argumentparser.add_argument(
        "--output",
        action = "store",
        help = "path to folder where the output will be saved",
        type = str,
        required = True
    )
    argumentparser.add_argument(
        "--config",
        action = "store",
        help = "configuration file",
        type = str,
        required = True
    )
    argumentparser.add_argument(
        "--treeName",
        action = "store",
        help = "Name of the TTree",
        type = str,
        default="tree"
    )
    argumentparser.add_argument(
        "--additionalVariables",
        action = "store",
        nargs = "+",
        help = "Additonal Variables that can be set on runtime",
        type = str,
        default=["njets", "nBDeepCSVM", "nBDeepCSVL"]
    )
    argumentparser.add_argument(
        "--indexVariables",
        action = "store",
        nargs = "+",
        help = "Variables that are used for indexing",
        type = str,
        default=["evt", "run", "lumi"]
    )
    
    args = argumentparser.parse_args()
    #
    ##############################################################################################################
    ##############################################################################################################
    initLogging(args.log)
    
    thisConfig = Config(path = args.config, addVars = args.additionalVariables,
                        indexVars = args.indexVariables, output = args.output)
    
    convertTree(thisConfig, args.treeName)

    logging.info("Exiting....")
