"""
Converter of flat ROOT::TTree to hdf5 file that can be used by Keras

K. Schweiger, 2019
"""
import os
import logging

from collections import namedtuple

from preprocessing.dataset import Dataset
from utils.utils import initLogging
#sys.path.insert(0, os.path.abspath('../utils/'))
from utils.ConfigReader import ConfigReaderBase



class Config(ConfigReaderBase):
    """
    Containter for setting from the config file

    Args:
      path (str) : path to the config file --> Relative to directory of convertTree.py
      addVars (list) : Additional vars that are used for the output, that are not set in config
      indexVars (list) : Variables that are used for indexing the dataframe
      output (str) : Abs. ouput folder path
    """
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

        self.allCategories = self.getList(self.readConfig.get("General", "categories"))

        self.categories = {}
        catTuple = namedtuple("CatTuple", ["selection", "name"])
        for cat in self.allCategories:
            thisSelection = self.readConfig.get(cat, "selection")
            thisName = self.readConfig.get(cat, "name")
            self.categories[cat] = catTuple(selection=thisSelection, name=thisName)

        logging.debug("------ Config ------")
        logging.debug("output folder: %s", self.outputFolder)
        logging.debug("output prefix: %s", self.outputPrefix)
        logging.debug("output variables: %s", self.outputVariables)
        logging.debug("maxEvents: %s", self.maxEvents)
        logging.debug("Sample:")
        logging.debug("  name: %s", self.sampleName)
        logging.debug("  filelist: %s", self.fileList)
        logging.debug("  num Files: %s", len(self.files))
        logging.debug("  selection: %s", self.sampleSelection)
        logging.debug("Categories: %s", self.allCategories)
        for cat in self.allCategories:
            logging.debug("  Category: %s", cat)
            logging.debug("    Name: %s", self.categories[cat].name)
            logging.debug("    Selection: %s", self.categories[cat].selection)

    def __repr__(self):
        """ Representation for printing (mainly debugging) """
        confRepr = "------ Config ------\n"
        confRepr += "output folder: %s\n"%(self.outputFolder)
        confRepr += "output prefix: %s\n"%(self.outputPrefix)
        confRepr += "output variables: %s\n"%(self.outputVariables)
        confRepr += "maxEvents: %s\n"%(self.maxEvents)
        confRepr += "Sample:\n"
        confRepr += "  name: %s\n"%(self.sampleName)
        confRepr += "  filelist: %s\n"%(self.fileList)
        confRepr += "  num Files: %s\n"%(len(self.files))
        for i, f in enumerate(self.files):
            confRepr += "    File %s: %s\n"%(i, f)
        confRepr += "  selection: %s\n"%(self.sampleSelection)
        confRepr += "Categories: %s"%(self.allCategories)
        for cat in self.allCategories:
            confRepr += "  Category: %s"%(cat)
            confRepr += "    Name: %s"%(self.categories[cat].name)
            confRepr += "    Selection: %s"%(self.categories[cat].selection)

        return confRepr

def convertTree(config, treeName, category):
    """ Wrapper for the functionality of preprocessing.dataset  """
    logging.info("Starting conversion")

    datasetName = config.outputPrefix+"_"+config.sampleName+"_"+config.categories[category].name
    dataset = Dataset(datasetName, config.outputFolder, treeName)

    logging.info("Setting sample selection: %s", config.sampleSelection)
    dataset.sampleSelection = config.sampleSelection
    logging.info("Setting category selection: %s", config.categories[category].selection)
    dataset.selection = config.categories[category].selection

    logging.info("Setting files")
    dataset.addFiles(config.files)

    logging.info("Setting output branches")
    dataset.setOutputBranches(config.outputVariables)

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
        action="store",
        type=int,
        help="Define logging level: CRITICAL - 50, ERROR - 40, WARNING - 30, INFO - 20, DEBUG - 10, \
        NOTSET - 0 \nSet to 0 to activate ROOT root messages",
        default=20
    )
    argumentparser.add_argument(
        "--output",
        action="store",
        type=str,
        help="path to folder where the output will be saved",
        required=True
    )
    argumentparser.add_argument(
        "--config",
        action="store",
        type=str,
        help="configuration file",
        required=True
    )
    argumentparser.add_argument(
        "--treeName",
        action="store",
        type=str,
        help="Name of the TTree",
        default="tree"
    )
    argumentparser.add_argument(
        "--additionalVariables",
        action="store",
        nargs="+",
        type=str,
        help="Additonal Variables that can be set on runtime",
        default=["njets", "nBDeepCSVM", "nBDeepCSVL"]
    )
    argumentparser.add_argument(
        "--indexVariables",
        action="store",
        nargs="+",
        type=str,
        help="Variables that are used for indexing",
        default=["evt", "run", "lumi"]
    )
    argumentparser.add_argument(
        "--categories",
        action="store",
        nargs="+",
        type=str,
        help="Variables that are used for indexing",
        default=None
    )

    args = argumentparser.parse_args()
    #
    ##############################################################################################################
    ##############################################################################################################
    initLogging(args.log)

    thisConfig = Config(path=args.config, addVars=args.additionalVariables,
                        indexVars=args.indexVariables, output=args.output)

    runCats = []
    if args.categories is None:
        runCats = thisConfig.allCategories
    else:
        runCats = args.categories

    for thisCat in runCats: #Not ideal for performance...
        convertTree(thisConfig, args.treeName, thisCat)

    logging.info("Exiting....")
