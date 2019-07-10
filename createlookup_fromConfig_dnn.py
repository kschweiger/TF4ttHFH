import sys
import os
import logging
import time

from utils.utils import initLogging, checkNcreateFolder
from utils.ConfigReader import ConfigReaderBase
import createlookup_dnn

class Config(ConfigReaderBase):
    """ Container for running lookup table generation from config """
    def __init__(self, pathToConfig):
        super(Config, self).__init__(pathToConfig)

        self.output = self.readConfig.get("General", "output")
        self.runCategories = self.getList(self.readConfig.get("General", "categories"))
        self.runDatasets = []

        logging.debug("Config has sections: %s", self.readConfig.sections())

        self.datasets = {}
        
        for section in self.readConfig.sections():
            if section == "General" or section in self.runCategories:
                continue
            self.runDatasets.append(section)
            self.datasets[section] = self.readConfig.get(section, "input")
            logging.debug("Added dataset %s with value %s", section, self.datasets[section])
            
        self.catSettings = {}
        for cat in self.runCategories:
            self.catSettings[cat] = { "selection" : self.readConfig.get(cat, "selection"),
                                      "model" : self.readConfig.get(cat, "model")}

def main(config):
    checkNcreateFolder(config.output)
    for dataset in config.runDatasets:
        logging.info("Processing dataset %s", dataset)
        finalDF = processDataset(config, dataset)
        createlookup_dnn.writeLookupTable(finalDF, config.output, dataset)
        
def processDataset(config, dataset):
    catPredictions = []
    
    _, inputDataFull = createlookup_dnn.getSampleData(createlookup_dnn.getModelDefinitions(config.catSettings[config.runCategories[0]]["model"]),
                                                      config.datasets[dataset])
    inputDataFull = inputDataFull.getTestData(asMatrix=False)

    for cat in config.runCategories:
        catPrediction, _, _ = createlookup_dnn.processData(config.catSettings[cat]["model"],
                                                           config.datasets[dataset],
                                                           config.catSettings[cat]["selection"])
        catPredictions.append(catPrediction)

    return mergePredictions(inputDataFull, catPredictions)
    

def mergePredictions(inputDataFull, allPredictions):
    finalDF = inputDataFull.copy()
    finalDF["DNNPred"] = inputDataFull.shape[0]*[-1.0]
    t0 = time.time()
    tdiff = t0
    for iPred, pred in enumerate(allPredictions):
        logging.info("Transferring prediction %s to input data",iPred)
        finalDF.update(pred)

    logging.debug("Finished transfer of predictions")
    return finalDF
            
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
        "--config",
        action="store",
        type=str,
        help="configuration file",
        required=True
    )
    
    return argumentparser.parse_args(args)

if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])
    initLogging(args.log)

    thisConfig = Config(args.config)
    
    main(thisConfig)

    
