"""
Top level training script for autoencoder
"""
import sys
import os
import logging
import shutil

from collections import namedtuple

import numpy as np
import pickle

from utils.ConfigReader import ConfigReaderBase
from training.dataProcessing import Sample, Data
from training.DNN import DNN

from train_autoencoder import initialize

from plotting.plotUtils import make1DHistoPlot

from utils.utils import initLogging, checkNcreateFolder

from tensorflow import Session, device, ConfigProto

class TrainingConfig(ConfigReaderBase):
    """
    Containter for setting from the config file. 
    
    Required sections (required options) [optional options]:
    General (output, trainingVariables, samples, testPercentage) [selection,ShuffleData, SuffleSeed, lumi]
    NeutralNet (defaultActivationEncoder, defaultActivationDecoder, name, inputDimention) [useWeightDecay, robustAutoencoder, hiddenLayers]
    Encoder (dimention) [activation]
    
    Section dependent on required options (required options) [optional options]
    **SampleName** as defined in General.samples (input, label, aatatype) [xsec nGen] 
    HiddenLayer_**0+** one per 1+ intager in NeuralNet.hiddeLayers (dimention) [activationDecoderSide, activationEncoderSide]

    Optional selctions (required options)
    Decoder (activation)

    Args:
      path (str) : path to the config file --> Relative to directory of convertTree.py
    """
    def __init__(self, path):
        super(TrainingConfig, self).__init__(path)

        self.output = self.readConfig.get("General", "output")
        self.trainingVariables = self.getList(self.readConfig.get("General", "trainingVariables"))
        logging.debug("Got %s input variables", len(self.trainingVariables))
        self.lumi = self.setOptionWithDefault("General", "lumi", 1.0, "float")
        self.testPercent = self.readConfig.getfloat("General", "testPercentage")
        self.samples = self.getList(self.readConfig.get("General", "samples"))

        self.selection = self.setOptionWithDefault("General", "selection", None)
        self.ShuffleData = self.setOptionWithDefault("General", "ShuffleData", True, "bool")
        self.SuffleSeed = self.setOptionWithDefault("General", "SuffleSeed", None, "str")

        self.selection = None if self.selection == "None" else self.selection
        if self.SuffleSeed is not None:
            self.SuffleSeed = None if self.SuffleSeed == "None"  else int(self.SuffleSeed)
        
        netTuple = namedtuple("netTuple", ["activation", "outputActivation",
                                           "useWeightDecay", "name",
                                           "layerDimentions", "inputDimention", "trainEpochs",
                                           "loss", "validationSplit", "optimizer", "batchSize",
                                           "doEarlyStopping", "StoppingPatience"])
        
        self.net = netTuple(
            activation = self.readConfig.get("NeuralNet", "activation"),
            outputActivation = self.readConfig.get("NeuralNet", "outputActivation"),
            useWeightDecay = self.setOptionWithDefault("NeuralNet", "useWeightDecay", False, "bool"),
            name = self.readConfig.get("NeuralNet", "name"),
            inputDimention = self.readConfig.getint("NeuralNet", "inputDimention"),
            layerDimentions = self.setOptionWithDefault("NeuralNet", "layerDimentions", [self.readConfig.getint("NeuralNet", "inputDimention")], "intlist"),
            trainEpochs = self.readConfig.getint("NeuralNet", "epochs"),
            loss = self.readConfig.get("NeuralNet", "loss"),
            validationSplit = self.setOptionWithDefault("NeuralNet", "validationSplit", 0.25, "float"),
            optimizer =  self.readConfig.get("NeuralNet", "optimizer"),
            batchSize = self.setOptionWithDefault("NeuralNet", "batchSize", 128, "int"),
            doEarlyStopping = self.setOptionWithDefault("NeuralNet", "doEarlyStopping", False, "bool"),
            StoppingPatience =  self.setOptionWithDefault("NeuralNet", "patience", 0, "int")
        )

        logging.debug("Get layer dimentions: %s",self.net.layerDimentions)
        
        self.nLayers = len(self.net.layerDimentions)
        
        self.sampleInfos = {}
        sampleTuple = namedtuple("sampleTuple", ["input", "label", "xsec", "nGen", "datatype"])
        
        for sample in self.samples:
            if not self.readConfig.has_section(sample):
                raise KeyError("Sample %s not defined in config (as section) only defined in General.samples"%sample)
            self.sampleInfos[sample] = sampleTuple(input =  self.readConfig.get(sample, "input"),
                                                    label =  self.readConfig.get(sample, "label"),
                                                    xsec =  self.setOptionWithDefault(sample, "xsec", 1.0, "float"),
                                                    nGen =  self.setOptionWithDefault(sample, "nGen", 1.0, "float"),
                                                    datatype =  self.readConfig.get(sample, "datatype"))

def trainDNN(config, batch=False):
    logging.debug("Output folder")
    checkNcreateFolder(config.output, onlyFolder=True)
    logging.debug("Copying used config to outputfolder")
    shutil.copy2(config.path, config.output+"/usedConfig.cfg")

    logging.info("Initializing samples and data")
    allSample, data = initialize(config, incGenWeights=True)

    logging.info("Initializing DNN")
    thisDNN = DNN(
        identifier = config.net.name,
        inputDim = config.net.inputDimention,
        layerDims = config.net.layerDimentions,
        weightDecay = config.net.useWeightDecay,
        activation = config.net.activation,
        outputActivation = config.net.outputActivation,
        loss = config.net.loss,
        metric = ["acc"],
        batchSize = config.net.batchSize
        )

    logging.info("Setting optimizer")
    logging.debug("In config: %s",config.net.optimizer)
    thisDNN.optimizer = config.net.optimizer

    logging.info("Building model")
    if config.net.loss == "binary_crossentropy":
        thisDNN.buildModel(nClasses = 1)
    else:
        thisDNN.buildModel(nClasses = len(data.outputClasses))
    logging.info("Compiling model")
    thisDNN.compileModel()

    thisDNN.network.summary()
    if not batch:
        input("Press ret")

    trainData = data.getTrainData()
    trainLabels = data.trainLabels
    trainWeights = data.trainTrainingWeights
    
    # print("------------------------------")
    # print(trainData)
    # print("------------------------------")
    # print(trainLabels)
    # print("------------------------------")
    # print(trainWeights, trainWeights.mean(), trainWeights.std())
    # print("------------------------------")
    # print(data.trainDF["eventWeightUnNorm"].values, data.trainDF["eventWeightUnNorm"].mean(), data.trainDF["eventWeightUnNorm"].std())
    # print("------------------------------")
    # print(data.trainDF["puWeight"].values, data.trainDF["puWeight"].mean(), data.trainDF["puWeight"].std())
    # print("------------------------------")
    # print(data.trainDF["nPVs"].values, data.trainDF["nPVs"].mean(), data.trainDF["nPVs"].std())
    # print("------------------------------")
    # print(data.trainDF["btagNorm"].values, data.trainDF["btagNorm"].mean(), data.trainDF["btagNorm"].std())
    # print("------------------------------")
    # print(data.trainDF["btagWeight_shape"].values, data.trainDF["btagWeight_shape"].mean(), data.trainDF["btagWeight_shape"].std())
    # print("------------------------------")
    # print(data.trainDF["weight_CRCorr"].values, data.trainDF["weight_CRCorr"].mean(), data.trainDF["weight_CRCorr"].std())

    logging.info("Training DNN")
    thisDNN.trainModel(
        trainData, trainLabels,
        trainWeights,
        config.output,
        epochs = config.net.trainEpochs,
        valSplit = config.net.validationSplit,
        earlyStopping = config.net.doEarlyStopping,
        patience = config.net.StoppingPatience
    )

    testData = data.getTestData()
    testLabels = data.testLabels
    testWeights = data.testTrainingWeights

    #TODO: Make this configurable
    ROCMetrics = []
    for metric in ["MEM", "Detaj_5"]:
        ROCMetrics.append((metric, data.getTestData(asMatrix=False)[metric].values))
    
    logging.info("Model evaluation")
    thisDNN.evalModel(testData,  testWeights,
                      testLabels,
                      trainData, trainWeights,
                      trainLabels,
                      config.trainingVariables,
                      config.output,
                      plotMetics=True,
                      saveData=True,
                      addROCMetrics = ROCMetrics)

    logging.info("Saving model")
    thisDNN.saveModel(config.output, data.transformations)
    
        
def main(args, config):
    trainDNN(config, batch=args.batchMode)
    
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
    argumentparser.add_argument(
        "--batchMode",
        action="store_true",
    )
    argumentparser.add_argument(
        "--stopEarly",
        action="store_true",
    )
    return argumentparser.parse_args(args)

if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])
    initLogging(args.log)
    config = TrainingConfig(args.config)
    
    main(args, config)
