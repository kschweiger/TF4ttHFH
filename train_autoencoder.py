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
from training.autoencoder import Autoencoder
from training.trainUtils import r_square

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
        self.lumi = self.setOptionWithDefault("General", "lumi", 1.0, "float")
        self.testPercent = self.readConfig.getfloat("General", "testPercentage")
        self.samples = self.getList(self.readConfig.get("General", "samples"))

        self.selection = self.setOptionWithDefault("General", "selection", None)
        self.ShuffleData = self.setOptionWithDefault("General", "ShuffleData", True, "bool")
        self.SuffleSeed = self.setOptionWithDefault("General", "SuffleSeed", None, "str")

        self.selection = None if self.selection == "None" else self.selection
        if self.SuffleSeed is not None:
            self.SuffleSeed = None if self.SuffleSeed == "None"  else int(self.SuffleSeed)
        
        netTuple = namedtuple("netTuple", ["defaultActivationEncoder", "defaultActivationDecoder",
                                           "useWeightDecay", "robustAutoencoder", "name",
                                           "hiddenLayers", "inputDimention", "trainEpochs",
                                           "loss", "validationSplit", "optimizer", "batchSize",
                                           "doEarlyStopping", "StoppingPatience"])
        
        self.net = netTuple(
            defaultActivationEncoder = self.readConfig.get("NeuralNet", "defaultActivationEncoder"),
            defaultActivationDecoder = self.readConfig.get("NeuralNet", "defaultActivationDecoder"),
            useWeightDecay = self.setOptionWithDefault("NeuralNet", "useWeightDecay", False, "bool"),
            robustAutoencoder = self.setOptionWithDefault("NeuralNet", "robustAutoencoder", False, "bool"),
            name = self.readConfig.get("NeuralNet", "name"),
            hiddenLayers = self.setOptionWithDefault("NeuralNet", "hiddenLayers", 0, "int"),
            inputDimention = self.readConfig.getint("NeuralNet", "inputDimention"),
            trainEpochs = self.readConfig.getint("NeuralNet", "epochs"),
            loss = self.readConfig.get("NeuralNet", "loss"),
            validationSplit = self.setOptionWithDefault("NeuralNet", "validationSplit", 0.25, "float"),
            optimizer =  self.readConfig.get("NeuralNet", "optimizer"),
            batchSize = self.setOptionWithDefault("NeuralNet", "batchSize", 128, "int"),
            doEarlyStopping = self.setOptionWithDefault("NeuralNet", "doEarlyStopping", False, "bool"),
            StoppingPatience =  self.setOptionWithDefault("NeuralNet", "patience", 0, "int")
        )

        self.nHiddenLayers = self.net.hiddenLayers
        self.hiddenLayers = []

        hiddenLayerTuple = namedtuple("hiddenLayerTuple" , ["dimention", "activationDecoderSide", "activationEncoderSide"])
        for i in range(self.nHiddenLayers):
            sectionName = "HiddenLayer_"+str(i)
            if not self.readConfig.has_section(sectionName):
                raise KeyError("Expected a section name: %s"%sectionName)
            self.hiddenLayers.append(hiddenLayerTuple(dimention = self.readConfig.getint(sectionName, "dimention"),
                                                      activationDecoderSide = self.readConfig.get(sectionName, "activationDecoderSide"),
                                                      activationEncoderSide = self.readConfig.get(sectionName, "activationEncoderSide")))
        
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
        
        coderTuple = namedtuple("coderTuple", ["activation", "dimention"])
        self.encoder = coderTuple(activation =  (self.net.defaultActivationEncoder
                                                 if not self.readConfig.has_option("Encoder", "activation")
                                                 else self.readConfig.get("Encoder", "activation") ),
                                  dimention =  self.readConfig.getint("Encoder", "dimention"))


        decoderActivation = self.net.defaultActivationDecoder
        if self.readConfig.has_section("Decoder"):
            decoderActivation = self.readConfig.get("Decoder", "activation")
        self.decoder = coderTuple(activation =  decoderActivation,
                                  dimention =  self.net.inputDimention)
        

def buildActivations(config):
    """ Function building the activations are expected by the Autoencoder class """
    if config.nHiddenLayers == 0:
        return config.encoder.activation, config.decoder.activation
    else:
        encoderActivations = []
        decoderActivations = []
        for i in range(config.nHiddenLayers):
            decoderActivations.append(config.hiddenLayers[i].activationDecoderSide)
            encoderActivations.append(config.hiddenLayers[i].activationEncoderSide)
        decoderActivations.append(config.decoder.activation)
        encoderActivations.append(config.encoder.activation)

        if (all(x==encoderActivations[0] for x in encoderActivations) and
            all(x==decoderActivations[0] for x in decoderActivations)):
            return (encoderActivations[0], decoderActivations[0])
        else:
            return (encoderActivations, decoderActivations)

def initialize(config):
    """ Initialze samples and data  """
    #Get samples
    allSamples = []
    for iSample, sample in enumerate(config.samples):
        logging.info("Adding sample %s", sample)
        allSamples.append(
            Sample(
                inFile = config.sampleInfos[sample].input,
                label = config.sampleInfos[sample].label,
                labelID = iSample,
                xsec = config.sampleInfos[sample].xsec,
                nGen = config.sampleInfos[sample].nGen,
                dataType = config.sampleInfos[sample].datatype
            )
        )
        logging.info("Added Sample - %s",allSamples[iSample].getLabelTuple())

    logging.info("Creating training data")
    data = Data(
        samples = allSamples,
        trainVariables = config.trainingVariables,
        testPercent = config.testPercent,
        selection = config.selection,
        shuffleData = config.ShuffleData,
        shuffleSeed = config.SuffleSeed,
        lumi = config.lumi,
        normalizedWeight = True
    )
    
    return allSamples, data

def trainAutoencoder(config, useDevice, batch=False):
    logging.debug("Output folder")
    checkNcreateFolder(config.output, onlyFolder=True)
    logging.debug("Copying used config to outputfolder")
    shutil.copy2(config.path, config.output+"/usedConfig.cfg")

    
    logging.info("Initializing samples and data")
    allSample, data = initialize(config)
    logging.debug("Getting activations")
    thisEncoderActivation, thisDecoderActivation = buildActivations(config)
    logging.debug("Encoder: %s", thisEncoderActivation)
    logging.debug("Decoder: %s", thisDecoderActivation)
    
    logging.info("Initializing autoencoder")
    thisAutoencoder = Autoencoder(
        identifier = config.net.name,
        inputDim = config.net.inputDimention,
        encoderDim = config.encoder.dimention,
        hiddenLayerDim = [config.hiddenLayers[i].dimention for i in range(config.nHiddenLayers)],
        weightDecay = config.net.useWeightDecay,
        robust = config.net.robustAutoencoder,
        encoderActivation = thisEncoderActivation,
        decoderActivation = thisDecoderActivation,
        loss = config.net.loss,
        metric = ['mae',"msle","acc"],
        batchSize = config.net.batchSize
    )

    logging.info("Setting optimizer")
    logging.debug("In config: %s",config.net.optimizer)
    thisAutoencoder.optimizer = config.net.optimizer
    #thisAutoencoder.setOptimizer(optimizerName=config.net.optimizer)

    logging.info("Building model")
    thisAutoencoder.buildModel()
    logging.info("Compiling model")
    thisAutoencoder.compileModel()
    
    trainData = data.getTrainData()
    # print(data.trainVariables)
    # print(data.trainDF[data.trainVariables])
    # #print(data.untransfromedDF[data.trainVariables[1]])
    
    # print(trainData)
    # input("Press ret")
    
    testData = data.getTestData()

    trainWeights = data.trainTrainingWeights
    testWeights = data.testTrainingWeights

    logging.info("Fitting model")
    thisAutoencoder.autoencoder.summary()
    if not batch:
        input("Press ret")
    thisAutoencoder.trainModel(trainData,
                               trainWeights,
                               config.output,
                               epochs = config.net.trainEpochs,
                               valSplit = config.net.validationSplit,
                               thisDevice = "",
                               earlyStopping = (args.stopEarly or config.net.doEarlyStopping),
                               patience = config.net.StoppingPatience)


    logging.info("Evaluation....")
    predictedData = thisAutoencoder.evalModel(testData, testWeights, data.trainVariables, config.output, True, True)
    logging.info("Getting reco error for loss")
    reconstMetric, reconstErrTest = thisAutoencoder.getReconstructionErr(testData)
    reconstMetric, reconstErrTrain = thisAutoencoder.getReconstructionErr(trainData)
    make1DHistoPlot([reconstErrTest, reconstErrTrain], None,
                    "{0}/{1}_{2}".format(config.output, "TrainingReconst", reconstMetric),
                    20,
                    (0, 2),
                    "Loss function",
                    ["Test Sample", "Training Sample"],
                    normalized=True)
    logging.info("Saving testData and weights")
    data2Pickle = { "variables" : config.trainingVariables,
                    "testInputData" : testData,
                    "testWeights" : testWeights,
                    "testPredictionData" : predictedData}

    with open("{0}/testDataArrays.pkl".format(config.output), "wb") as pickleOut:
        pickle.dump(data2Pickle, pickleOut)

    

    thisAutoencoder.saveModel(config.output, data.transformations)
    
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
        "--device",
        action="store",
        type=str,
        help="configuration file",
        default="CPU:0"
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

def main(args, config):
    trainAutoencoder(config, "", batch=args.batchMode)


if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])
    initLogging(args.log)
    config = TrainingConfig(args.config)
    
    main(args, config)
