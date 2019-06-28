"""
Top level training script for autoencoder
"""
import sys
import os
import logging
import shutil
import json

from collections import namedtuple

import numpy as np
import pickle

from utils.ConfigReader import ConfigReaderBase
from training.dataProcessing import Sample, Data
from training.autoencoder import Autoencoder
from training.trainUtils import r_square
from train_autoencoder import TrainingConfig
from utils.utils import initLogging, checkNcreateFolder

from plotting.plotUtils import make1DHistoPlot

class EvalConfig(ConfigReaderBase):
    """
    Containter for setting from the config file. 
    """
    def __init__(self, path):
        super(EvalConfig, self).__init__(path)

        self.trainingOutput = self.readConfig.get("General","trainingFolder")
        self.plottingOutput = self.readConfig.get("Plotting","output")
        self.plottingPrefix = self.readConfig.get("Plotting","prefix")
        self.plottingBins = self.readConfig.getint("Plotting","nBins")
        self.plottingRangeMin = self.readConfig.getfloat("Plotting","binRangeMin")
        self.plottingRangeMax = self.readConfig.getfloat("Plotting","binRangeMax")
        
        self.trainingAttr = None
        with open("{0}/autoencoder_attributes.json".format(self.trainingOutput), "r") as f:
            self.trainingAttr = json.load(f)

        self.trainingTransfromation = None
        with open("{0}/auoencoder_inputTransformation.json".format(self.trainingOutput), "r") as f:
            self.trainingTransfromation = json.load(f)
        
        self.trainingConifg = TrainingConfig("{0}/usedConfig.cfg".format(self.trainingOutput))

        self.path2Model = "{0}/trainedModel.h5py".format(self.trainingOutput)
        self.path2ModelWeights = "{0}/trainedModel_weights.h5".format(self.trainingOutput)

        self.samples = self.getList(self.readConfig.get("General", "samples"))
        self.sampleGroups = self.readMulitlineOption("General", "sampleGroups", "List", " = ")

        for group in self.sampleGroups:
            for sample in self.sampleGroups[group]:
                if sample not in self.samples:
                    raise RuntimeError("Sample %s from group %s not in defined samples"%(sample, group))

        self.lumi = self.readConfig.getfloat("General", "lumi")
                
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

        logging.debug("----- Config -----")
        logging.debug("trainingOutput: %s",self.trainingOutput)
        logging.debug("lumi: %s",self.lumi)
        logging.debug("plottingOutput: %s",self.plottingOutput)
        logging.debug("plottingPrefix: %s",self.plottingPrefix)
        logging.debug("plottingBins: %s",self.plottingBins)
        logging.debug("plottingRangeMin: %s",self.plottingRangeMin)
        logging.debug("plottingRangeMax: %s",self.plottingRangeMax)
        logging.debug("samples: %s",self.samples)
        logging.debug("  groups: %s",self.sampleGroups)
        for sample in self.samples:
            logging.debug("  Sample: %s", sample)
            logging.debug("    xsec: %s", self.sampleInfos[sample].xsec)
            logging.debug("    nGen: %s", self.sampleInfos[sample].nGen)
            logging.debug("    datatype: %s", self.sampleInfos[sample].datatype)
            

            
def initialize(config):
    """ Initialze samples and data  """
    #Get samples
    allSamples = {}
    for group in config.sampleGroups:
        logging.info("Processing group %s", group)
        allSamples[group] = []
        for iSample, sample in enumerate(config.sampleGroups[group]):
            logging.info("Adding sample %s", sample)
            allSamples[group].append(
                Sample(
                    inFile = config.sampleInfos[sample].input,
                    label = config.sampleInfos[sample].label,
                    labelID = iSample,
                    xsec = config.sampleInfos[sample].xsec,
                    nGen = config.sampleInfos[sample].nGen,
                    dataType = config.sampleInfos[sample].datatype
                )
            )
            logging.info("Added Sample - %s",allSamples[group][iSample].getLabelTuple())

    logging.info("Creating training data")
    data = {}
    for group in config.sampleGroups:
        logging.info("Generating data for group %s", group)
        logging.debug(" Group: %s", allSamples[group])
        data[group] = Data(
            samples = allSamples[group],
            trainVariables = config.trainingConifg.trainingVariables,
            testPercent = 1.0,
            selection = config.trainingConifg.selection,
            shuffleData = True,
            shuffleSeed = config.trainingConifg.SuffleSeed,
            lumi = config.lumi,
            transform = False
        )

        data[group].transformations = config.trainingTransfromation
        data[group].doTransformation = True
    return allSamples, data

        
def evalAutoencoder(config):
    checkNcreateFolder(config.plottingOutput, onlyFolder=True)
    logging.info("Initializing samples and data")
    allSample, data = initialize(config)

    logging.info("Initializing autoencoder")
    thisAutoencoder = Autoencoder(
        identifier = config.trainingConifg.net.name,
        inputDim = config.trainingConifg.net.inputDimention,
        encoderDim = config.trainingConifg.encoder.dimention,
        hiddenLayerDim = [config.trainingConifg.hiddenLayers[i].dimention for i in range(config.trainingConifg.nHiddenLayers)],
        weightDecay = config.trainingConifg.net.useWeightDecay,
        robust = config.trainingConifg.net.robustAutoencoder,
        encoderActivation = "", #Do not need that here
        decoderActivation = "",#Do not need that here
        loss = config.trainingConifg.net.loss,
        metric = ['mae',"msle","mse"],
        batchSize = config.trainingConifg.net.batchSize
    )

    thisAutoencoder.loadModel(config.trainingOutput)

    reconstErr = {}
    reconstErrList = []
    legend = []
    
    inputDatas = []
    inputWeights = []
    predictedDatas = []
    datasets = []
    for key in data:
        inputData = data[key].getTestData(asMatrix=True)
        inputWeight = data[key].testTrainingWeights
        predictedData = thisAutoencoder.evalModel(inputData, inputWeight,
                                                  config.trainingConifg.trainingVariables,
                                                  config.plottingOutput,
                                                  plotPrediction=True,
                                                  splitNetwork=False,
                                                  plotPostFix="_"+key)
        inputDatas.append(inputData)
        predictedDatas.append(predictedData)
        inputWeights.append(inputWeight)
        datasets.append(key)
        reconstMetric, reconstErr[key] = thisAutoencoder.getReconstructionErr(inputData)
        reconstErrList.append(reconstErr[key])
        legend.append(key)

    data2Pickle = { "variables" : config.trainingConifg.trainingVariables,
                    "datasets" : datasets,
                    "inputData" : inputDatas,
                    "predictionData" : predictedDatas}

    with open("{0}/evalDataArrays.pkl".format(config.plottingOutput), "wb") as pickleOut:
        pickle.dump(data2Pickle, pickleOut)

    with open("{0}/testDataArrays.pkl".format(config.trainingOutput), "rb") as testPickle:
        testDataTraining = pickle.load(testPickle)

    inputWeights.append(testDataTraining["testWeights"])
    inputDatas.append(testDataTraining["testInputData"])
    reconstMetricTest, reconstErrTest = thisAutoencoder.getReconstructionErr(testDataTraining["testInputData"])
    reconstErrList.append(reconstErrTest)
    for iVar, var in enumerate(config.trainingConifg.trainingVariables):
        make1DHistoPlot([inputDatas[i][:,iVar] for i in range(len(inputDatas))],
                        inputWeights,
                        "{0}/{1}_input_{2}".format(config.plottingOutput, config.plottingPrefix, var),
                        nBins = 40,
                        binRange = (-10, 10),
                        varAxisName = var,
                        legendEntries = legend,
                        normalized=True)
        
    make1DHistoPlot(reconstErrList, inputWeights,
                    "{0}/{1}_{2}".format(config.plottingOutput, config.plottingPrefix, reconstMetric),
                    config.plottingBins,
                    (config.plottingRangeMin, config.plottingRangeMax),
                    reconstMetric,
                    legend,
                    normalized=True)
    make1DHistoPlot(reconstErrList, inputWeights,
                    "{0}/{1}_{2}_log".format(config.plottingOutput, config.plottingPrefix, reconstMetric),
                    config.plottingBins,
                    (config.plottingRangeMin, config.plottingRangeMax),
                    reconstMetric,
                    legend,
                    normalized=True,
                    log=True)

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
    config = EvalConfig(args.config)

    evalAutoencoder(config)
