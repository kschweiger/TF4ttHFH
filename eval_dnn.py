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

from scipy.integrate import simps

from utils.ConfigReader import ConfigReaderBase
from utils.utils import reduceArray, getSigBkgArrays

from training.dataProcessing import Sample, Data
from training.DNN import DNN
from train_dnn import TrainingConfig
from eval_autoencoder import initialize
from utils.utils import initLogging, checkNcreateFolder, getROCs

from keras import backend as K

from plotting.plotUtils import make1DHistoPlot, makeROCPlot


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
        self.plotAdditionalDisc = self.getList(self.readConfig.get("Plotting", "addDiscriminators"))

        self.loadTrainingData = self.readConfig.getboolean("General", "loadTrainingData")

        if self.loadTrainingData:
            self.plotAdditionalDisc = []
        self.trainingAttr = None
        with open("{0}/network_attributes.json".format(self.trainingOutput), "r") as f:
            self.trainingAttr = json.load(f)

        self.isBinaryModel = self.trainingAttr["isBinary"]

        self.trainingTransfromation = None
        with open("{0}/network_inputTransformation.json".format(self.trainingOutput), "r") as f:
            self.trainingTransfromation = json.load(f)
        
        self.trainingConifg = TrainingConfig("{0}/usedConfig.cfg".format(self.trainingOutput))

        self.path2Model = "{0}/trainedModel.h5py".format(self.trainingOutput)
        self.path2ModelWeights = "{0}/trainedModel_weights.h5".format(self.trainingOutput)

        self.samples = self.getList(self.readConfig.get("General", "samples"))
        self.sampleGroups = self.readMulitlineOption("General", "sampleGroups", "List", " = ")
        self.signalSampleGroup = self.readConfig.get("General","signalSampleGroup")

        for group in self.sampleGroups:
            for sample in self.sampleGroups[group]:
                if sample not in self.samples:
                    raise RuntimeError("Sample %s from group %s not in defined samples"%(sample, group))

        self.lumi = self.readConfig.getfloat("General", "lumi")
                
        self.sampleInfos = {}
        sampleTuple = namedtuple("sampleTuple", ["input", "label", "xsec", "nGen", "datatype", "selection"])
        
        for sample in self.samples:
            if not self.readConfig.has_section(sample):
                raise KeyError("Sample %s not defined in config (as section) only defined in General.samples"%sample)
            self.sampleInfos[sample] = sampleTuple(input =  self.readConfig.get(sample, "input"),
                                                   label =  self.readConfig.get(sample, "label"),
                                                   xsec =  self.setOptionWithDefault(sample, "xsec", 1.0, "float"),
                                                   nGen =  self.setOptionWithDefault(sample, "nGen", 1.0, "float"),
                                                   datatype =  self.readConfig.get(sample, "datatype"),
                                                   selection =  self.setOptionWithDefault(sample, "selection", None))

        logging.debug("----- Config -----")
        logging.debug("trainingOutput: %s",self.trainingOutput)
        logging.debug("lumi: %s",self.lumi)
        logging.debug("isBinaryModel: %s",self.isBinaryModel)
        logging.debug("plottingOutput: %s",self.plottingOutput)
        logging.debug("plottingPrefix: %s",self.plottingPrefix)
        logging.debug("plottingBins: %s",self.plottingBins)
        logging.debug("plottingRangeMin: %s",self.plottingRangeMin)
        logging.debug("plottingRangeMax: %s",self.plottingRangeMax)
        logging.debug("samples: %s",self.samples)
        logging.debug("signalSampleGroup: %s",self.signalSampleGroup)
        logging.debug("  groups: %s",self.sampleGroups)
        for sample in self.samples:
            logging.debug("  Sample: %s", sample)
            logging.debug("    xsec: %s", self.sampleInfos[sample].xsec)
            logging.debug("    nGen: %s", self.sampleInfos[sample].nGen)
            logging.debug("    datatype: %s", self.sampleInfos[sample].datatype)
            
def getValues(config, allSample, data, thisDNN):
    logging.info("Getting values")
    inputs = {}
    predictions = {}
    weights = {}
    labels = {}
    logging.info("Loading data from train script")
    with open("{0}/testDataArrays.pkl".format(config.trainingOutput), "rb") as pickleIn:
        loadData = pickle.load(pickleIn)

    labelIDs = loadData["classes"]
    for group in data.keys():
        inputs[group] = data[group].getTestData()
        predictions[group] = thisDNN.getPrediction(data[group].getTestData())
        weights[group] = data[group].testTrainingWeights
    if config.loadTrainingData:
        logging.info("Found samples %s", loadData["classes"])
        loadedInput = getSigBkgArrays(loadData["testInputLabels"], loadData["testInputData"])
        loadedWeights = getSigBkgArrays(loadData["testInputLabels"], loadData["testInputWeight"])
        loadedPredictions = getSigBkgArrays(loadData["testInputLabels"], loadData["testPredictionData"])
        for name in loadData["classes"]:
            if name not in inputs.keys():
                raise RuntimeError("Class %s from training not defined", name)
            logging.info("Replacing values for %s with id %s",name, loadData["classes"][name])
            inputs[name] = np.array(loadedInput[int(loadData["classes"][name])])
            weights[name] =  np.array(loadedWeights[int(loadData["classes"][name])])
            predictions[name] =  np.array(loadedPredictions[int(loadData["classes"][name])])

        
    return inputs, predictions, weights, labels, labelIDs
            
def evalDNN_binary(config, allSample, data, thisDNN):
    inputs, predictions, weights, labels, labelIDs = getValues(config, allSample, data, thisDNN)
            
    logging.info("Making discriminator comparison plot")
    classLegend = []
    classValues = []
    classWeights = []
    for group in data.keys():
        classLegend.append(group)
        classValues.append(predictions[group])
        classWeights.append(weights[group])
    make1DHistoPlot(classValues, classWeights,
                    output = "{0}/{1}_{2}".format(config.plottingOutput, config.plottingPrefix, "Classifier"),
                    nBins = config.plottingBins,
                    binRange = (config.plottingRangeMin, config.plottingRangeMax),
                    varAxisName = "DNN Prediction",
                    legendEntries = classLegend,
                    normalized = True)
    

    bkgs = [g for g in data.keys() if g != config.signalSampleGroup]

    ROCPlotvals = {}
    AUCPlotvals = {}
    ROCPlotLabels = []
    for bkg in bkgs:
        logging.info("Getting ROCs for %s", bkg)
        nSignal = len(predictions[config.signalSampleGroup][:,0])
        nBackgorund = len(predictions[bkg][:,0])
        ROCPlotvals[bkg+"DNN"], AUCPlotvals[bkg+"DNN"] = getROCs(np.append(np.array(nSignal*[0]),
                                                               np.array(nBackgorund*[1])),
                                                     np.append(predictions[config.signalSampleGroup][:,0],
                                                               predictions[bkg][:,0]),
                                                     np.append(weights[config.signalSampleGroup],
                                                               weights[bkg]))
        
        ROCPlotLabels.append("DNN : {0} vs {1} - AUC {2:.2f}".format(config.signalSampleGroup, bkg, AUCPlotvals[bkg+"DNN"]))
        for addDisc in config.plotAdditionalDisc:
            ROCPlotvals[bkg+addDisc], AUCPlotvals[bkg+addDisc] = getROCs(np.append(np.array(nSignal*[0]),
                                                                                   np.array(nBackgorund*[1])),
                                                                         np.append(data[config.signalSampleGroup].getTestData(asMatrix=False)[addDisc].values,
                                                                                   data[bkg].getTestData(asMatrix=False)[addDisc].values),
                                                                         np.append(weights[config.signalSampleGroup],
                                                                                   weights[bkg]))
            
                                                         
            ROCPlotLabels.append("{3} : {0} vs {1} - AUC {2:.2f}".format(config.signalSampleGroup, bkg, AUCPlotvals[bkg+addDisc], addDisc))

    makeROCPlot(ROCPlotvals, AUCPlotvals,
                output = "{0}/{1}_{2}".format(config.plottingOutput, config.plottingPrefix, "ROCs"),
                passedLegend = ROCPlotLabels,
                colorOffset = 1)
        

def evalDNN_categorical(config, allSample, data, thisDNN, printMean=True):
    inputs, predictions, weights, labels, labelIDs = getValues(config, allSample, data, thisDNN)
    
    bkgs = [g for g in data.keys() if g != config.signalSampleGroup]
    trainbkgs = [g for g in labelIDs.keys() if g != config.signalSampleGroup]
    
    
    combinePredictions = {}
    for group in data.keys():
        logging.info("Getting combined prediction for %s", group)
        combinePredictions[group] = getCombinedPrediction(predictions[group],
                                                         labelIDs[config.signalSampleGroup],
                                                         [labelIDs[b] for b in trainbkgs])
    classLegend = []
    classValues = []
    classWeights = []
    for group in data.keys():
        if printMean:
            classLegend.append(group+" (Mean = {0:.2f})".format(combinePredictions[group].mean()))
        else:
            classLegend.append(group)
        classValues.append(combinePredictions[group])
        classWeights.append(weights[group])
        
    make1DHistoPlot(classValues, classWeights,
                    output = "{0}/{1}_{2}".format(config.plottingOutput, config.plottingPrefix, "CombClassifier"),
                    nBins = config.plottingBins,
                    binRange = (config.plottingRangeMin, config.plottingRangeMax),
                    varAxisName = "DNN Discriminator",
                    legendEntries = classLegend,
                    normalized = True)

    ROCPlotvals = {}
    AUCPlotvals = {}
    ROCPlotLabels = []

    for bkg in bkgs:
        logging.info("Getting ROCs for %s", bkg)
        nSignal = len(combinePredictions[config.signalSampleGroup])
        nBackgorund = len(combinePredictions[bkg])
        ROCPlotvals[bkg+"DNN"], AUCPlotvals[bkg+"DNN"] = getROCs(np.append(np.array(nSignal*[0]),
                                                               np.array(nBackgorund*[1])),
                                                     np.append(combinePredictions[config.signalSampleGroup],
                                                               combinePredictions[bkg]),
                                                     np.append(weights[config.signalSampleGroup],
                                                               weights[bkg]))
        ROCPlotLabels.append("DNN : {0} vs {1} - AUC {2:.2f}".format(config.signalSampleGroup, bkg, AUCPlotvals[bkg+"DNN"]))
        for addDisc in config.plotAdditionalDisc:
            ROCPlotvals[bkg+addDisc], AUCPlotvals[bkg+addDisc] = getROCs(np.append(np.array(nSignal*[0]),
                                                                   np.array(nBackgorund*[1])),
                                                         np.append(data[config.signalSampleGroup].getTestData(asMatrix=False)[addDisc].values,
                                                                   data[bkg].getTestData(asMatrix=False)[addDisc].values),
                                                         np.append(weights[config.signalSampleGroup],
                                                                   weights[bkg]))

                                                         
            ROCPlotLabels.append("{3} : {0} vs {1} - AUC {2:.2f}".format(config.signalSampleGroup, bkg, AUCPlotvals[bkg+addDisc], addDisc))
        
    makeROCPlot(ROCPlotvals, AUCPlotvals,
                output = "{0}/{1}_{2}".format(config.plottingOutput, config.plottingPrefix, "CombClass_ROCs"),
                passedLegend = ROCPlotLabels,
                colorOffset = 1)

def getBackgroundEstROC(DataROCVals, ttbarROCVals):
    pass
    
        
def getCombinedPrediction(prediction, signalID, bkgIDs, bkgKappas=None):
    """ 
    Combining the output of all categorical classification output nodes
    Note: If not bkgKappas are passed this is equal to only look at the node with ID signalID
    """
    if bkgKappas is None:
        bkgKappas = len(bkgIDs)*[1.0]
    assert len(bkgIDs) == len(bkgKappas)
    bkgSum = None
    for iID, ID in enumerate(bkgIDs):
        logging.debug("Getting column %s",ID)
        if iID == 0:
            bkgSum = bkgKappas[iID] * prediction[:,ID]
        else:
            bkgSum = bkgSum + (bkgKappas[iID] * prediction[:,ID])

    return prediction[:,signalID]/(prediction[:,signalID] + bkgSum)

    
def evalDNN(config):
    checkNcreateFolder(config.plottingOutput, onlyFolder=True)
    logging.info("Initializing samples and data")

    allSample, data = initialize(config)

    thisDNN = DNN(
        identifier = config.trainingConifg.net.name,
        inputDim = config.trainingConifg.net.inputDimention,
        layerDims = config.trainingConifg.net.layerDimentions,
        weightDecay = config.trainingConifg.net.useWeightDecay,
        activation = config.trainingConifg.net.activation,
        outputActivation = config.trainingConifg.net.outputActivation,
        loss = config.trainingConifg.net.loss,
        metric = ["acc"],
        batchSize = config.trainingConifg.net.batchSize
    )

    thisDNN.loadModel(config.trainingOutput)

    if config.isBinaryModel:
        logging.info("Evaluation binary Model")
        evalDNN_binary(config, allSample, data, thisDNN)
    else:
        evalDNN_categorical(config, allSample, data, thisDNN)


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

    evalDNN(config)
