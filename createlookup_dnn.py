import sys
import os
import logging
import shutil
import json
import pickle

import pandas as pd

from collections import namedtuple

from training.dataProcessing import Sample, Data
from training.DNN import DNN
from utils.utils import initLogging, checkNcreateFolder

from train_dnn import TrainingConfig


def getModelDefinitions(path2Model):
    definitons = namedtuple("definitons", ["config", "transformations", "attributes", "trainingOutput"])
    
    config = TrainingConfig("{0}/usedCOnfig.cfg".format(path2Model))

    trainingTransfromation = None
    with open("{0}/network_inputTransformation.json".format(path2Model), "r") as f:
            trainingTransfromation = json.load(f)

    trainingAttr = None
    with open("{0}/network_attributes.json".format(path2Model), "r") as f:
        trainingAttr = json.load(f)

    return definitons(config = config,
                      transformations = trainingTransfromation,
                      attributes = trainingAttr,
                      trainingOutput = path2Model)

def getSampleData(modelDefinitions, inputFile):
    inputSample = Sample(inFile = inputFile,
                         label = "Input",
                         labelID = 1)

    inputData = Data(samples = [inputSample],
                     trainVariables = modelDefinitions.config.trainingVariables,
                     testPercent = 1.0,
                     transform = False)

    inputData.transformations = modelDefinitions.transformations
    inputData.doTransformation = True
    
    return inputSample, inputData
                         
def getPredictions(thisDNN, modelDefinitions, sample, data):
    with open("{0}/testDataArrays.pkl".format(modelDefinitions.trainingOutput), "rb") as pickleIn:
        loadData = pickle.load(pickleIn)

    labelIDs = loadData["classes"]
    signalID = labelIDs["ttHbb"]

    inputs = data.getTestData()
    prediction = thisDNN.getPrediction(inputs)

    #Create Dataframe of input data and prediction
    #Get the input data as Dataframe 
    inputDF = data.getTestData(asMatrix=False)
    #save which columns are indices
    origIndices = inputDF.index.names
    #reset indeces (so prediction can be added w/o problems
    inputDF = inputDF.reset_index()

    #Convert predicton from KERAS (np.ndarray) to a pandas.DataFrame
    predictionDF = pd.DataFrame({"DNNPred": prediction[:,signalID]})

    #Add prediction to dataframe with input
    DFwPrediction = inputDF.join(predictionDF)
    DFwPrediction.set_index(origIndices, inplace = True)

    return DFwPrediction

def setupDNN(modelDefinitions):
    thisDNN = DNN(
        identifier = modelDefinitions.config.net.name,
        inputDim = modelDefinitions.config.net.inputDimention,
        layerDims = modelDefinitions.config.net.layerDimentions,
        weightDecay = modelDefinitions.config.net.useWeightDecay,
        activation = modelDefinitions.config.net.activation,
        outputActivation = modelDefinitions.config.net.outputActivation,
        loss = modelDefinitions.config.net.loss,
        metric = ["acc"],
        batchSize = modelDefinitions.config.net.batchSize
    )

    thisDNN.loadModel(modelDefinitions.trainingOutput
    )

    return thisDNN
 
def writeLookupTable(outputData, outPath, outName):
    indices = outputData.index.names
    loopupTable = {}
    for index, row in outputData.iterrows():
        thisIndex = ":".join(str(x) for x in index)
        if thisIndex in  loopupTable.keys():
            raise RuntimeError("Indices should be unique")
        loopupTable[thisIndex] = row["DNNPred"]

    pickleOutputname = "{0}/{1}.pkl".format(outPath, outName)
    logging.info("Saving lookup table at %s", pickleOutputname)
    with open(pickleOutputname, "wb") as pickleOut:
        pickle.dump(loopupTable, pickleOut)
    
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
        "--model",
        action="store",
        type=str,
        help="Path to trained model",
        required=True
    )
    argumentparser.add_argument(
        "--input",
        action="store",
        type=str,
        help="h5py file for which to lookup table is written",
        required=True
    )
    argumentparser.add_argument(
        "--output",
        action="store",
        type=str,
        help="path to output folder",
        required=True
    )
    
    return argumentparser.parse_args(args)

if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])
    initLogging(args.log)

    checkNcreateFolder(args.output, onlyFolder=True)
    
    modelDefinitions = getModelDefinitions(args.model)

    thisDNN = setupDNN(modelDefinitions)
    
    inputSample, inputData = getSampleData(modelDefinitions, args.input)
    
    dfPrediction = getPredictions(thisDNN, modelDefinitions, inputSample, inputData)

    inputFileName = args.input.split("/")[-1]
    inputFileName = inputFileName.split(".")[0]
    
    writeLookupTable(dfPrediction, args.output, inputFileName)
