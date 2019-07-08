"""
Test for the dnn training toplevel script

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import configparser
import argparse

import uproot as root
import numpy as np
import pandas as pd
import copy

from train_dnn import TrainingConfig, parseArgs#, buildActivations, initialize, parseArgs, trainAutoencoder, main
from training.dataProcessing import Sample, Data

import pytest

@pytest.fixture(scope="module")
def configExpectationRequired():
    expectation = {}
    expectation["General"] = {"output": "/path/to/output",
                              "trainingVariables": "variable0,variable1,variable2",
                              "samples" : "Sample1",
                              "testPercentage" : 0.2}
    expectation["Sample1"] = {"input" : "/path/to/input.h5",
                              "label" : "Sample1Label",
                              "datatype" : "data"}
    expectation["NeuralNet"] = {"activation" : "relu",
                                "outputActivation" : "softmax",
                                "name" : "DefaultDNN",
                                "optimizer" : "adagrad",
                                "inputDimention" : 10,
                                "epochs" : 10,
                                "loss" : "categorical_crossentropy"}
    return expectation

@pytest.fixture(scope="module")
def mockExpectationConfig():
    config = configparser.ConfigParser()
    config.optionxform = str
    expectation = {}
    expectation["General"] = {"output": "/path/to/output",
                              "trainingVariables": "variable0,variable1,variable2",
                              "samples" : "Sample1,Sample2",
                              "lumi" : 1.0,
                              "testPercentage" : 0.2,
                              "selection" : "None",
                              "ShuffleData" : True,
                              "SuffleSeed" : "None"}
    expectation["Sample1"] = {"input" : "/path/to/input.h5",
                              "label" : "Sample1Label",
                              "xsec" : 0.25,
                              "nGen" : 100000,
                              "datatype" : "mc"}
    expectation["Sample2"] = {"input" : "/path/to/input.h5",
                              "label" : "Sample2Label",
                              "xsec" : -1,
                              "nGen" : -1,
                              "datatype" : "data",
                              "selection" : "SomeSel"}
    expectation["NeuralNet"] = {"activation" : "relu",
                                "outputActivation" : "softmax",
                                "useWeightDecay" : False,
                                "weightDecayLambda" : 1e-3,
                                "name" : "AllOptDNN",
                                "layerDimentions" : "20,20",
                                "optimizer" : "adagrad",
                                "inputDimention" : 10,
                                "epochs" : 10,
                                "validationSplit" : 0.3,
                                "loss" : "categorical_crossentropy",
                                "batchSize" : 128,
                                "doEarlyStopping" : True,
                                "patience" : 25}    
    

    config['General'] = expectation["General"]
    config['Sample1'] = expectation["Sample1"]
    config['Sample2'] = expectation["Sample2"]
    config['NeuralNet'] = expectation["NeuralNet"]
    
    return expectation, config

def test_config_required(mocker, configExpectationRequired):
    mockConfig = configparser.ConfigParser()
    mockConfig.optionxform = str
    for key in configExpectationRequired:
        mockConfig[key] = configExpectationRequired[key]
        
    mocker.patch.object(TrainingConfig, "readConfig", return_value = mockConfig)
    testConfig = TrainingConfig(path = "/path/to/config.cfg")

    expectedSampels = configExpectationRequired["General"]["samples"].split(",")
    
    assert testConfig.output == configExpectationRequired["General"]["output"]
    assert testConfig.trainingVariables == configExpectationRequired["General"]["trainingVariables"].split(",")
    assert testConfig.lumi == 1.0
    assert testConfig.testPercent == float(configExpectationRequired["General"]["testPercentage"])
    assert testConfig.samples == expectedSampels
    
    assert testConfig.selection is None
    assert testConfig.ShuffleData
    assert testConfig.SuffleSeed is None
    
    assert testConfig.net.activation == configExpectationRequired["NeuralNet"]["activation"]
    assert testConfig.net.outputActivation == configExpectationRequired["NeuralNet"]["outputActivation"]
    assert testConfig.net.useWeightDecay == False
    assert testConfig.net.weightDecayLambda == 1e-5
    assert testConfig.net.name == configExpectationRequired["NeuralNet"]["name"]
    assert testConfig.net.layerDimentions == [int(configExpectationRequired["NeuralNet"]["inputDimention"])]
    assert testConfig.net.inputDimention == int(configExpectationRequired["NeuralNet"]["inputDimention"])
    assert testConfig.net.trainEpochs == int(configExpectationRequired["NeuralNet"]["epochs"])
    assert testConfig.net.loss == configExpectationRequired["NeuralNet"]["loss"]
    assert testConfig.net.validationSplit == 0.25
    assert testConfig.net.optimizer == configExpectationRequired["NeuralNet"]["optimizer"]
    assert testConfig.net.batchSize == 128
    assert testConfig.net.doEarlyStopping == False
    assert testConfig.net.StoppingPatience == 0

    
    for sample in expectedSampels:
        testConfig.sampleInfos[sample].input == configExpectationRequired[sample]["input"]
        testConfig.sampleInfos[sample].label == configExpectationRequired[sample]["label"]
        testConfig.sampleInfos[sample].xsec == 1.0
        testConfig.sampleInfos[sample].nGen == 1.0
        testConfig.sampleInfos[sample].datatype == configExpectationRequired[sample]["datatype"]
        testConfig.sampleInfos[sample].selection == None
        
    assert testConfig.nLayers == 1
    
    if True:
        with open("data/dnn_example_minimal.cfg","w") as f:
            testConfig.readConfig.write(f)

def test_config_all(mocker, mockExpectationConfig):
    configExpectation, mockConfig = mockExpectationConfig        
    mocker.patch.object(TrainingConfig, "readConfig", return_value = mockConfig)
    testConfig = TrainingConfig(path = "/path/to/config.cfg")

    expectedSampels = configExpectation["General"]["samples"].split(",")
    
    assert testConfig.output == configExpectation["General"]["output"]
    assert testConfig.trainingVariables == configExpectation["General"]["trainingVariables"].split(",")
    assert testConfig.lumi == float(configExpectation["General"]["lumi"])
    assert testConfig.testPercent == float(configExpectation["General"]["testPercentage"])
    assert testConfig.samples == expectedSampels
    
    assert testConfig.selection == None if configExpectation["General"]["selection"] == "None" else configExpectation["General"]["selection"]
    assert testConfig.ShuffleData == configExpectation["General"]["ShuffleData"]
    assert testConfig.SuffleSeed == None if configExpectation["General"]["SuffleSeed"] == "None" else configExpectation["General"]["SuffleSeed"]
    
    assert testConfig.net.activation == configExpectation["NeuralNet"]["activation"]
    assert testConfig.net.outputActivation == configExpectation["NeuralNet"]["outputActivation"]
    assert testConfig.net.useWeightDecay == configExpectation["NeuralNet"]["useWeightDecay"]
    assert testConfig.net.weightDecayLambda ==  configExpectation["NeuralNet"]["weightDecayLambda"]
    assert testConfig.net.name == configExpectation["NeuralNet"]["name"]
    assert testConfig.net.layerDimentions == [int(x) for x in configExpectation["NeuralNet"]["layerDimentions"].split(",")]
    assert testConfig.net.inputDimention == int(configExpectation["NeuralNet"]["inputDimention"])
    assert testConfig.net.trainEpochs == int(configExpectation["NeuralNet"]["epochs"])
    assert testConfig.net.loss == configExpectation["NeuralNet"]["loss"]
    assert testConfig.net.validationSplit == configExpectation["NeuralNet"]["validationSplit"]
    assert testConfig.net.optimizer == configExpectation["NeuralNet"]["optimizer"]
    assert testConfig.net.batchSize == int(configExpectation["NeuralNet"]["batchSize"])
    assert testConfig.net.doEarlyStopping == configExpectation["NeuralNet"]["doEarlyStopping"]
    assert testConfig.net.StoppingPatience == int(configExpectation["NeuralNet"]["patience"])

    
    for sample in expectedSampels:
        testConfig.sampleInfos[sample].input == configExpectation[sample]["input"]
        testConfig.sampleInfos[sample].label == configExpectation[sample]["label"]
        testConfig.sampleInfos[sample].xsec == configExpectation[sample]["xsec"]
        testConfig.sampleInfos[sample].nGen == configExpectation[sample]["nGen"]
        testConfig.sampleInfos[sample].datatype == configExpectation[sample]["datatype"]
        if "selection" in configExpectation[sample].keys():
            testConfig.sampleInfos[sample].selection == configExpectation[sample]["selection"]
        else:
            testConfig.sampleInfos[sample].selection == None
            
    assert testConfig.nLayers == len(configExpectation["NeuralNet"]["layerDimentions"].split(","))

    if True:
        with open("data/dnn_example.cfg","w") as f:
            testConfig.readConfig.write(f)


def test_parseArgs():
    args = parseArgs(["--config","path/to/config"])
    assert isinstance(args, argparse.Namespace)
