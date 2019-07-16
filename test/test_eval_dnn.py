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

import eval_dnn
import train_dnn

import pytest

@pytest.fixture(scope="module")
def configExpectationRequired():
    config = configparser.ConfigParser()
    config.optionxform = str
    expectation = {}
    expectation["General"] = {"trainingFolder": "/path/to/output",
                              "loadTrainingData": False,
                              "samples" : "Sample1,Sample2",
                              "sampleGroups" : "Sample1 = Sample1\nSample2 = Sample2",
                              "signalSampleGroup" : "Sample1",
                              "lumi" : 41.5}
    
    expectation["Sample1"] = {"input" : "/path/to/input.h5",
                              "label" : "Sample1Label",
                              "xsec" : 1,
                              "nGen" : 1,
                              "datatype" : "data"}
    expectation["Sample2"] = {"input" : "/path/to/input.h5",
                              "label" : "Sample2Label",
                              "xsec" : 0.25,
                              "nGen" : 100000,
                              "datatype" : "mc"}
    expectation["Plotting"] = {"output"  : "/path/to/plotting/output/",
                               "prefix" : "Prefix",
                               "nBins" : 10 ,
                               "binRangeMin" : 0,
                               "binRangeMax" : 1,
                               "addDiscriminators" : "MEM"}
    config['General'] = expectation["General"]
    config['Sample1'] = expectation["Sample1"]
    config['Sample2'] = expectation["Sample2"]
    config['Plotting'] = expectation["Plotting"]
    
    return expectation, config

@pytest.fixture(scope="module")
def trainingsAttr():
    return {
        "LayerInitializerBias": "zeros",
        "LayerInitializerKernel": "random_uniform",
        "activation": "relu",
        "batchSize": 1000,
        "dropoutSeed": None,
        "earlyStop": True,
        "earlyStopMonitor": "val_loss",
        "inputDimention": 5,
        "isBinary": True,
        "layerDimention": [20],
        "loss": "binary_crossentropy",
        "metrics": ["acc"],
        "modelBuilt": True,
        "modelCompiled": True,
        "modelEvaluationTest": [0.5, 0.7],
        "modelTrained": True,
        "nLayer": 1,
        "name": "TestDNNMulti_wDecay",
        "net": None,
        "outputActivation": "sigmoid",
        "useWeightDecay": True,
        "varWeights": {
            "var0": 1,
            "var1": 1,
            "var2": 1,
            "var3": 1,
            "var4": 1,
        },
        "weightDecayLambda": 0.0001
    }

@pytest.fixture(scope="module")
def transformations():
    return {
        "unweighted": {
            "mu" : {
                "var0": 1,
                "var1": 1,
                "var2": 1,
                "var3": 1,
                "var4": 1,
            },
            "std" : {
                "var0": 0.5,
                "var1": 0.5,
                "var2": 0.5,
                "var3": 0.5,
                "var4": 0.5,
            }
        }
    }


def getConfigFromJSON(inJSON):
    config = configparser.ConfigParser()
    config.optionxform = str
    expectation = {}
    expectation["General"] = {"output": "/path/to/output",
                              "trainingVariables": "var0,var1,var2,var3,var4",
                              "samples" : "Sample1",
                              "testPercentage" : 0.2}
    expectation["Sample1"] = {"input" : "/path/to/input.h5",
                              "label" : "Sample1Label",
                              "datatype" : "data"}
    expectation["NeuralNet"] = {"activation" : inJSON["activation"],
                                "outputActivation" : inJSON["outputActivation"],
                                "useWeightDecay" : inJSON["useWeightDecay"],
                                "weightDecayLambda" : inJSON["weightDecayLambda"],
                                "name" : inJSON["name"],
                                "layerDimentions" : ",".join(inJSON["layerDimention"]) if len(inJSON["layerDimention"]) > 1 else inJSON["layerDimention"][0],
                                "optimizer" : "adam",
                                "inputDimention" : inJSON["inputDimention"],
                                "epochs" : 200,
                                "validationSplit" : 0.2,
                                "loss" : inJSON["loss"],
                                "batchSize" : inJSON["batchSize"],
                                "doEarlyStopping" : inJSON["earlyStop"],
                                "patience" : 50}
    
    config['General'] = expectation["General"]
    config['Sample1'] = expectation["Sample1"]
    config['NeuralNet'] = expectation["NeuralNet"]

    return config
    
    
def test_config(mocker, configExpectationRequired, trainingsAttr, transformations):
    configExpectation, mockConfig = configExpectationRequired

    mocker.patch.object(eval_dnn.EvalConfig, "readConfig", return_value = mockConfig)
    mocker.patch('builtins.open', mocker.mock_open(read_data="blubb"))
    mocker.patch("json.load", side_effect = [trainingsAttr, transformations])
    mocker.patch.object(train_dnn.TrainingConfig, "readConfig", return_value = getConfigFromJSON(trainingsAttr))
    
    thisConfig = eval_dnn.EvalConfig("/path/to/config.cfg")

    assert thisConfig.trainingOutput == configExpectation["General"]["trainingFolder"]
    assert thisConfig.plottingOutput == configExpectation["Plotting"]["output"]
    assert thisConfig.plottingPrefix == configExpectation["Plotting"]["prefix"]
    assert thisConfig.plottingBins == configExpectation["Plotting"]["nBins"]
    assert thisConfig.plottingRangeMin == configExpectation["Plotting"]["binRangeMin"]
    assert thisConfig.plottingRangeMax == configExpectation["Plotting"]["binRangeMax"]
    assert thisConfig.plotAdditionalDisc == configExpectation["Plotting"]["addDiscriminators"].split(",")

    assert thisConfig.trainingAttr == trainingsAttr
    assert thisConfig.trainingTransfromation == transformations

    assert thisConfig.isBinaryModel == trainingsAttr["isBinary"]

    assert isinstance(thisConfig.trainingConifg, train_dnn.TrainingConfig)

    expectedSampels = configExpectation["General"]["samples"].split(",")
    assert thisConfig.samples == expectedSampels

    expectedGroups = {}
    _groups = configExpectation["General"]["sampleGroups"]
    for group in _groups.split("\n"):
        groupName, groupElements = group.split(" = ")
        expectedGroups[groupName] = groupElements.split(",")

    assert thisConfig.sampleGroups == expectedGroups
    assert thisConfig.signalSampleGroup ==  configExpectation["General"]["signalSampleGroup"]
    
    for sample in expectedSampels:
        assert thisConfig.sampleInfos[sample].input == configExpectation[sample]["input"]
        assert thisConfig.sampleInfos[sample].label == configExpectation[sample]["label"]
        assert thisConfig.sampleInfos[sample].xsec == configExpectation[sample]["xsec"]
        assert thisConfig.sampleInfos[sample].nGen == configExpectation[sample]["nGen"]
        assert thisConfig.sampleInfos[sample].datatype == configExpectation[sample]["datatype"]
        assert thisConfig.sampleInfos[sample].selection == None

def test_config_exceptions_sampleGroups(mocker, configExpectationRequired, trainingsAttr, transformations):
    _, mockConfig = configExpectationRequired

    mockConfig.__dict__["_sections"]["General"]["sampleGroups"] = mockConfig.__dict__["_sections"]["General"]["sampleGroups"].replace("Sample2", "Sample2,Sample3")
    
    mocker.patch.object(eval_dnn.EvalConfig, "readConfig", return_value = mockConfig)
    mocker.patch('builtins.open', mocker.mock_open(read_data="blubb")) #Mock the open statements
    mocker.patch("json.load", side_effect = [trainingsAttr, transformations]) # Return the read json dict 
    mocker.patch.object(train_dnn.TrainingConfig, "readConfig", return_value = getConfigFromJSON(trainingsAttr))

    
    with pytest.raises(RuntimeError):
        thisConfig = eval_dnn.EvalConfig("/path/to/config.cfg")    

def test_config_exceptions_nonExSample(mocker, configExpectationRequired, trainingsAttr, transformations):
    _, mockConfig = configExpectationRequired

    mockConfig.__dict__["_sections"]["General"]["samples"] += ",Sample3"
    
    mocker.patch.object(eval_dnn.EvalConfig, "readConfig", return_value = mockConfig)
    mocker.patch('builtins.open', mocker.mock_open(read_data="blubb")) #Mock the open statements
    mocker.patch("json.load", side_effect = [trainingsAttr, transformations]) # Return the read json dict 
    mocker.patch.object(train_dnn.TrainingConfig, "readConfig", return_value = getConfigFromJSON(trainingsAttr))
    
    with pytest.raises(KeyError):
        thisConfig = eval_dnn.EvalConfig("/path/to/config.cfg")    

    

        
def test_argParse():
    args = eval_dnn.parseArgs(["--config","path/to/config"])
    assert isinstance(args, argparse.Namespace)

