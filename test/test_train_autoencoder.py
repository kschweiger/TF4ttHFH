"""
Test for the autoencoder training toplevel script

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

from train_autoencoder import TrainingConfig, buildActivations, initialize, parseArgs, trainAutoencoder, main
from training.dataProcessing import Sample, Data

import pytest

@pytest.fixture()
def testData(means = (10, 12, 7), stddev = (2, 1.5, 1)):
    size = 10000
    data = {"evt" :  np.arange(size),
            "run" :  np.arange(size),
            "lumi" :  np.arange(size),
            "puWeight" : np.array(size*[1.0]),
            "genWeight" : np.array(size*[1.0]),
            "btagWeight_shape" : np.array(size*[1.0]),
            "weight_CRCorr" : np.array(size*[1.0]),
            "triggerWeight" : np.array(size*[1.0])}

    names = []
    for i in range(len(means)):
        names.append("variable"+str(i))

    for iname, name in enumerate(names):
        data[name] = np.random.normal(means[iname], stddev[iname], size)
    df = pd.DataFrame(data)
    df.set_index(["evt","run","lumi"], inplace=True)
    return df

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
    expectation["NeuralNet"] = {"defaultActivationEncoder" : "tanh",
                                "defaultActivationDecoder" : "tanh",
                                "name" : "AutoEncoder",
                                "optimizer" : "rmsprop",
                                "inputDimention" : 30,
                                "epochs" : 10,
                                "loss" : "RMS"}
    expectation["Encoder"] = {"dimention" : 5} 
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
                              "datatype" : "data"}
    expectation["NeuralNet"] = {"defaultActivationEncoder" : "tanh",
                                "defaultActivationDecoder" : "tanh",
                                "useWeightDecay" : False,
                                "robustAutoencoder" : False,
                                "name" : "AutoEncoder",
                                "hiddenLayers" : 1,
                                "optimizer" : "rmsprop",
                                "inputDimention" : 30,
                                "epochs" : 10,
                                "validationSplit" : 0.3,
                                "loss" : "RMS",
                                "batchSize" : 16}
    expectation["Decoder"] = {"activation" : "linear"}
    expectation["Encoder"] = {"dimention" : 5,
                              "activation" : "tanh"} 
    expectation["HiddenLayer_0"] = {"dimention" : 20,
                                    "activationDecoderSide" : "sigmoid",
                                    "activationEncoderSide" : "tanh"}
    
    

    config['General'] = expectation["General"]
    config['Sample1'] = expectation["Sample1"]
    config['Sample2'] = expectation["Sample2"]
    config['NeuralNet'] = expectation["NeuralNet"]
    config['Decoder'] = expectation["Decoder"]
    config['Encoder'] = expectation["Encoder"]
    config['HiddenLayer_0'] = expectation["HiddenLayer_0"]
    
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
    
    assert testConfig.net.defaultActivationEncoder == configExpectationRequired["NeuralNet"]["defaultActivationEncoder"]
    assert testConfig.net.defaultActivationDecoder == configExpectationRequired["NeuralNet"]["defaultActivationDecoder"]
    assert testConfig.net.useWeightDecay == False
    assert testConfig.net.robustAutoencoder == False
    assert testConfig.net.name == configExpectationRequired["NeuralNet"]["name"]
    assert testConfig.net.hiddenLayers == 0
    assert testConfig.net.inputDimention == int(configExpectationRequired["NeuralNet"]["inputDimention"])
    assert testConfig.net.trainEpochs == int(configExpectationRequired["NeuralNet"]["epochs"])
    assert testConfig.net.loss == configExpectationRequired["NeuralNet"]["loss"]
    assert testConfig.net.validationSplit == 0.25
    assert testConfig.net.optimizer == configExpectationRequired["NeuralNet"]["optimizer"]
    assert testConfig.net.batchSize == 128
    
    for sample in expectedSampels:
        testConfig.sampleInfos[sample].input == configExpectationRequired[sample]["input"]
        testConfig.sampleInfos[sample].label == configExpectationRequired[sample]["label"]
        testConfig.sampleInfos[sample].xsec == 1.0
        testConfig.sampleInfos[sample].nGen == 1.0
        testConfig.sampleInfos[sample].datatype == configExpectationRequired[sample]["datatype"]
        
    assert testConfig.nHiddenLayers == 0
    assert testConfig.hiddenLayers == []

    assert testConfig.encoder.activation == configExpectationRequired["NeuralNet"]["defaultActivationEncoder"]
    assert testConfig.encoder.dimention == configExpectationRequired["Encoder"]["dimention"]
    assert testConfig.decoder.activation == configExpectationRequired["NeuralNet"]["defaultActivationEncoder"]
    assert testConfig.decoder.dimention == configExpectationRequired["NeuralNet"]["inputDimention"]

    if True:
        with open("data/autoencoder_example_minimal.cfg","w") as f:
            testConfig.readConfig.write(f)
    
def test_config_allplusHidden(mocker, mockExpectationConfig):
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
    
    assert testConfig.net.defaultActivationEncoder == configExpectation["NeuralNet"]["defaultActivationEncoder"]
    assert testConfig.net.defaultActivationDecoder == configExpectation["NeuralNet"]["defaultActivationDecoder"]
    assert testConfig.net.useWeightDecay == configExpectation["NeuralNet"]["useWeightDecay"]
    assert testConfig.net.robustAutoencoder == configExpectation["NeuralNet"]["robustAutoencoder"]
    assert testConfig.net.name == configExpectation["NeuralNet"]["name"]
    assert testConfig.net.hiddenLayers == int(configExpectation["NeuralNet"]["hiddenLayers"])
    assert testConfig.net.inputDimention == int(configExpectation["NeuralNet"]["inputDimention"])
    assert testConfig.net.trainEpochs == int(configExpectation["NeuralNet"]["epochs"])
    assert testConfig.net.loss == configExpectation["NeuralNet"]["loss"]
    assert testConfig.net.validationSplit == configExpectation["NeuralNet"]["validationSplit"]
    assert testConfig.net.optimizer == configExpectation["NeuralNet"]["optimizer"]
    assert testConfig.net.batchSize == int(configExpectation["NeuralNet"]["batchSize"])
    
    for sample in expectedSampels:
        testConfig.sampleInfos[sample].input == configExpectation[sample]["input"]
        testConfig.sampleInfos[sample].label == configExpectation[sample]["label"]
        testConfig.sampleInfos[sample].xsec == configExpectation[sample]["xsec"]
        testConfig.sampleInfos[sample].nGen == configExpectation[sample]["nGen"]
        testConfig.sampleInfos[sample].datatype == configExpectation[sample]["datatype"]

    assert testConfig.nHiddenLayers == int(configExpectation["NeuralNet"]["hiddenLayers"])

    assert testConfig.encoder.activation == configExpectation["Encoder"]["activation"]
    assert testConfig.encoder.dimention == configExpectation["Encoder"]["dimention"]
    assert testConfig.decoder.activation == configExpectation["Decoder"]["activation"]
    assert testConfig.decoder.dimention == configExpectation["NeuralNet"]["inputDimention"]

    for i in range(int(configExpectation["NeuralNet"]["hiddenLayers"])):
        assert testConfig.hiddenLayers[i].dimention == configExpectation["HiddenLayer_"+str(i)]["dimention"]
        assert testConfig.hiddenLayers[i].activationDecoderSide == configExpectation["HiddenLayer_"+str(i)]["activationDecoderSide"]
        assert testConfig.hiddenLayers[i].activationEncoderSide == configExpectation["HiddenLayer_"+str(i)]["activationEncoderSide"]
    
    if True:
        with open("data/autoencoder_example.cfg","w") as f:
            testConfig.readConfig.write(f)


def test_buildActivation_nohiddenLayers():
    relevandConfig = SimpleNamespace()
    setattr(relevandConfig, "nHiddenLayers", 0)
    encoder = SimpleNamespace()
    setattr(encoder, "activation", "sigmoid")
    decoder = SimpleNamespace()
    setattr(decoder, "activation", "linear")
    
    setattr(relevandConfig, "decoder", decoder)
    setattr(relevandConfig, "encoder", encoder)

    assert buildActivations(relevandConfig) == ("sigmoid", "linear")

def test_buildActivation_hiddenLayers_same():
    relevandConfig = SimpleNamespace()
    setattr(relevandConfig, "nHiddenLayers", 1)
    encoder = SimpleNamespace()
    setattr(encoder, "activation", "sigmoid")
    decoder = SimpleNamespace()
    setattr(decoder, "activation", "linear")
    hiddenLayer = SimpleNamespace()
    setattr(hiddenLayer, "activationDecoderSide", "linear")
    setattr(hiddenLayer, "activationEncoderSide", "sigmoid")
    
    setattr(relevandConfig, "decoder", decoder)
    setattr(relevandConfig, "encoder", encoder)
    setattr(relevandConfig, "hiddenLayers", [hiddenLayer])

    assert buildActivations(relevandConfig) == ("sigmoid", "linear")

def test_buildActivation_hiddenLayers_different():
    relevandConfig = SimpleNamespace()
    setattr(relevandConfig, "nHiddenLayers", 1)
    encoder = SimpleNamespace()
    setattr(encoder, "activation", "sigmoid")
    decoder = SimpleNamespace()
    setattr(decoder, "activation", "linear")
    hiddenLayer = SimpleNamespace()
    setattr(hiddenLayer, "activationDecoderSide", "tanh")
    setattr(hiddenLayer, "activationEncoderSide", "tanh")
    
    setattr(relevandConfig, "decoder", decoder)
    setattr(relevandConfig, "encoder", encoder)
    setattr(relevandConfig, "hiddenLayers", [hiddenLayer])

    assert buildActivations(relevandConfig) == (["tanh","sigmoid"], ["tanh", "linear"])


def test_initialize_samplesAndData(mocker, mockExpectationConfig, testData):
    configExpectation, mockConfig = mockExpectationConfig
    mocker.patch.object(TrainingConfig, "readConfig", return_value = mockConfig)
    testConfig = TrainingConfig(path = "/path/to/config.cfg")
    testConfig.samples = ["Sample1"]
    mocker.patch("pandas.read_hdf", return_value=testData)
    print(testData)
    initSamples, initData = initialize(testConfig)

    for sample in initSamples:
        assert isinstance(sample, Sample)
    assert isinstance(initData, Data)
    
def test_parseArgs():
    args = parseArgs(["--config","path/to/config"])
    assert isinstance(args, argparse.Namespace)
