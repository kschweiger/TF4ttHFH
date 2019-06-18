"""
Test for the autoencoder training toplevel script

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import configparser

 
import uproot as root
import pandas as pd
import copy

from train_autoencoder import TrainingConfig, AutoencoderTrainer, buildActivations

import pytest

@pytest.fixture(scope="module")
def configExpectationRequired():
    expectation = {}
    expectation["General"] = {"output": "/path/to/output",
                              "trainingVariables": "variable1,varibale2,variable3",
                              "samples" : "Sample1",
                              "testPercentage" : 0.2}
    expectation["Sample1"] = {"input" : "/path/to/input.h5",
                              "label" : "Sample1Label",
                              "datatype" : "data"}
    expectation["NeuralNet"] = {"defaultActivationEncoder" : "tanh",
                                 "defaultActivationDecoder" : "tanh",
                                 "name" : "AutoEncoder",
                                 "inputDimention" : 30}
    expectation["Encoder"] = {"dimention" : 5} 
    return expectation

@pytest.fixture(scope="module")
def mockExpectationConfig():
    config = configparser.ConfigParser()
    expectation = {}
    expectation["General"] = {"output": "/path/to/output",
                              "trainingVariables": "variable1,varibale2,variable3",
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
                                 "inputDimention" : 30}
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
    
    for sample in expectedSampels:
        testConfig.trainSamples[sample].input == configExpectationRequired[sample]["input"]
        testConfig.trainSamples[sample].label == configExpectationRequired[sample]["label"]
        testConfig.trainSamples[sample].xsec == 1.0
        testConfig.trainSamples[sample].nGen == 1.0
        testConfig.trainSamples[sample].datatype == configExpectationRequired[sample]["datatype"]

    assert testConfig.nHiddenLayers == 0
    assert testConfig.hiddenLayers == []

    assert testConfig.encoder.activation == configExpectationRequired["NeuralNet"]["defaultActivationEncoder"]
    assert testConfig.encoder.dimention == configExpectationRequired["Encoder"]["dimention"]
    assert testConfig.decoder.activation == configExpectationRequired["NeuralNet"]["defaultActivationEncoder"]
    assert testConfig.decoder.dimention == configExpectationRequired["NeuralNet"]["inputDimention"]

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
    
    for sample in expectedSampels:
        testConfig.trainSamples[sample].input == configExpectation[sample]["input"]
        testConfig.trainSamples[sample].label == configExpectation[sample]["label"]
        testConfig.trainSamples[sample].xsec == configExpectation[sample]["xsec"]
        testConfig.trainSamples[sample].nGen == configExpectation[sample]["nGen"]
        testConfig.trainSamples[sample].datatype == configExpectation[sample]["datatype"]

    assert testConfig.nHiddenLayers == int(configExpectation["NeuralNet"]["hiddenLayers"])

    assert testConfig.encoder.activation == configExpectation["Encoder"]["activation"]
    assert testConfig.encoder.dimention == configExpectation["Encoder"]["dimention"]
    assert testConfig.decoder.activation == configExpectation["Decoder"]["activation"]
    assert testConfig.decoder.dimention == configExpectation["NeuralNet"]["inputDimention"]

    for i in range(int(configExpectation["NeuralNet"]["hiddenLayers"])):
        assert testConfig.hiddenLayers[i].dimention == configExpectation["HiddenLayer_"+str(i)]["dimention"]
        assert testConfig.hiddenLayers[i].activationDecoderSide == configExpectation["HiddenLayer_"+str(i)]["activationDecoderSide"]
        assert testConfig.hiddenLayers[i].activationEncoderSide == configExpectation["HiddenLayer_"+str(i)]["activationEncoderSide"]
    


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
