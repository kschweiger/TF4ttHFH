"""
Top level training script for autoencoder
"""
import os
import logging

from collections import namedtuple

from utils.ConfigReader import ConfigReaderBase

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
        self.SuffleSeed = self.setOptionWithDefault("General", "SuffleSeed", None)

        self.selection = None if self.selection == "None" else self.selection
        self.SuffleSeed = None if self.SuffleSeed == "None" else self.SuffleSeed

        netTuple = namedtuple("netTuple", ["defaultActivationEncoder", "defaultActivationDecoder",
                                           "useWeightDecay", "robustAutoencoder", "name",
                                           "hiddenLayers", "inputDimention"])
        
        self.net = netTuple(defaultActivationEncoder = self.readConfig.get("NeuralNet", "defaultActivationEncoder"),
                            defaultActivationDecoder = self.readConfig.get("NeuralNet", "defaultActivationDecoder"),
                            useWeightDecay = self.setOptionWithDefault("NeuralNet", "useWeightDecay", False, "bool"),
                            robustAutoencoder = self.setOptionWithDefault("NeuralNet", "robustAutoencoder", False, "bool"),
                            name = self.readConfig.get("NeuralNet", "name"),
                            hiddenLayers = self.setOptionWithDefault("NeuralNet", "hiddenLayers", 0, "int"),
                            inputDimention = self.readConfig.getint("NeuralNet", "inputDimention"))

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
        
        self.trainSamples = {}
        sampleTuple = namedtuple("sampleTuple", ["input", "label", "xsec", "nGen", "datatype"])
        
        for sample in self.samples:
            if not self.readConfig.has_section(sample):
                raise KeyError("Sample %s not defined in config (as section) only defined in General.samples"%sample)
            self.trainSamples[sample] = sampleTuple(input =  self.readConfig.get(sample, "input"),
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
        
            
    def setOptionWithDefault(self, section, option, default, getterType="str"):
        if self.readConfig.has_option(section, option):
            if getterType == "float":
                return self.readConfig.getfloat(section, option)
            elif getterType == "int":
                return self.readConfig.getint(section, option)
            elif getterType == "bool":
                return self.readConfig.getboolean(section, option)
            else:
                return self.readConfig.get(section, option)
        else:
            return default
        
class AutoencoderTrainer:
    """
    Trainer for a autoencoder
    """
    def __init__(self, testData, trainData, autoencoder):
        self.autoencoder = autoencoder

        self.trainData = trainData
        self.testData = testData


    def train(self, optimizer, loss, epochs):
        pass


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
    pass
