"""
Test for the classes used for the autoencoder setup

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import configparser

 
import uproot as root
import pandas as pd
import numpy as np
import copy
import tensorflow as tf

from training.autoencoder import Autoencoder

import pytest

@pytest.mark.parametrize( "nameExp, weightDecay, robust", [("ShallowAutoEncoder", False, False),
                                                           ("ShallowAutoEncoder_wDecay_robust", True, True)])
def test_autoencoder_init_shallow(nameExp, weightDecay, robust):
    autoencoder = Autoencoder("ShallowAutoEncoder", 20, 5, encoderActivation="tanh", decoderActivation="tanh", weightDecay=weightDecay, robust=robust)

    assert autoencoder.name == nameExp

    assert autoencoder.inputDimention == 20
    assert autoencoder.reconstructionDimention == 20
    assert autoencoder.encoderDimention == 5

    assert autoencoder.nHidden == 0
    assert autoencoder.hiddenLayerDimention == []

    assert autoencoder.encoderActivation == "tanh"
    assert autoencoder.decoderActivation == "tanh"

@pytest.mark.parametrize("encoderActivation, decoderActivation", [("singleActivationEncoder", "singleActivationdecoder"),
                                                                  (["hiddenLayer1ActivationEncoder", "inputLayerActivationEncoder"],
                                                                   ["hiddenLayer1ActivationDecoder", "reconstructionLayerActivationDecoder"])]) 
def test_autoencoder_init_deep(encoderActivation, decoderActivation):
    autoencoder = Autoencoder("DeepAutoEnocder", 30, 5, hiddenLayerDim = [20], encoderActivation=encoderActivation, decoderActivation=decoderActivation)

    assert autoencoder.nHidden == 1

    if isinstance(encoderActivation, str) and isinstance(decoderActivation, str):
        encoderActivation = (1+autoencoder.nHidden)*[encoderActivation]
        decoderActivation = (1+autoencoder.nHidden)*[decoderActivation]
    assert autoencoder.encoderActivation == encoderActivation[-1]
    assert autoencoder.decoderActivation == decoderActivation[-1]

    for i in range(autoencoder.nHidden):
        assert autoencoder.hiddenLayerEncoderActivation[i] == encoderActivation[i]
        assert autoencoder.hiddenLayerDecoderActivation[i] == decoderActivation[i]

def test_autoencoder_init_exceptions():
    #If hiddenLayerDim = [] is passed encoderActivation and decoderAtivation are required to be string
    with pytest.raises(TypeError):
        autoencoder = Autoencoder("AutoEnocder", 30, 5, hiddenLayerDim = [], encoderActivation=["Act1","Act2"], decoderActivation=["Act1","Act2"])
    #If passed as list activation can not be longer than nHidden+1
    with pytest.raises(RuntimeError):
        autoencoder = Autoencoder("AutoEnocder", 30, 5, hiddenLayerDim = [20], encoderActivation=["Act1","Act2","Act3"], decoderActivation=["Act1","Act2","Act3"])
    #Both encoderActivation and decoderActivation need to be same type and if list same lenghts
    with pytest.raises(TypeError):
        autoencoder = Autoencoder("AutoEnocder", 30, 5, hiddenLayerDim = [20], encoderActivation="Act", decoderActivation=["Act1","Act2"])
    with pytest.raises(TypeError):
        autoencoder = Autoencoder("AutoEnocder", 30, 5, hiddenLayerDim = [20], encoderActivation=["Act1","Act2"], decoderActivation="Act")
    with pytest.raises(RuntimeError):
        autoencoder = Autoencoder("AutoEnocder", 30, 5, hiddenLayerDim = [20], encoderActivation=["Act1","Act2","Atc3"], decoderActivation=["Act1","Act2"])

def test_audtoencoder_build_deep():
    autoencoder = Autoencoder("DeepAutoEncoder", 30, 5, hiddenLayerDim = [20])

    #Basically just test if the setup does not crash ;)
    assert autoencoder.buildModel(plot=False)

def test_audtoencoder_build_shallow():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    #Basically just test if the setup does not crash ;)
    assert autoencoder.buildModel(plot=False)

def test_autoencoder_getLoss():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    lossFunct = autoencoder._getLossInstanceFromName("MSE")

    assert callable(lossFunct)

def test_autoencoder_getLoss():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    with pytest.raises(NameError):
        autoencoder._getLossInstanceFromName("UnsupportedName")

def test_autoencoder_setOptimizer():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    autoencoder.setOptimizer(optimizerName="rmsprop")

    assert autoencoder.optimizer is not None

    autoencoder.optimizer = None
    autoencoder.setOptimizer(optimizerName="adagrad")

    assert autoencoder.optimizer is not None


def test_autoencoder_setOptimizer_wArg():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    autoencoder.setOptimizer(optimizerName="rmsprop", rho=0.8)

    assert autoencoder.optimizer is not None
    # should also test if value is set correclty but it's pain sine it is a tf.Variable

    autoencoder.optimizer = None
    autoencoder.setOptimizer(optimizerName="adagrad", decay=0.99)

    assert autoencoder.optimizer is not None

def test_autoencoder_setOptimizer_wArgMulti():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    autoencoder.setOptimizer(optimizerName="rmsprop", rho=0.8, decay=0.99)

    assert autoencoder.optimizer is not None

    
def test_autoencoder_setOptimizer_exceptions():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    with pytest.raises(NotImplementedError):
        autoencoder.setOptimizer(optimizerName="blubb")
    
    with pytest.raises(RuntimeError):
        autoencoder.setOptimizer(optimizerName="rmsprop", notaArgument=0.8)


def test_autoencoder_compileModel_exceptions():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)
    with pytest.raises(RuntimeError):
        autoencoder.compileModel()
    autoencoder.setOptimizer(optimizerName="rmsprop")
    with pytest.raises(RuntimeError):
        autoencoder.compileModel()
        
def test_autoencoder_compileModel():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)
    autoencoder.setOptimizer(optimizerName="rmsprop")
    autoencoder.buildModel()
    assert autoencoder.compileModel(writeyml=True)
    
def test_autoencoder_fitModel_exceptions():
    testDataArray = np.ndarray(shape=(20,5))
    testWeightDataArray = np.ndarray(shape=(20,1))
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)
    autoencoder.setOptimizer(optimizerName="rmsprop")
    autoencoder.buildModel()
    with pytest.raises(RuntimeError):
        autoencoder.trainModel(trainingData=testDataArray, trainingWeights=testWeightDataArray, outputFolder="SomeFolder")
    autoencoder.compileModel()
    with pytest.raises(TypeError):
        autoencoder.trainModel(trainingData = None, trainingWeights=testWeightDataArray, outputFolder="SomeFolder")
    with pytest.raises(TypeError):
        autoencoder.trainModel(trainingData = testDataArray,  trainingWeights=None, outputFolder="SomeFolder")

def test_autoencoder_evalModel_exceptions():
    testDataArray = np.ndarray(shape=(20,5))
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)
    with pytest.raises(RuntimeError):
        autoencoder.evalModel(testData=testDataArray, testWeights=None, variables=[], outputFolder="some/Folder")

def test_autoencoder_getInfoDict():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    info = autoencoder.getInfoDict()

    assert isinstance(info, dict)
    assert info.keys() != []

def test_autoencoder_getReconstrionError_exception():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    print(isinstance(np.ndarray(shape=(5,1)), np.ndarray), type(np.ndarray(shape=(5,1))))
    
    inputData = np.ndarray(shape=(5,1))
    
    with pytest.raises(TypeError):
        autoencoder.getReconstructionErr(None)

    with pytest.raises(RuntimeError):
        autoencoder.getReconstructionErr(inputData)

def test_autoencoder_getReconstrionError(mocker):
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)
    inputDataList = [(1.0,2.0,3.0,4.0),(2.0,7.0,4.0,5.0), (3.0,10.0,2.0,8.0)]
    #inputDataList = [(2.0,7.0,4.0,5.0)]
    inputData = np.array(inputDataList, dtype=float)
    predictionDataList = [(2.0,1.0,1.0,8.0), (3.0,6.0,4.0,7.0), (7.0,2.0,4.0,7.0)]
    #predictionDataList = [(3.0,6.0,4.0,7.0)]

    expectedErr = []
    for i in range(len(inputDataList)):
        squaredSum = 0
        print(predictionDataList[i],inputDataList[i])
        for j in range(len(inputDataList[i])):
            print(predictionDataList[i][j],inputDataList[i][j])
            squaredSum += (predictionDataList[i][j] - inputDataList[i][j])**2
        expectedErr.append(squaredSum/float(len(inputDataList[i])))
        
    print(expectedErr)
    
    mocker.patch.object(Autoencoder, "_getAutoencoderPrediction",
                        return_value=np.array(predictionDataList, dtype=float))

    
    autoencoder.modelTrained = True

    metric, err = autoencoder.getReconstructionErr(inputData)


    print(err)
    
    assert metric == autoencoder.loss
    assert isinstance(err, np.ndarray)

    for iVal, val in enumerate(expectedErr):
        assert val == err[iVal]
