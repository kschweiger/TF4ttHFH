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
import copy

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
    assert autoencoder.buildNetwork(plot=False)

def test_audtoencoder_build_shallow():
    autoencoder = Autoencoder("ShallowAutoEncoder", 30, 5)

    #Basically just test if the setup does not crash ;)
    assert autoencoder.buildNetwork(plot=False)
