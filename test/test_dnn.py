"""
Test for the classes used for the dnn setup

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

from training.DNN import DNN
import pytest


def test_dnn_init():
    dnn = DNN("Name", 10, [40])

    assert dnn.name == "Name"

    assert dnn.inputDimention == 10
    assert dnn.nLayer == 1
    assert isinstance(dnn.layerDimention, list)
    assert dnn.layerDimention[0] == 40

    assert dnn.activation == "relu"
    assert dnn.outputActivation == "softmax"

def test_dnn_init_exceptions():
    with pytest.raises(TypeError):
        DNN("Name", 10, "Blubb")
    
    with pytest.raises(RuntimeError):
        DNN("Name", 10, [])
        
def test_dnn_buildModel():
    dnn = DNN("Name", 10, [40])

    assert dnn.buildModel(plot=True)

def test_dnn_compileModel_exceptions():
    dnn = DNN("Name", 10, [40])
    with pytest.raises(RuntimeError):
        dnn.compileModel()
    dnn.optimizer = "adagrad"
    with pytest.raises(RuntimeError):
        dnn.compileModel()

def test_dnn_compileModel():
    dnn = DNN("Name", 10, [40])
    dnn.optimizer = "adagrad"
    dnn.buildModel()
    assert dnn.compileModel(writeyml=True)
