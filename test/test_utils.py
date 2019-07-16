"""
Test for various util funcitons and classes

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import configparser
import glob
 
import uproot as root
import pandas as pd
import numpy as np
import copy

from utils.ConfigReader import ConfigReaderBase
from utils.utils import initLogging, reduceArray, getSigBkgArrays
import utils.utils

from utils.makeFileList import main as makeFileList

import pytest

@pytest.fixture(scope="module")
def mockExpectationConfig(scope="module"):
    config = configparser.ConfigParser()
    expectation = {}
    expectation["SECTION"] = {"list": "item1,item2,item3",
                              "MulitlineSingle": "subsection1 : suboption1\nsubsection2 : suboption2\n\n",
                              "MulitlineList": "subsection1 : suboption11,suboption12,suboption13\nsubsection2 : suboption21,suboption22,suboption23"}
    config['SECTION'] = expectation["SECTION"]


    return expectation, config

def test_ConfigReaderBase(mocker, mockExpectationConfig):
    configExpection, mockConfig = mockExpectationConfig
    mocker.patch.object(ConfigReaderBase, "readConfig", return_value = mockConfig)
    
    config = ConfigReaderBase("/path/to/config.cfg")

    mulitLineSingle = config.readMulitlineOption("SECTION", "MulitlineSingle", "Single")
    mulitLineList = config.readMulitlineOption("SECTION", "MulitlineList", "List")

    with pytest.raises(RuntimeError):
        config.readMulitlineOption("SECTION", "MulitlineList", "Blubb")
    
    expectedSingleLine = {}
    for line in configExpection["SECTION"]["MulitlineSingle"].split("\n"):
        if line == "":
            continue
        name, value = line.split(" : ")
        expectedSingleLine[name] = value

    expectedListLine = {}
    for line in configExpection["SECTION"]["MulitlineList"].split("\n"):
        if line == "":
            continue
        name, value = line.split(" : ")
        value = config.getList(value)
        expectedListLine[name] = value

    assert isinstance(config, ConfigReaderBase)
    assert mulitLineSingle == expectedSingleLine
    assert mulitLineList == expectedListLine    

@pytest.mark.parametrize("level", [10,20,30,40,50,0])
def test_logging_setup(level):
    assert initLogging(level)

@pytest.mark.parametrize("inputArray, level, outputArray", [(np.array([1,2,3,4]), 2, np.array([1.5, 3.5])),
                                                            (np.array([1,2,3,4,5]), 2, np.array([1.5, 3.5, 5]))])
def test_reduceArray(inputArray, level, outputArray):
    assert (outputArray == reduceArray(inputArray, level)).all()
    
@pytest.mark.parametrize("inputArray, inputLabels, expectedArray", [(np.array([1,2,3,4]), np.array([0,1,0,1]), (np.array([1, 3]), np.array([2, 4]))),
                                                                     (np.array([1,2,3,4]), np.array([0,2,1,2]), (np.array([1, 3]),np.array([3]), np.array([2, 4])))])
def test_getSigBkgArrays(inputArray, inputLabels, expectedArray):
    retArrays = getSigBkgArrays(inputLabels, inputArray)
    for iarray, array in enumerate(expectedArray):
        (array == retArrays[iarray]).all()


def test_makefileList(mocker):
    m = mocker.mock_open()
    mocker.patch('builtins.open', m)
    mocker.patch.object(utils.utils, "checkNcreateFolder", return_value=True)
    mocker.patch.object(os,"makedirs", return_value = True)
    fileSideEffects = ["folder1/file1.root", "folder1/file2.root"]
    mocker.patch.object(glob, "glob", side_effect=[["folder1"], fileSideEffects])
    makeFileList("/base/path/", "/output/path/")


    callList = [mocker.call.write(file_+"\n") for file_ in fileSideEffects]
    assert m().method_calls == callList
