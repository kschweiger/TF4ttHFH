"""
Test for various util funcitons and classes

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import configparser

 
import uproot as root
import pandas as pd
import copy

from utils.ConfigReader import ConfigReaderBase
from preprocessing.utils import initLogging

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
