"""
Test for the lookup table generation

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import uproot as root
import pandas as pd
import numpy as np
import copy

import configparser
import argparse

from preprocessing.dataset import Dataset
from createlookup_fromConfig_dnn import Config, parseArgs, mergePredictions

import pytest

@pytest.fixture(scope="module")
def mockExpectationConfig():
    config = configparser.ConfigParser()
    config.optionxform = str
    expectation = {}
    expectation["General"] = {"output": "/path/to/output/",
                              "categories" : "7Jet,8Jet"}
    
    expectation["7Jet"] = {"selection" : "numJets == 7",
                           "model" : "/path/to/model/"}
    expectation["8Jet"] = {"selection" : "numJets == 8",
                           "model" : "/path/to/other/model/"}

    expectation["Dataset1"] = {"input" : "/path/to/filename1.h5"}
    expectation["Dataset2"] = {"input" : "/path/to/filename2.h5"}

    config['General'] = expectation["General"]
    config['7Jet'] = expectation["7Jet"]
    config['8Jet'] = expectation["8Jet"]
    config['Dataset1'] = expectation["Dataset1"]
    config['Dataset2'] = expectation["Dataset2"]
    
    return expectation, config

@pytest.fixture(scope="module")
def data():
    size= 10
    fullsize = 20
    maxsize = 25
    variabe = np.random.normal(2, 0.2, maxsize)
    data1 = {"evt" : list(range(0,size)),
             "run" : size*[1],
             "lumi": size*[1],
             "SomeVarWeDoNotCare" : variabe[0:size],
             "DNNPred" : size*[0.8]}
    data2 = {"evt" : list(range(size,fullsize)),
             "run" : (fullsize-size)*[1],
             "lumi": (fullsize-size)*[1],
             "SomeVarWeDoNotCare" : variabe[size:fullsize],
             "DNNPred" : (fullsize-size)*[0.6]}
    dataFull = {"evt" : list(range(0,maxsize)),
                "run" : maxsize*[1],
                "lumi": maxsize*[1],
                "SomeVarWeDoNotCare" : variabe[0:maxsize]}

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    dfFull = pd.DataFrame(dataFull)

    df1.set_index(["evt","run","lumi"], inplace=True)
    df2.set_index(["evt","run","lumi"], inplace=True)
    dfFull.set_index(["evt","run","lumi"], inplace=True)

    return df1, df2, dfFull

    

def test_config(mocker, mockExpectationConfig):
    configExpectation, mockConfig = mockExpectationConfig        
    mocker.patch.object(Config, "readConfig", return_value = mockConfig)

    testconf = Config("/path/to/some/file.cfg")
    
    expectedCats = configExpectation["General"]["categories"].split(",")
    expectedDS = [x for x in configExpectation.keys() if not (x in expectedCats or x == "General")]

    assert testconf.output == configExpectation["General"]["output"]
    assert testconf.runCategories == expectedCats
    assert testconf.runDatasets == expectedDS

    expectedDatasetInfos = {}
    for DS in expectedDS:
        expectedDatasetInfos[DS] = configExpectation[DS]["input"]

    for key in testconf.datasets.keys():
        assert testconf.datasets[key] == expectedDatasetInfos[key]
        
    expectedCatInfos = {}
    for cat in expectedCats:
        expectedCatInfos[cat] = {"selection" : configExpectation[cat]["selection"],
                                 "model" : configExpectation[cat]["model"]}


    assert testconf.catSettings.keys() == expectedCatInfos.keys()

    for key in testconf.catSettings.keys():
        assert testconf.catSettings[key]["selection"] == expectedCatInfos[key]["selection"]
        assert testconf.catSettings[key]["model"] == expectedCatInfos[key]["model"]
        

def test_parseArgs():
    args = parseArgs(["--config","path/to/config"])
    assert isinstance(args, argparse.Namespace)

def test_mergePrediction(data):
    df1, df2, dfFull = data

    dfFullwDefault = mergePredictions(dfFull, [])

    assert "DNNPred" in list(dfFullwDefault.columns)
    assert list(dfFullwDefault["DNNPred"]) == dfFull.shape[0]*[-1.0]

    dfFullMerged = mergePredictions(dfFull, [df1, df2])

    assert "DNNPred" in list(dfFullMerged.columns)
    assert list(dfFullMerged["DNNPred"]) != dfFull.shape[0]*[-1.0]

    for index, row in df1.iterrows():
        assert dfFullMerged.loc(axis=0)[index]["DNNPred"] == df1.loc(axis=0)[index]["DNNPred"]

    for index, row in df2.iterrows():
        assert dfFullMerged.loc(axis=0)[index]["DNNPred"] == df2.loc(axis=0)[index]["DNNPred"]

    n = 0
    for index, row in dfFullMerged.iterrows():
        if  dfFullMerged.loc(axis=0)[index]["DNNPred"] == -1.0:
            n += 1

    assert n == (dfFullMerged.shape[0]-df1.shape[0]-df2.shape[0])
