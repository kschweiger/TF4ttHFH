"""
Test for the classes used in the preprocessing scripts

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import configparser

 
import uproot as root
import pandas as pd
import copy

from convertTree import Config, MultiConfig, convertTree

import pytest

class tree(dict):
    def __init__(self, passedDict):
        super().__init__(passedDict)
        self.pandas = SimpleNamespace()

        data = {}
        inBrList = [key.decode("UTF-8") for key in self.keys()]
        for ibr, br in enumerate(inBrList):
            data[br] = list(range(0,10))
            
        dataframe = pd.DataFrame(data)
        dataframe.index.name = "entry"
        self.dataframe = dataframe
        self.setDF()
        
    def setDF(self):
        setattr(self.pandas, "df", lambda : self.dataframe)
        
@pytest.fixture(scope="module")
def mockTree():
    branches = ["variable1", "variable2", "variable3", "variable4", "variable5"]
    treedict = {}
    for br in branches:
        treedict[br.encode("UTF-8")] = 1

    mockedTree = tree(treedict)
    return mockedTree

@pytest.fixture(scope="module")
def mockExpectationConfig(scope="module"):
    config = configparser.ConfigParser()
    expectation = {}
    expectation["General"] = {"outputPrefix": "outPre",
                              "maxEvents": 999999,
                              "outputVariables": "variable1,varibale2,variable3",
                              "categories" : "cat1,cat2"}
    expectation["Sample"] = {"name" : "SampleName",
                             "path" : "data/testfiles.txt",
                             "selection" : "variable1 == 2 and variable2 > 6"}
    expectation["cat1"] = {"selection" : "variable1 >= 3",
                           "name" : "cat1Name"}
    expectation["cat2"] = {"selection" : "variable1 >= 4",
                           "name" : "cat2Name"}
    config['General'] = expectation["General"]
    config["Sample"] = expectation["Sample"]
    config["cat1"] = expectation["cat1"]
    config["cat2"] = expectation["cat2"]
    
    return expectation, config

@pytest.fixture(scope="module")
def mockExpectationConfigMulti(scope="Module"):
    config = configparser.ConfigParser()
    expectation = {}
    expectation["General"] = {"outputPrefix": "outPre",
                              "maxEvents": 999999,
                              "outputVariables": "variable1,varibale2,variable3",
                              "categories" : "cat1,cat2",
                              "samples" : "Sample1,Sample2",
                              "addRatio" : True}
    expectation["Sample"] = {"name" : "SampleName"}
    expectation["Sample1"] = {"name" : "SampleName1",
                              "path" : "data/testfiles1.txt",
                              "selection" : "variable1 == 2 and variable2 > 6",
                              "addSF" : 1.5}
    expectation["Sample2"] = {"name" : "SampleName2",
                              "path" : "data/testfiles2.txt",
                              "selection" : "variable1 == 3 and variable2 > 6"}
    expectation["cat1"] = {"selection" : "variable1 >= 3",
                           "name" : "cat1Name"}
    expectation["cat2"] = {"selection" : "variable1 >= 4",
                           "name" : "cat2Name"}
    config['General'] = expectation["General"]
    config["Sample"] = expectation["Sample"]
    config["Sample1"] = expectation["Sample1"]
    config["Sample2"] = expectation["Sample2"]
    config["cat1"] = expectation["cat1"]
    config["cat2"] = expectation["cat2"]
    
    return expectation, config


def test_config(mocker, mockExpectationConfig):
    mockExpectation, mockConfig = mockExpectationConfig
    mocker.patch.object(Config, "readConfig", return_value = mockConfig)
    mocker.patch('builtins.open', mocker.mock_open(read_data="fil1.root\nfile2.root"))
    testConfig = Config(path = "/path/to/config.cfg",
                        addVars = ["Varinable3"],
                        indexVars = ["Varinable3"],
                        output = "/path/to/output")
    print(testConfig)
    assert isinstance(testConfig, Config)
    assert not testConfig.multiSample
    assert testConfig.outputVariables == list(set(mockExpectation["General"]["outputVariables"].split(",") + ["Varinable3"]))
    assert testConfig.files == ["fil1.root","file2.root"]
    assert testConfig.sampleSelection == mockExpectation["Sample"]["selection"]
    assert testConfig.allCategories == mockExpectation["General"]["categories"].split(",")
    for cat in mockExpectation["General"]["categories"].split(","):
        assert testConfig.categories[cat].selection == mockExpectation[cat]["selection"]
        assert testConfig.categories[cat].name == mockExpectation[cat]["name"]


def test_multiSampleConfig(mocker, mockExpectationConfigMulti):
    mockExpectation, mockConfig = mockExpectationConfigMulti
    mocker.patch.object(MultiConfig, "readConfig", return_value = mockConfig)
    mocker.patch('builtins.open', mocker.mock_open(read_data="fil1.root\nfile2.root"))
    testConfig = MultiConfig(path = "/path/to/config.cfg",
                             addVars = ["Varinable3"],
                             indexVars = ["Varinable3"],
                             output = "/path/to/output")
    
    assert isinstance(testConfig, MultiConfig)
    assert testConfig.outputVariables == list(set(mockExpectation["General"]["outputVariables"].split(",") + ["Varinable3"]))
    inputSamples = [x for x in mockExpectation.keys() if (x.startswith("Sample") and not x == "Sample")]
    assert testConfig.samples == inputSamples
    assert testConfig.multiSample
    assert testConfig.addRatio == mockExpectation["General"]["addRatio"]
    for sample in inputSamples:
        assert sample in testConfig.sampleInfo.keys()
        assert mockExpectation[sample]["name"] == testConfig.sampleInfo[sample].name
        assert mockExpectation[sample]["selection"] == testConfig.sampleInfo[sample].selection
        if "addSF" in mockExpectation[sample].keys():
            assert mockExpectation[sample]["addSF"] == testConfig.sampleInfo[sample].addSF
        else:
            assert 1.0 == testConfig.sampleInfo[sample].addSF
        assert ["fil1.root","file2.root"] == testConfig.sampleInfo[sample].files
    assert testConfig.allCategories == mockExpectation["General"]["categories"].split(",")
    for cat in mockExpectation["General"]["categories"].split(","):
        assert testConfig.categories[cat].selection == mockExpectation[cat]["selection"]
        assert testConfig.categories[cat].name == mockExpectation[cat]["name"]
