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

from convertTree import Config, convertTree

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
                              "outputVariables": "variable1,varibale2,variable3"}
    expectation["Sample"] = {"name" : "SampleName",
                             "path" : "data/testfiles.txt",
                             "selection" : "variable1 == 2 and variable2 > 6"}
    config['General'] = expectation["General"]
    config["Sample"] = expectation["Sample"]

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
    assert testConfig.outputVariables == list(set(mockExpectation["General"]["outputVariables"].split(",") + ["Varinable3"]))
    assert testConfig.files == ["fil1.root","file2.root"]
    assert testConfig.sampleSelection == mockExpectation["Sample"]["selection"]    
