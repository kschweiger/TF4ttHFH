"""
Test for the plotting

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import configparser
from itertools import permutations
import argparse

import numpy as np
import pandas as pd
import copy

import plotting.checkInputData
import plotting.compareTransfromedVariables
from plotting.style import StyleConfig
import plotting.plotUtils 
import pytest

doPlots = False #Set to True if you want to see the test plots

def getNormalDataframe(names, means, stddev, size):
    data = {"Index" :  np.arange(size)}
    for iname, name in enumerate(names):
        data[name] = np.random.normal(means[iname], stddev[iname], size)

    df = pd.DataFrame(data)
    df.set_index(["Index"], inplace=True)
    return df

@pytest.fixture(scope="module")
def testDataOne(means = (10, 12, 7), stddev = (2, 1.5, 1)):
    size = 10000
    names = []
    for i in range(len(means)):
        names.append("Var"+str(i))
    return getNormalDataframe(names, means, stddev, size)

@pytest.fixture(scope="module")
def testDataTwo(means = (10.5, 10, 7.1), stddev = (2.3, 2, 0.7)):
    size = 10000
    names = []
    for i in range(len(means)):
        names.append("Var"+str(i))
    return getNormalDataframe(names, means, stddev, size)

@pytest.fixture(scope="module")
def testDataThree(means = (1, 1, 1), stddev = (1, 1, 1)):
    size = 100
    names = []
    for i in range(len(means)):
        names.append("Var"+str(i))
        names.append("Bar"+str(i))
        names.append("Vbr"+str(i))
    return getNormalDataframe(names, 3*means, 3*stddev, size)



@pytest.fixture(scope="module")
def mockExpectationConfig():
    config = configparser.ConfigParser()
    expectation = {}
    expectation["General"] = {"defaultnBins":"20",
                              "defaultRangeMin": "0",
                              "defaultRangeMax" : "20"}
    expectation["Var1"] = {"nBins":"20",
                           "binRangeMin" : "5",
                           "binRangeMax" : "15",
                           "axisName" : "axisVar1"}
    
    config['General'] = expectation["General"]
    config['Var1'] = expectation["Var1"]
    return expectation, config

@pytest.mark.parametrize("whitelist, blacklist", [(["All"],[]),
                                                  (["All"],["Var1"]),
                                                  (["Var1","Var2"],["Var1"])])
def test_generateVariableList(testDataOne, whitelist, blacklist):
    if whitelist == ["All"]:
        expectedColumns = list(testDataOne.columns)
    else:
        expectedColumns = whitelist
    for c in blacklist:
        if c in expectedColumns:
            expectedColumns.remove(c)
        
    assert plotting.checkInputData.generateVariableList(testDataOne, whitelist, blacklist) == expectedColumns

def test_generateVariableList_expections(testDataOne):
    with pytest.raises(KeyError):
        plotting.checkInputData.generateVariableList(testDataOne, ["Var0", "someRandomName"], [])


def test_generateVariableList_blackListWildcards_atStart(testDataThree):
    expectedColumns = []
    for c in testDataThree.columns:
        if not c.endswith("0"):
            expectedColumns.append(c)
           
    assert plotting.checkInputData.generateVariableList(testDataThree, ["All"], ["*0"]) == expectedColumns

def test_generateVariableList_blackListWildcards_atEnd(testDataThree):
    expectedColumns = []
    for c in testDataThree.columns:
        if not c.startswith("Bar"):
            expectedColumns.append(c)
           
    assert plotting.checkInputData.generateVariableList(testDataThree, ["All"], ["Bar*"]) == expectedColumns

def test_generateVariableList_blackListWildcards_in(testDataThree):
    expectedColumns = []
    for c in testDataThree.columns:
        if not "ar" in c:
            expectedColumns.append(c)
           
    assert plotting.checkInputData.generateVariableList(testDataThree, ["All"], ["*ar*"]) == expectedColumns

def test_generateVariableList_blackListWildcards_exceptions(testDataThree):
    with pytest.raises(RuntimeError): 
        assert plotting.checkInputData.generateVariableList(testDataThree, ["All"], ["0*0"])
    with pytest.raises(RuntimeError): 
        assert plotting.checkInputData.generateVariableList(testDataThree, ["All"], ["*0*0*"])
    with pytest.raises(RuntimeError): 
        assert plotting.checkInputData.generateVariableList(testDataThree, ["All"], ["*0*0"])

        
def test_plotDataframeVars(mocker, testDataOne, testDataTwo):
    mocker.patch("plotting.checkInputData.getWeights", side_effect=[pd.DataFrame(np.array((testDataOne.shape[0])*[1.0])),
                                                                    pd.DataFrame(np.array((testDataTwo.shape[0])*[1.0]))])
    assert plotting.checkInputData.plotDataframeVars(
        [testDataOne, testDataTwo],
        "output1d",
        "Var0",
        ["DF1","DF2"],
        nBins = 20,
        binRange = (0, 20),
        varAxisName = "Variable 0",
        savePDF = doPlots
    )

def test_plotCorrelation(mocker, testDataOne):
    mocker.patch("plotting.checkInputData.getWeights", return_value=pd.DataFrame(np.array((testDataOne.shape[0])*[1.0])))
    assert plotting.checkInputData.plotCorrelation(
        testDataOne,
        "output2d",
        "Var0",
        20,
        (0, 20),
        "Variable 0",
        "Var1",
        20,
        (0, 20),
        "Variable 1",
        savePDF=doPlots
    )
        
def test_style_init(mocker, mockExpectationConfig):
    mockExpectation, mockConfig = mockExpectationConfig
    mocker.patch.object(StyleConfig, "readConfig", return_value = mockConfig)
    testConfig = StyleConfig(path = "/path/to/config.cfg")

    emptyKey = "someKey"
    defaultStyle = testConfig.style[emptyKey]
    assert (defaultStyle.nBins == int(mockExpectation["General"]["defaultnBins"]) and
            defaultStyle.binRange[0] == float(mockExpectation["General"]["defaultRangeMin"]) and
            defaultStyle.binRange[1] == float(mockExpectation["General"]["defaultRangeMax"]) and
            defaultStyle.axisName == emptyKey)

    for key in mockExpectation:
        if key == "General":
            continue
        setStyle = testConfig.style[key]
        assert (setStyle.nBins == int(mockExpectation[key]["nBins"]) and
                setStyle.binRange[0] == float(mockExpectation[key]["binRangeMin"]) and
                setStyle.binRange[1] == float(mockExpectation[key]["binRangeMax"]) and
                setStyle.axisName == mockExpectation[key]["axisName"])

    
def test_getCorrelation(mocker, mockExpectationConfig, testDataOne):
    mockExpectation, mockConfig = mockExpectationConfig
    mocker.patch.object(StyleConfig, "readConfig", return_value = mockConfig)
    style = StyleConfig(path = "/path/to/config.cfg")


    mocker.patch("plotting.checkInputData.plotCorrelation", return_value=True)
    mocker.spy(plotting.checkInputData, "plotCorrelation")
    
    vars2Test = ["Var2", "Var1", "Var0"]
    plotting.checkInputData.getCorrealtions(style, testDataOne, "outputBase", vars2Test)
    
    assert len(list(permutations(vars2Test, 2))) == plotting.checkInputData.plotCorrelation.call_count


def test_getDistributions(mocker, mockExpectationConfig, testDataOne, testDataTwo):
    mockExpectation, mockConfig = mockExpectationConfig
    mocker.patch.object(StyleConfig, "readConfig", return_value = mockConfig)
    style = StyleConfig(path = "/path/to/config.cfg")
    mocker.patch("plotting.checkInputData.plotDataframeVars", return_value=True)
    mocker.spy(plotting.checkInputData, "plotDataframeVars")

    vars2Test = ["Var2", "Var1", "Var0"]
    plotting.checkInputData.getDistributions(style, [testDataOne, testDataTwo], "outputBase", vars2Test)

    assert len(vars2Test) == plotting.checkInputData.plotDataframeVars.call_count


def test_checkInputData_parseArgs(mocker):
    args = plotting.checkInputData.parseArgs(["--input","someFile1","someFile2","--output","some/path/to/output+Prefix"])
    assert isinstance(args, argparse.Namespace)
    
def test_checkInputData_main(mocker, mockExpectationConfig, testDataOne, testDataTwo):
    mockExpectation, mockConfig = mockExpectationConfig
    mocker.patch.object(StyleConfig, "readConfig", return_value = mockConfig)
    mocker.patch("plotting.checkInputData.getDataframes", return_value=[testDataOne, testDataTwo])
    style = StyleConfig(path = "/path/to/config.cfg")
    args = plotting.checkInputData.parseArgs(["--input","someFile1","someFile2",
                                              "--output","some/path/to/output+Prefix",
                                              "--plotVars", "Var1", "Var2",
                                              "--plotCorr",
                                              "--plotDist"])

    
    mocker.patch("plotting.checkInputData.getDistributions", return_value=True)
    mocker.spy(plotting.checkInputData, "getDistributions")
    mocker.patch("plotting.checkInputData.getCorrealtions", return_value=True)
    mocker.spy(plotting.checkInputData, "getCorrealtions")
    plotting.checkInputData.process(args, style)

    assert plotting.checkInputData.getCorrealtions.call_count == 2
    assert plotting.checkInputData.getDistributions.call_count == 1

def test_checkInputData_getWeights():
    size = 10
    data = {"evt" :  np.arange(size),
            "run" :  np.arange(size),
            "lumi" :  np.arange(size),
            "puWeight" : np.array(size*[1.0]),
            "genWeight" : np.array(size*[2.0]),
            "btagWeight_shape" : np.array(size*[3.0]),
            "weight_CRCorr" : np.array(size*[4.0]),
            "triggerWeight" : np.array(size*[5.0])
    }
    expected = 1.0 * 2.0 * 3.0 * 4.0 * 5.0
    df = pd.DataFrame(data)
    df.set_index(["evt","run","lumi"], inplace=True)
    weights = plotting.checkInputData.getWeights(df)

    for elem in weights.values:
        assert elem == expected

def test_checkInputData_transformDataframe(testDataOne):
    transformVars =  ["Var0", "Var1"]
    transformedDataOne = plotting.checkInputData.transformDataframe(testDataOne, transformVars)

    for var in list(testDataOne.columns):
        print("Checking var", var)
        if var in transformVars:
            assert (transformedDataOne[var] != testDataOne[var]).all()
            assert np.isclose([transformedDataOne[var].mean()], [0])
            assert np.isclose([transformedDataOne[var].std()], [1])
        else:
            assert (transformedDataOne[var] == testDataOne[var]).all()
            assert np.isclose([transformedDataOne[var].mean()],
                              [testDataOne[var].mean()])
            assert np.isclose([transformedDataOne[var].std()],
                              [testDataOne[var].std()])
    

def test_checkInputData_mergeInputs(mocker, testDataOne, testDataTwo):
    args = plotting.checkInputData.parseArgs(["--input","someFile1","mergeName",
                                              "--output","some/path/to/output+Prefix",
                                              "--merge", "someFile2Merge1", "someFile2Merge2",
                                              "--nGen", "1000", "2000",
                                              "--xsec", "0.2", "0.4"])

    lumi = 1.0
    mocker.patch("plotting.checkInputData.getDataframes", return_value=[testDataOne.copy(), testDataTwo.copy()])
    oneVar = testDataOne.columns[0]
    name, mergedDataset, weightedData = plotting.checkInputData.mergeDatasets(args, list(testDataOne.columns), lumi=lumi)

    assert name == "Name"
    assert testDataOne[oneVar][0] * (lumi*1000*args.xsec[0]/args.nGen[0]) == weightedData[0][oneVar][0]
    mergedTestDataset = (testDataOne * (lumi*1000*args.xsec[0]/args.nGen[0])).append(testDataTwo * (lumi*1000*args.xsec[1]/args.nGen[1]))

    for var in mergedTestDataset.columns:
        assert (mergedDataset[var] == mergedTestDataset[var]).all()
    
    
def test_plotUtils_xyPlot():
    data1 = [1,3,6,14,18,20]
    yData1 = np.array(data1)
    xData1 = np.array(range(len(data1)))

    data2 = [2,5,7,11,17,22]
    yData2 = np.array(data2)
    xData2 = np.array(range(len(data2)))

    passData = [(xData1, yData1), (xData2, yData2)]
    
    assert plotting.plotUtils.make1DPlot(passData, "1dTest", "xName", "yName", ["Data1","Data2"], savePDF=True)

def test_compareTransformedVars_transfrom():
    data = np.array([range(4),range(4,8),range(8,12)])
    transformation = {"mu" : { "Var0" : 1,
                               "Var1" : 2,
                               "Var2" : 3,
                               "Var3" : 4},
                      "std" : {"Var0" : 3,
                               "Var1" : 2,
                               "Var2" : 4,
                               "Var3" : 1}}
    stdArray = np.array([3,2,4,1])
    muArray = np.array([1,2,3,4])
    transfromedData = plotting.compareTransfromedVariables.transform("Gauss",
                                                                     data,
                                                                     ["Var0","Var1","Var2","Var3"],
                                                                     transformation)
    tansfromExp = (data * stdArray)+muArray
    assert (transfromedData == tansfromExp).all()
