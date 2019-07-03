"""
Test for the data processing in the training

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

from training.dataProcessing import Sample, Data

import pytest


@pytest.fixture()
def testData(means = (10, 12, 7), stddev = (2, 1.5, 1)):
    size = 10000
    data = {"evt" :  np.arange(size),
            "run" :  np.arange(size),
            "lumi" :  np.arange(size),
            "puWeight" : np.array(size*[1.0]),
            "genWeight" : np.array(size*[1.0]),
            "btagWeight_shape" : np.array(size*[1.0]),
            "weight_CRCorr" : np.array(size*[1.0]),
            "triggerWeight" : np.array(size*[1.0])}

    names = []
    for i in range(len(means)):
        names.append("Var"+str(i))

    for iname, name in enumerate(names):
        data[name] = np.random.normal(means[iname], stddev[iname], size)
    df = pd.DataFrame(data)
    df.set_index(["evt","run","lumi"], inplace=True)
    return df

@pytest.fixture(scope="module")
def testDataSmall(means = (9, 5, 14), stddev = (2, 1.5, 1)):
    size = 200
    data = {"evt" :  np.arange(size),
            "run" :  np.arange(size),
            "lumi" :  np.arange(size),
            "puWeight" : np.array(size*[1.0]),
            "genWeight" : np.array(size*[1.0]),
            "btagWeight_shape" : np.array(size*[1.0]),
            "weight_CRCorr" : np.array(size*[1.0]),
            "triggerWeight" : np.array(size*[1.0])}

    names = []
    for i in range(len(means)):
        names.append("Var"+str(i))

    for iname, name in enumerate(names):
        data[name] = np.random.normal(means[iname], stddev[iname], size)
    df = pd.DataFrame(data)
    df.set_index(["evt","run","lumi"], inplace=True)
    return df



@pytest.mark.parametrize( "label, labelID, datatype", [("DataLabel", 0 ,"data"),
                                                       ("MCLabel", 1, "mc")])
def test_Sample_init(label, labelID, datatype):
    typeExpectation = True if datatype == "data" else False
    testSample = Sample("/path/to/some/file.h5", label, labelID, dataType=datatype)

    assert testSample.inFile == "/path/to/some/file.h5"
    assert testSample.label == label
    assert testSample.labelID == labelID
    assert testSample.isData == typeExpectation
    assert testSample.xsec == 1.0 and testSample.nGen == 1.0

@pytest.mark.parametrize( "datatype,labelID, xsec, nGen, selection", [("data",0 , -1.0, -1.0, None),
                                                                      ("mc", 0, 0.25, 10000, None),
                                                                      ("mc", 1, 0.25, 10000, "Var1 >= 10")])
def test_Sample_loadDataframe(mocker, testData, datatype, labelID, xsec, nGen, selection):
    mocker.patch("pandas.read_hdf", return_value=testData)

    testSample = Sample("/path/to/some/file.h5", label="SomeLabel", labelID=labelID, xsec=xsec, nGen=nGen, dataType=datatype)

    assert testSample.xsec == (1.0 if datatype == "data" else xsec)
    assert testSample.nGen == (1.0 if datatype == "data" else nGen)
    
    assert testSample.loadDataframe(selection, lumi=1.0)

    print("--------------",testSample.data["lumiWeight"][0])
    assert testSample.data["lumiWeight"].values[0] == (1.0 if datatype == "data" else (1000*xsec*1.0)/nGen)

    if selection is None:
        assert testSample.data.shape[0] == testData.shape[0]
    else:
        testDataSel = testData.query(selection)
        assert testSample.data.shape[0] == testDataSel.shape[0]

    print(type(testSample.data["labelID"].values), testSample.data["labelID"].values)
    assert (testSample.data["labelID"].values == int(labelID)).all()

@pytest.mark.parametrize( "label, labelID", [("Label0", 0),
                                             ("Label1", 1)])
def test_Sample_getMethods(label, labelID):
    testSample = Sample("/path/to/some/file.h5", label, labelID, dataType="mc")
    assert testSample.getLabelTuple() == (label, labelID)
    
def test_data_init(mocker, testData, testDataSmall):
    mocker.patch("pandas.read_hdf", side_effect=[testData, testDataSmall])
    unshuffeledDF = pd.concat([testData, testDataSmall])
    
    samples = [ Sample("/path/to/some/file.h5", label="Sample0", labelID=0) ,
                Sample("/path/to/some/file.h5", label="Sample1", labelID=1)]

    testPercentage = 0.2
    data = Data(samples, ["Var0", "Var1", "Var2"], testPercentage, transform=False)

    #Check event count and splitting
    assert data.fullDF.shape[0] == testData.shape[0]+testDataSmall.shape[0]

    assert data.testDF.shape[0] == int(data.fullDF.shape[0]*testPercentage)
    assert data.testDF.shape[0] == data.nTest

    assert data.trainDF.shape[0] == int(data.fullDF.shape[0] - (data.fullDF.shape[0]*testPercentage))
    assert data.trainDF.shape[0] == data.nTrain

    #Check labels
    labels = data.fullDF["labelID"].values
    unique, counts = np.unique(labels, return_counts=True)
    assert counts[0] == testData.shape[0]
    assert counts[1] == testDataSmall.shape[0]

    #Check that shuffle is deactivated
    assert (data.fullDF["Var0"].values == unshuffeledDF["Var0"].values).all()
    
    assert data.trainVariables ==  ["Var0", "Var1", "Var2"]
    assert data.allVariables == list(unshuffeledDF.columns)+["labelID", "weight", "eventWeight", "trainWeight", "lumiWeight"]
    assert data.outputClasses == {"Sample0" : 0 , "Sample1" : 1}

def test_data_init_execption(mocker, testData):
    mocker.patch("pandas.read_hdf", side_effect=[testData, testDataSmall])

    samples = [ Sample("/path/to/some/file.h5", label="Sample0", labelID=0)]

    with pytest.raises(ValueError):
        data = Data(samples, ["Var0" "Var2"], 10000)

    with pytest.raises(RuntimeError):
        data = Data(samples, ["Var0", "SomeRandomVariableName", "Var2"], 0.2)

    
    
def test_data_init_shuffle(mocker, testData, testDataSmall):
    mocker.patch("pandas.read_hdf", side_effect=[testData, testDataSmall])

    samples = [ Sample("/path/to/some/file.h5", label="Sample0", labelID=0) ,
                Sample("/path/to/some/file.h5", label="Sample1", labelID=1)]

    testPercentage = 0.2
    data = Data(samples, ["Var0", "Var1", "Var2"], testPercentage, shuffleData=True, transform=False)

    unshuffeledDF = pd.concat([testData, testDataSmall])

    
    assert not (data.fullDF["Var0"].values == unshuffeledDF["Var0"].values).all()

def test_data_getData(mocker, testData):
    mocker.patch("pandas.read_hdf", side_effect=[testData, testDataSmall])

    samples = [ Sample("/path/to/some/file.h5", label="Sample0", labelID=0)]

    testPercentage = 0.2
    trainVars = ["Var0", "Var1", "Var2"]
    data = Data(samples, trainVars, testPercentage, transform=False)

    retTrainDFMatrix = data._getData(getTrain=True)
    retTrainDF = data._getData(getTrain=True, asMatrix=False)

    assert list(retTrainDF.columns) == trainVars

    assert isinstance(retTrainDFMatrix, np.ndarray)    
    assert isinstance(retTrainDF, pd.DataFrame)

def test_data_getTrainTestData(mocker, testData):
    mocker.patch("pandas.read_hdf", side_effect=[testData, testDataSmall])

    samples = [ Sample("/path/to/some/file.h5", label="Sample0", labelID=0)]

    testPercentage = 0.2
    trainVars = ["Var0", "Var1", "Var2"]
    data = Data(samples, trainVars, testPercentage, transform=False)

    trainDataframe = data.getTrainData(asMatrix=False)
    testDataframe = data.getTestData(asMatrix=False)

    assert trainDataframe.shape[0] == int(testData.shape[0] - int(testData.shape[0]*testPercentage))
    assert testDataframe.shape[0] == int(int(testData.shape[0]*testPercentage))

def test_data_properties(mocker, testData):
    mocker.patch("pandas.read_hdf", side_effect=[testData, testDataSmall])

    samples = [ Sample("/path/to/some/file.h5", label="Sample0", labelID=0)]

    testPercentage = 0.2
    trainVars = ["Var0", "Var1", "Var2"]
    data = Data(samples, trainVars, testPercentage, transform=False)

    #Training set
    assert (data.trainTrainingWeights == data.trainDF["trainWeight"].values).all()
    assert (data.trainLumiWeights == data.trainDF["lumiWeight"].values).all()
    assert (data.trainLabels == data.trainDF["labelID"].values).all()

    #Testing set
    assert (data.testTrainingWeights == data.testDF["trainWeight"].values).all()
    assert (data.testLumiWeights == data.testDF["lumiWeight"].values).all()
    assert (data.testLabels == data.testDF["labelID"].values).all()


def test_data_transfromation_gauss(mocker, testData):
    mocker.patch("pandas.read_hdf", side_effect=[testData, testDataSmall])

    samples = [ Sample("/path/to/some/file.h5", label="Sample0", labelID=0)]

    testPercentage = 0.2
    trainVars = ["Var0", "Var1"]
    data = Data(samples, trainVars, testPercentage, transform=True)

    fullDataset = data.getTestData(asMatrix=False)
    fullDataset = fullDataset.append(data.getTrainData(asMatrix=False))
    for var in trainVars:
        assert (fullDataset[var] != testData[var]).all()
        assert np.isclose([fullDataset[var].mean()], [0])
        assert np.isclose([fullDataset[var].std()], [1])


def test_data_getTrainTestData_reweighting_train(mocker, testData):
    testDatamod = testData.copy()
    testDatamod["puWeight"] =  testDatamod["puWeight"].mul(1.5, axis=0)
    
    mocker.patch("pandas.read_hdf", side_effect=[testDatamod, testDataSmall])
    samples = [Sample("/path/to/some/file.h5", label="Sample0", labelID=0)]

    testPercentage = 0.2
    trainVars = ["Var0", "Var1", "Var2"]
    data = Data(samples, trainVars, 0.0, transform=False)

    trainDataframe = data.getTrainData(asMatrix=False, applyTrainWeight=True)

    for var in ["Var0", "Var1", "Var2"]:
        print(trainDataframe[var])
        print(testData[var])
        assert (trainDataframe[var] != testData[var]).all()
        assert (testData[var].mul(1.5, axis=0) == trainDataframe[var]).all()


def test_data_getTrainTestData_reweighting_Lumi_MC(mocker, testData):
    mocker.patch("pandas.read_hdf", side_effect=[testData, testDataSmall])
    _xsec = 1.0
    _nGen = 2.0
    _lumi = 3.0
    samples = [Sample("/path/to/some/file.h5", label="Sample0", labelID=0, dataType="mc", xsec=_xsec, nGen=_nGen)]

    testPercentage = 0.2
    trainVars = ["Var0", "Var1", "Var2"]
    data = Data(samples, trainVars, 0.0, transform=False, lumi=_lumi)
    
    trainDataframe = data.getTrainData(asMatrix=False, applyLumiWeight=True)
    for var in ["Var0", "Var1", "Var2"]:
        print(trainDataframe[var])
        print(testData[var])
        assert (trainDataframe[var] != testData[var]).all()
        assert (testData[var].mul((1000*_xsec*_lumi)/_nGen, axis=0) == trainDataframe[var]).all()

def test_data_getTrainTestData_reweighting_Lumi_data(mocker, testData):
    mocker.patch("pandas.read_hdf", side_effect=[testData, testDataSmall])
    _xsec = 1.0
    _nGen = 2.0
    _lumi = 3.0
    samples = [Sample("/path/to/some/file.h5", label="Sample0", labelID=0, dataType="data", xsec=_xsec, nGen=_nGen)]

    testPercentage = 0.2
    trainVars = ["Var0", "Var1", "Var2"]
    data = Data(samples, trainVars, 0.0, transform=False, lumi=_lumi)
    
    trainDataframe = data.getTrainData(asMatrix=False, applyLumiWeight=True)
    for var in ["Var0", "Var1", "Var2"]:
        print(trainDataframe[var])
        print(testData[var])
        assert (trainDataframe[var] == testData[var]).all()
