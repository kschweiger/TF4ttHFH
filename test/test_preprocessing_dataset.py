"""
Test for the classes used in the preprocessing scripts

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import uproot as root
import pandas as pd
import copy

from preprocessing.dataset import Dataset

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
    branches = ["branch1", "branch2", "branch3"]
    treedict = {}
    for br in branches:
        treedict[br.encode("UTF-8")] = 1

    mockedTree = tree(treedict)
    return mockedTree

 
def test_Dataset_init():
    newDataset = Dataset("someName")

    assert isinstance(newDataset, Dataset)
    assert newDataset.outputName == "someName"


def test_Dataset_getBranchesFromFile(mockTree, mocker):
    newDataset = Dataset("someName")

    def openROOTFile(inputfile):
        return {newDataset.treeName : mockTree}
    
    mocker.patch("uproot.open", new=openROOTFile)
    
    branches = newDataset.getBranchesFromFile("someFile.root")

    expectedBranches = [br.decode(newDataset.encoding) for br in mockTree.keys()]
    
    assert branches == expectedBranches

def test_Dataset_addFiles(mockTree, mocker):
    newDataset = Dataset("someName")
    def openROOTFile(inputfile):
        return {newDataset.treeName : mockTree}
    
    mocker.patch("uproot.open", new=openROOTFile)

    newDataset.addFiles(["someFile.root"])

    assert newDataset.filesAdded and newDataset.files == ["someFile.root"]
    
    newDataset.addFiles(["someOtherFile.root"])

    assert newDataset.files == ["someFile.root", "someOtherFile.root"]

def test_Dataset_addFiles_exception(mockTree, mocker):
    newDataset = Dataset("someName")
    
    def openROOTFile(inputfile):
        return {newDataset.treeName : mockTree}
    
    mocker.patch("uproot.open", new=openROOTFile)

    newDataset.filesAdded = True
    newDataset.branches = ["branch5"]
    
    with pytest.raises(RuntimeError):
        newDataset.addFiles(["someOtherFile.root"])        


def test_Dataset_setOutputBranches(mockTree):
    newDataset = Dataset("someName")    
    inBrList = [br.decode(newDataset.encoding) for br in mockTree.keys()]
    newDataset.filesAdded = True
    newDataset.branches = inBrList
    
    expectedBranches = inBrList[0:1]

    newDataset.setOutputBranches(expectedBranches)

    assert expectedBranches == newDataset.outputBranches and newDataset.outputBranchesSet
    
def test_Dataset_setOutputBranches_exception(mockTree):
    newDataset = Dataset("someName")

    inBrList = [br.decode(newDataset.encoding) for br in mockTree.keys()]
    expectedBranches = inBrList[0:1]
    
    with pytest.raises(RuntimeError):
        newDataset.setOutputBranches(expectedBranches)

    newDataset.filesAdded = True
    newDataset.branches = inBrList
        
    with pytest.raises(KeyError):
        newDataset.setOutputBranches(["noPresentBranch"])


@pytest.mark.parametrize("sampleSel, sel", [("branch1 > 2",""),
                                            ("","branch2 < 8"),
                                            ("branch1 > 2","branch2 < 8"),
                                            ("","")])
def test_Dataset_getSelectedDataframe(sampleSel, sel, mockTree, mocker):
    newDataset = Dataset("someName")
    
    newDataset.sampleSelection = sampleSel
    newDataset.selection = sel
    
    dataframe = mockTree.pandas.df()
    print(dataframe)
    if sampleSel != "":
        dataframe = dataframe.query(sampleSel)
    if sel != "":
        dataframe = dataframe.query(sel)
    print(dataframe)
    selectedDF = newDataset.getSelectedDataframe(mockTree)

    assert selectedDF.equals(dataframe)


def test_Dataset_process(mockTree, mocker):
    newDataset = Dataset("someName")

    mockTree_1 = copy.deepcopy(mockTree)
    mockTree_2 = copy.deepcopy(mockTree)

    mockTree_1.dataframe.update(pd.DataFrame({'branch2': list(range(2,12))[::-1]}))
    mockTree_1.setDF()
    mockTree_2.dataframe.update(pd.DataFrame({'branch1': (list(range(0,10)))[::-1]}))
    mockTree_2.setDF()

    newDataset.filesAdded = True
    newDataset.files = ["file1.root", "file2.root"]
    newDataset.branches = ["branch1","branch2","branch3"]
    
    newDataset.outputBranchesSet = True
    newDataset.outputBranches = ["branch1", "branch3"]

    def openROOTFile(*args, **kwargs):
        mm = mocker.MagicMock()
        inputfile = args[0]
        if inputfile == "file1.root":
            mm.__enter__ = mocker.Mock(return_value =
                                       {newDataset.treeName : copy.deepcopy(mockTree_1)}
            )
        else:
            mm.__enter__ = mocker.Mock(return_value =
                                       {newDataset.treeName : copy.deepcopy(mockTree_2)}
            )
        return mm
    
    m = mocker.MagicMock() #This mocker, mocks the open call
    m.side_effect = openROOTFile #Returns a mocker to deal with the with statement
    mocker.patch("uproot.open", m, create=True)
    
    newDataset.selection = "branch1 >= 7 and branch2 >=2"

    mockTree_1_df = mockTree_1.dataframe
    mockTree_2_df = mockTree_2.dataframe
    mockTree_1_df = mockTree_1_df.query("branch1 >= 7 and branch2 >=2")
    mockTree_2_df = mockTree_2_df.query("branch1 >= 7 and branch2 >=2")
    expected = pd.concat([mockTree_1_df, mockTree_2_df])
    expected.drop(columns=["branch2"], inplace=True)

    outputDF = newDataset.process(skipOutput = True)
    
    assert outputDF.equals(expected)
