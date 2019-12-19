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

def test_Dataset_cleanBranchList(mockTree):
    newDataset = Dataset("someName")    
    inBrList = [br.decode(newDataset.encoding) for br in mockTree.keys()]
    newDataset.filesAdded = True
    newDataset.branches = inBrList
    
    expectedBranches = inBrList[0:1]

    newDataset.setOutputBranches(expectedBranches)

    assert expectedBranches == newDataset.outputBranches
    newDataset.cleanBranchList([expectedBranches[0]])
    expectedBranches.remove(expectedBranches[0])
    assert expectedBranches == newDataset.outputBranches
    
def test_Dataset_resolveWildcardBranch_expection():
    newDataset = Dataset("someName") 
    with pytest.raises(RuntimeError):
        newDataset._resolveWildcardBranch("NoStarPassed")
    
    with pytest.raises(NotImplementedError):
        newDataset._resolveWildcardBranch("e*f")
        
    with pytest.raises(NotImplementedError):
        newDataset._resolveWildcardBranch("e*f*")
    
@pytest.mark.parametrize("selector, inbranches, expectBranches", [("AAA*",["aAAA","AAA1","aAAAb"],["AAA1"]),
                                                                  ("*AAA",["aAAA","AAA1","aAAAb"], ["aAAA"]),
                                                                  ("*AAA*",["aAAA","AAA1","aAAAb","BBB"], ["aAAA","AAA1","aAAAb"])])
def test_Dataset_resolveWildcardBranch(selector, inbranches, expectBranches):
    newDataset = Dataset("someName") 
    newDataset.filesAdded = True
    newDataset.branches = inbranches
    
    assert expectBranches == newDataset._resolveWildcardBranch(selector)
    
def test_Dataset_setOutputBranches_wildcard_all(mocker, mockTree):
    newDataset = Dataset("someName")    
    inBrList = [br.decode(newDataset.encoding) for br in mockTree.keys()]
    newDataset.filesAdded = True
    newDataset.branches = inBrList
    
    expectedBranches = inBrList

    mocker.spy(newDataset, "_resolveWildcardBranch")
    
    newDataset.setOutputBranches("*")

    assert newDataset._resolveWildcardBranch.call_count == 0
    
    assert sorted(expectedBranches) == sorted(newDataset.outputBranches) and newDataset.outputBranchesSet

@pytest.mark.parametrize("selector, expectBranches, count", [("*AAA*",["var_AAA_1","var_AAA_2"], 1),
                                                             (["*AAA*","*BBB*"],["var_AAA_1","var_AAA_2","var_BBB_1","var_BBB_2"], 2),
                                                             (["*AAA*","var_CCC_1"],["var_AAA_1","var_AAA_2","var_CCC_1"], 1)])
def test_Dataset_setOutputBranches_wildcard_contains(selector, expectBranches, count, mocker, mockTree):
    newDataset = Dataset("someName")    
    inBrList = ["var_AAA_1","var_AAA_2","var_BBB_1","var_BBB_2","var_CCC_1"]
    newDataset.filesAdded = True
    newDataset.branches = inBrList
    
    expectedBranches = expectBranches

    mocker.spy(newDataset, "_resolveWildcardBranch")

    newDataset.setOutputBranches(selector)

    assert newDataset._resolveWildcardBranch.call_count == count
    
    assert sorted(expectedBranches) == sorted(newDataset.outputBranches) and newDataset.outputBranchesSet

    
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


def test_Dataset_addFlatSFtoDataframe(mockTree, mocker):
    SFName = "flatSF"
    SF = 1.2
    inputDF = copy.deepcopy(mockTree.dataframe)
    expected = copy.deepcopy(mockTree.dataframe)
    expected["flatSF"] = len(expected)*[SF]
    print(expected)

    newDataset = Dataset("someName")
    newDataset.setSF(SF, SFName)
    newDataset.addFlatSFtoDataframe(inputDF)
    assert expected.equals(inputDF)

def test_Dataset_addFlatSFtoDataframe_exceptions(mockTree, mocker):
    newDataset = Dataset("someName")
    inputDF = copy.deepcopy(mockTree.dataframe)
    with pytest.raises(RuntimeError):
        newDataset.addFlatSFtoDataframe(inputDF)
    
    with pytest.raises(TypeError):
        newDataset.setSF("Hallo", "OkayName")
    with pytest.raises(TypeError):
        newDataset.setSF(1.2, [])
    
