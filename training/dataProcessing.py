"""
Module for processing routines during training

K. Schweiger, 2019
"""

import logging

from sklearn.utils import shuffle
import pandas as pd
import numpy as np

class Sample:
    """
    Container loading h5 files (form convertTree) and prepares them to be used in the training
    """
    def __init__(self, inFile, label, labelID, xsec=1.0, nGen=1.0, dataType="data"):
        self.inFile = inFile
        self.label = label
        self.labelID = labelID
        self.isData = True if dataType == "data" else False
        if not self.isData:
            self.xsec = xsec
            self.nGen = nGen
        else:
            self.xsec = 1.0
            self.nGen = 1.0

        self.data = None
        self.nEvents = -1

    def loadDataframe(self, selection=None, lumi=1.0):
        """ Function for loading the data from the file and adding combined weight variables to dataframe"""
        logging.info("Loading dartaframe from fole %s", self.inFile)
        df = pd.read_hdf(self.inFile)

        if selection is not None:
            logging.debug("Applying Selection")
            logging.debug("Events before selection: %s", df.shape[0])
            df.query(selection, inplace=True)

        logging.debug("Events selected events: %s", df.shape[0])
        self.nEvents = df.shape[0]

        logging.debug("Setting label id to %s", self.labelID)
        df = df.assign(labelID=lambda x: int(self.labelID))

        ##### Add combined wightes to the dataframe
        # It is expected that all these weights are present in the dataset- even if they are all 1.0
        df = df.assign(eventWeight=lambda x: x.puWeight * x.genWeight * x.btagWeight_shape * x.weight_CRCorr * x.triggerWeight)

        weightSum = sum(df["eventWeight"].values)
        df = df.assign(trainWeight=lambda x: x.eventWeight/weightSum)

        # add lumi weight
        if not self.isData:
            df = df.assign(lumiWeight=lambda x: (1000 * self.xsec * lumi)/self.nGen)
        else:
            df = df.assign(lumiWeight=lambda x: 1.0)
        self.data = df

        return True

    def getLabelTuple(self):
        """ Helper function for getting label, id tuples"""
        return (self.label, self.labelID)

class Data:
    """
    Container for training/testing data. Combination of all samples into singel dataframe

    Args:
      samples (list) : List of Sample object
      trainVariables (list) : List of variables used for the training
      testPercent (float) : Percentage of the passed sample that is used for testing
      selection (string) : Additional selection to be applied to the samples
      shuffleData (bool) : If True the dataset will be suffled before splitting
      shuffleSeed (int) : Random seed for the shuffle. If none is passed and shuffleData is True a random seed will be generated
      lumi (floatQ) : Luminoisty. Will be passed to sample

    Raises:
      ValueError : If testPercent is > 1
      RuntimeError : If an invalid trainig variable is passed
    """
    def __init__(self, samples, trainVariables, testPercent, selection=None, shuffleData=False, shuffleSeed=None, lumi=41.5):
        if testPercent > 1.0:
            raise ValueError("testPercent is required to be less than 1. Passed value = %s"%testPercent)
        trainDataframes = []
        classes = {}
        for sample in samples:
            logging.info("Loading dataframe for sample %s (Label: %s, labelID: %s, isData: %s)", sample, sample.label, sample.labelID, sample.isData)
            sample.loadDataframe(selection=selection, lumi=lumi)
            trainDataframes.append(sample.data)
            labelName, labelID = sample.getLabelTuple()
            classes[labelName] = labelID

        logging.debug("Concat dataframes")
        df = pd.concat(trainDataframes)
        logging.debug("Number of events after concat = %s", df.shape[0])

        del trainDataframes

        #Check if all passed trainVariables are valid
        for var in trainVariables:
            if var not in list(df.columns):
                raise RuntimeError("Passed training variable %s not in df variables"%var)

        self.shuffleData = shuffleData
        self.shuffleSeed = shuffleSeed
        if self.shuffleData and self.shuffleSeed is None:
            self.shuffleSeed = np.random.randint(low=0, high=2**16)

        if self.shuffleData:
            df = shuffle(df, random_state=self.shuffleSeed)

        self.fullDF = df.copy()

        self.nTest = int(df.shape[0] * testPercent)
        self.testDF = df.head(self.nTest)
        self.nTrain = int(df.shape[0] - self.nTest)
        self.trainDF = df.tail(df.shape[0] - self.nTest)
        logging.info("nTest = %s | nTrain = %s", self.nTest, self.nTrain)

        self.trainVariables = trainVariables
        self.allVariables = list(self.fullDF.columns)
        self.outputClasses = classes

    def _getData(self, getTrain=True, asMatrix=True):
        """
        Helper function for getting training or testing data
        """
        requestedDF = self.trainDF if getTrain else self.testDF
        if asMatrix:
            return requestedDF[self.trainVariables].values
        else:
            return requestedDF[self.trainVariables]

    def getTrainData(self, asMatrix=True):
        """ Public wrapper for _getData for training dataset """
        return self._getData(getTrain=True, asMatrix=asMatrix)

    def getTestData(self, asMatrix=True):
        """ Public wrapper for _getData for testing dataset """
        return self._getData(getTrain=False, asMatrix=asMatrix)

    @property
    def trainTrainingWeights(self):
        return self.trainDF["trainWeight"].values

    @property
    def testTrainingWeights(self):
        return self.testDF["trainWeight"].values

    @property
    def trainLumiWeights(self):
        return self.trainDF["lumiWeight"].values

    @property
    def testLumiWeights(self):
        return self.testDF["lumiWeight"].values

    @property
    def trainLables(self):
        return self.trainDF["labelID"].values

    @property
    def testLables(self):
        return self.testDF["labelID"].values
