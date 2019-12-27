"""
Module for processing routines during training

K. Schweiger, 2019
"""

import logging

from keras.utils import to_categorical
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

class Sample:
    """
    Container loading h5 files (form convertTree) and prepares them to be used in the training
    """
    def __init__(self, inFile, label, labelID, xsec=1.0, nGen=1.0, dataType="data", selection=None):
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
        self.selection = selection

    def loadDataframe(self, selection=None, lumi=1.0, normalizedWeight=False, includeGenWeight=False):
        """ Function for loading the data from the file and adding combined weight variables to dataframe"""
        logging.info("Loading dartaframe from fole %s", self.inFile)
        df = pd.read_hdf(self.inFile)

        if self.selection is not None:
            if selection is None:
                selection = self.selection
            else:
                selection = "{0} and {1}".format(selection, self.selection)
        
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
        #logging.warning("Trigger weight disabled")
        # NOTE: Some ugly code here that can be removed once files are generated with a consitant configuration. At this point mainly the data is the problem
        #       Once consitant keep only first if clause.
        #################################################### TEMP CODE ###############################################################
        if "btagDeepFlavWeight_shape" in df:
            if "sampleRatio" in df:
                maxRatioW = 1/max(df["sampleRatio"])
                df = df.assign(weight=lambda x: x.weight_pu * x.genWeight * x.btagDeepFlavWeight_shape * x.weight_CRCorr * x.triggerWeight * maxRatioW * x.sampleRatio)
            else:
                df = df.assign(weight=lambda x: x.weight_pu * x.genWeight * x.btagDeepFlavWeight_shape * x.weight_CRCorr * x.triggerWeight)
        else: # Only in data atm
            logging.error("I have a file with the old btagWeight here. Make sure this is okay")
            if "sampleRatio" in df:
                maxRatioW = 1/max(df["sampleRatio"])
                if "weight_pu" in df:
                    df = df.assign(weight=lambda x: x.weight_pu * x.genWeight * x.btagWeight_shape * x.weight_CRCorr * x.triggerWeight * maxRatioW * x.sampleRatio)
                elif "weightPURecalc" in df: #2016 adn 2017
                    logging.error("Found a file with weightPURecalc. Make sure this is okay")
                    df = df.assign(weight=lambda x: x.weightPURecalc * x.genWeight * x.btagWeight_shape * x.weight_CRCorr * x.triggerWeight * maxRatioW * x.sampleRatio)
                else: # 2018
                    logging.error("Found a file with puWeight. Make sure this is okay")
                    df = df.assign(weight=lambda x: x.puWeight * x.genWeight * x.btagWeight_shape * x.weight_CRCorr * x.triggerWeight * maxRatioW * x.sampleRatio)
            else:
                if "weight_pu" in df:
                    df = df.assign(weight=lambda x: x.weight_pu * x.genWeight * x.btagWeight_shape * x.weight_CRCorr * x.triggerWeight)
                elif "weightPURecalc" in df: #2016 adn 2017
                    logging.error("Found a file with weightPURecalc. Make sure this is okay")
                    df = df.assign(weight=lambda x: x.weightPURecalc * x.genWeight * x.btagWeight_shape * x.weight_CRCorr * x.triggerWeight)
                else: # 2018
                    logging.error("Found a file with puWeight. Make sure this is okay")
                    df = df.assign(weight=lambda x: x.puWeight * x.genWeight * x.btagWeight_shape * x.weight_CRCorr * x.triggerWeight)
        #################################################### TEMP CODE ###############################################################
        if includeGenWeight:
            logging.warning("Will include xsec and nGen in train weight")
            logging.debug("Which are: xsec = %s and nGen = %s", self.xsec, self.nGen )
            df = df.assign(eventWeight=lambda x: x.weight * 1000 * lumi * self.xsec * (1/self.nGen))
        else:
            df = df.assign(eventWeight=lambda x: x.weight)

        #df = df.assign(eventWeightUnNorm=lambda x: x.puWeight * x.genWeight * x.btagWeight_shape * x.weight_CRCorr)# * x.triggerWeight)
        logging.debug("Sample %s weight mean=%s", self.label, df["eventWeight"].mean())
        weightSum = sum(df["eventWeight"].values)
        if normalizedWeight:
            logging.debug("WeightSum=%s", weightSum)
            df = df.assign(trainWeight=lambda x: x.eventWeight/weightSum)
        else:
            df = df.assign(trainWeight=lambda x: x.eventWeight)

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
    def __init__(self, samples, trainVariables, testPercent, transform=True, selection=None, shuffleData=False,
                 shuffleSeed=None, lumi=41.5, normalizedWeight=False, includeGenWeight=False):
        if testPercent > 1.0:
            raise ValueError("testPercent is required to be less than 1. Passed value = %s"%testPercent)
        logging.debug("Using lumi = %s", lumi)
        logging.info("ShuffelData is %s", shuffleData)
        trainDataframes = []
        classes = {}
        for sample in samples:
            logging.info("Loading dataframe for sample %s (Label: %s, labelID: %s, isData: %s)", sample, sample.label, sample.labelID, sample.isData)
            sample.loadDataframe(selection=selection, lumi=lumi, normalizedWeight=normalizedWeight, includeGenWeight=includeGenWeight)
            trainDataframes.append(sample.data)
            labelName, labelID = sample.getLabelTuple()
            classes[labelName] = labelID

        logging.debug("Concat dataframes")
        df = pd.concat(trainDataframes)
        nDF = df.shape[0]
        logging.debug("Number of events after concat = %s", nDF)

        
        df = df[~df.index.duplicated(keep='first')]
        if df.shape[0] != nDF:
            logging.warning("Dropped %s events from dataframe",nDF-df.shape[0])
        
        if normalizedWeight:
            logging.debug("nEvents = %s | sum Weights = %s | sum EventWeight = %s",df.shape[0], sum(df["weight"].values), sum(df["eventWeight"].values))
            df["trainWeight"] = df["trainWeight"]*sum(df["weight"].values)/len(samples)
            logging.debug("Train weight mean= %s, std=%s", df["trainWeight"].mean(), df["trainWeight"].std())
        
        del trainDataframes

        self.trainVariables = trainVariables
        self.outputClasses = classes
        
        #Check if all passed trainVariables are valid
        for var in trainVariables:
            if var not in list(df.columns):
                errMSG = "Passed training variable %s not in df variables"%var
                raise RuntimeError(errMSG)

        self.shuffleData = shuffleData
        self.shuffleSeed = shuffleSeed
        if self.shuffleData and self.shuffleSeed is None:
            self.shuffleSeed = np.random.randint(low=0, high=2**16)

        if self.shuffleData:
            df = shuffle(df, random_state=self.shuffleSeed)

        self.fullDF = df.copy()
        #self.untransfromedDF = df.copy()

        self.nTest = int(self.fullDF.shape[0] * testPercent)
        self.testDF = self.fullDF.head(self.nTest)
        self.nTrain = int(self.fullDF.shape[0] - self.nTest)
        self.trainDF = self.fullDF.tail(df.shape[0] - self.nTest)
        logging.info("nTest = %s | nTrain = %s", self.nTest, self.nTrain)

        self.allVariables = list(self.fullDF.columns)

        self.transfromationMethod = "Gauss"
        self.doTransformation = False
        
        self.transformations = {}
        
        self.transformations["unweighted"] = self.getTransformation()

        self.doTransformation = transform

        
    def getTransformation(self, applyTrainWeight=False, applyLumiWeight=False):
        fullDF = self.getTrainData(asMatrix=False, applyTrainWeight=applyTrainWeight, applyLumiWeight=applyLumiWeight)
        fullDF = fullDF.append(self.getTestData(asMatrix=False, applyTrainWeight=applyTrainWeight, applyLumiWeight=applyLumiWeight))
        retConversion = {}
        if self.transfromationMethod == "Gauss":
            retConversion["mu"] = {}
            retConversion["std"] = {}
            for variable in self.trainVariables:
                retConversion["mu"][variable] = float(fullDF[variable].mean())
                retConversion["std"][variable] = float(fullDF[variable].std())
                logging.debug("Added transformation with mean = %s and std = %s",float(fullDF[variable].mean()), float(fullDF[variable].std()))
        elif method == "Norm":
            raise NotImplementedError("Transforming variables to [0,1] not yet implemented")
        else:
            raise NotImplementedError("Method %s is not supported")
                
        return retConversion


    def applyTransformation(self, dataframe, applyTrainWeight=False, applyLumiWeight=False):
        if not applyTrainWeight and not applyLumiWeight:
            transformation = self.transformations["unweighted"]
        else:
            raise RuntimeError

        logging.debug("Using transfomrats: %s", transformation)
        modDataFram = dataframe.copy()
        if self.transfromationMethod == "Gauss":
            # print("Std transfromation: %s", pd.Series(transformation["std"]))
            # print("mu transfromation: %s", pd.Series(transformation["mu"]))
            # print(dataframe[self.trainVariables])
            modDataFram[self.trainVariables] = ((dataframe[self.trainVariables] - pd.Series(transformation["mu"]))/pd.Series(transformation["std"]))
            #print(modDataFram[self.trainVariables])
        elif method == "Norm":
            raise NotImplementedError("Transforming variables to [0,1] not yet implemented")
        else:
            raise NotImplementedError("Method %s is not supported")
        return modDataFram
        
    def _getData(self, getTrain=True, asMatrix=True, applyTrainWeight=False, applyLumiWeight=False):
        """
        Helper function for getting training or testing data
        """
        if getTrain:
            requestedDF = self.trainDF.copy()
        else:
            requestedDF = self.testDF.copy()
        if applyTrainWeight:
            requestedDF = requestedDF.mul(self.trainTrainingWeights if getTrain else self.testTrainingWeights,
                                          axis=0)
        if applyLumiWeight:
            requestedDF = requestedDF.mul(self.trainLumiWeights if getTrain else self.testLumiWeights,
                                          axis=0)
        if self.doTransformation:
            requestedDF = self.applyTransformation(requestedDF)#, applyTrainWeight=applyTrainWeight, applyLumiWeight=applyLumiWeight)
        if asMatrix:
            return requestedDF[self.trainVariables].values
        else:
            return requestedDF[self.trainVariables]
        
    def getTrainData(self, asMatrix=True, applyTrainWeight=False, applyLumiWeight=False):
        """ Public wrapper for _getData for training dataset """
        return self._getData(getTrain=True,
                             asMatrix=asMatrix,
                             applyTrainWeight=applyTrainWeight,
                             applyLumiWeight=applyLumiWeight)

    def getTestData(self, asMatrix=True, applyTrainWeight=False, applyLumiWeight=False):
        """ Public wrapper for _getData for testing dataset """
        return self._getData(getTrain=False,
                             asMatrix=asMatrix,
                             applyTrainWeight=applyTrainWeight,
                             applyLumiWeight=applyLumiWeight)

    @property
    def trainTrainingWeights(self):
        logging.warning("Mean train training weight: %s",self.trainDF["trainWeight"].mean())
        return self.trainDF["trainWeight"].values

    @property
    def testTrainingWeights(self):
        logging.warning("Mean test training weight: %s",self.testDF["trainWeight"].mean())
        return self.testDF["trainWeight"].values

    @property
    def trainLumiWeights(self):
        return self.trainDF["lumiWeight"].values

    @property
    def testLumiWeights(self):
        return self.testDF["lumiWeight"].values

    #Maybe also needed as dataframes? TBD
    @property
    def trainLabels(self):
        return self.trainDF["labelID"].values

    @property
    def testLabels(self):
        return self.testDF["labelID"].values

    @property
    def trainLabelsCat(self):
        return to_categorical(self.trainDF["labelID"].values)

    @property
    def testLabelsCat(self):
        return to_categorical(self.testDF["labelID"].values)
