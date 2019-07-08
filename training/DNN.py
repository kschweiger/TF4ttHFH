import inspect
import logging
import json
import os

import numpy as np
import pickle

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import regularizers, losses, callbacks, optimizers, metrics, initializers
from keras.utils import plot_model, print_summary, to_categorical
from tensorflow import Session, device
from keras.callbacks import EarlyStopping, ModelCheckpoint

from scipy.integrate import simps

from utils.utils import reduceArray, getSigBkgArrays, getROCs
from training.trainUtils import r_square
from training.networkBase import NetWork
import plotting.plotUtils

class DNN(NetWork):
    """
    Class for setting up and building a neural net for classification. Models wit hat leas on layer are supported.
    """
    def __init__(self, identifier, inputDim, layerDims, weightDecay=False, weightDecayLambda = 1e-5, activation="relu",
                 outputActivation = "softmax", loss = "binary_crossentropy", metric = None, batchSize = 128):
        
        if not isinstance(layerDims, list):
            raise TypeError("layerDims are required to be of type list but is %s",type(layerDims))
        super().__init__()
        self.inputDimention = inputDim
        self.nLayer = 0
        self.layerDimention = []
        for layerDim in layerDims:
            self.nLayer += 1
            self.layerDimention.append(layerDim)
        self.useWeightDecay = weightDecay
        self.weightDecayLambda = weightDecayLambda

        if self.nLayer == 0:
            raise RuntimeError("At least one layer must be defines")
        
        self.loss = loss

        self.activation = activation
        self.outputActivation = outputActivation
        self.metrics = metric
        self.optimizer = None
        self.batchSize = batchSize

        self.earlyStopMonitor = 'val_loss'
        self.earlyStop = False
        self.StopValues = None
        
        self.net = None

        self.isBinary = False
        
        self.name = "%s"%identifier
        self.name += "_wDecay" if self.useWeightDecay else ""


    def buildModel(self, nClasses=2, plot=False):
        """ Building the network """
        logging.debug("Got nClasses = %s", nClasses)
        kernelRegulizer = regularizers.l2(self.weightDecayLambda) if self.useWeightDecay else regularizers.l1(0.)
        
        inputLayer = Input(shape=(self.inputDimention,))

        if nClasses == 1:
            logging.warning("Setting DNN to bianry classification")
            self.isBinary = True

        hiddenLayers = {}
        layersSet = 0
        for iLayer in range(self.nLayer):
            hiddenLayers[iLayer] =  Dense(self.layerDimention[iLayer],
                                          kernel_initializer=self.LayerInitializerKernel,
                                          bias_initializer=self.LayerInitializerBias,
                                          activation=self.activation,
                                          kernel_regularizer=kernelRegulizer,
                                          name = "hiddenLayer_"+str(iLayer))
            logging.info("Setting hidden encoder layer %s with", layersSet)
            logging.info("  Dimention %s | Activation %s",
                         self.layerDimention[iLayer],
                         self.activation)
            logging.info("  regualizer %s | name %s", kernelRegulizer,"hiddenEncoderLayer_"+str(iLayer))
            layersSet += 1



        outputLayer = Dense(nClasses,
                            kernel_initializer=self.LayerInitializerKernel,
                            bias_initializer=self.LayerInitializerBias,
                            activation=self.outputActivation,
                            kernel_regularizer=kernelRegulizer,
                            name = "outputLayer")


        for iLayer in range(self.nLayer):
            if iLayer == 0:
                network = hiddenLayers[iLayer](inputLayer)
            else:
                network = hiddenLayers[iLayer](network)
            logging.debug("Adding Layer %s", hiddenLayers[iLayer])

        network = outputLayer(network)


        logging.debug("Setting network")
        self.network = Model(inputLayer, network)

        if plot:
            plot_model(self.network, to_file="model.png", show_shapes=True)
        
        self.modelBuilt = True
        return True



    def trainModel(self, trainingData, trainingLabels, trainingWeights, outputFolder, epochs=100, valSplit=0.25, earlyStopping=False, patience=0):
        if not self.modelCompiled:
            raise RuntimeError("Model not compiled")
        if not isinstance(trainingData, np.ndarray):
            raise TypeError("trainingdata should be np.ndarray but is %s"%type(trainingData))
        if not isinstance(trainingWeights, np.ndarray):
            raise TypeError("trainingdata should be np.ndarray but is %s"%type(trainingWeights))

        logging.info("Starting training of the network %s", self.network)

        allCallbacks = []
        #checkpoint = ModelCheckpoint("{0}/best_model.h5py".format(outputFolder), monitor='val_loss', mode='min', verbose=1)
        #allCallbacks.append(checkpoint)
        if earlyStopping:
            logging.warning("Adding early stopping by validation loss")
            logging.debug("Variable paramters: Patience : %s", patience)
            
            earlyStoppingLoss = EarlyStopping(monitor=self.earlyStopMonitor, verbose=1, patience=patience, restore_best_weights=True)
            allCallbacks.append(earlyStoppingLoss)        	
        if not allCallbacks:
            allCallbacks = None

        if not self.isBinary:
            logging.info("Converting labels to categorical")
            trainingLabels = to_categorical(trainingLabels)
            
        print(trainingData, trainingLabels)
        self.trainedModel = self.network.fit(trainingData, trainingLabels,
                                             batch_size = self.batchSize,
                                             epochs = epochs,
                                             shuffle = True,                                                     
                                             validation_split = valSplit,
                                             sample_weight = trainingWeights,
                                             callbacks = allCallbacks)

        self.modelTrained = True

        if len(self.trainedModel.__dict__["epoch"]) != epochs:
            self.earlyStop = True
            monitorHistory = self.trainedModel.__dict__["history"][self.earlyStopMonitor]
            minMonitor = min(monitorHistory)
            minEpoch = -1
            for iVal, val in enumerate(monitorHistory):
                if val == minMonitor:
                    minEpoch = iVal+1
                    break
                
            logging.debug("Min at epoch %s with monitor value %s", iVal, minMonitor)
            earlyStopHistory = {}
            for metric in self.trainedModel.__dict__["history"].keys():
                earlyStopHistory[metric] = self.trainedModel.__dict__["history"][metric][minEpoch-1]
            self.StopValues = (minEpoch, earlyStopHistory)

        else:
            StopHistory = {}
            for metric in self.trainedModel.__dict__["history"].keys():
                StopHistory[metric] = self.trainedModel.__dict__["history"][metric][epochs-1]
            self.StopValues = (epochs, StopHistory)
            
        return True

    def evalModel(self, testData, testWeights, testLabels,
                  trainData, trainWeights, trainLabels,
                  variables, outputFolder, classes, 
                  plotMetics=False, saveData=False,
                  plotPostFix="", addROCMetrics = []):
        """ 
        Evaluate trained model. Do not pass categorical lables in case of multiclassification! Will be converted in function
        
        Args:
          testData (np.ndarray) : Test dataset
          testWeights (np.array) : Weight of the test dataset
          testLabels (np.array) : Labels of the test dataset
          trainData (np.ndarray) : Train dataset
          trainWeights (np.array) : Weight of the train dataset
          trainLabels (np.array) : Labels of the train dataset 
          variables (list) : List of input variables
          outputFolder (str) : rel/abs path to output folder 
          plotMetics (bool) : If True loss+metrics will be plotted per epoch
          saveData (bbol) : If True Data will be saved as pickle
          plotPostFix (str) : Postfix for the filenames of the output plots 
          ROCMetrics (list of (str, np.arrays)) : Further metrics to be plotted in the ROC curve plot 
        """
        if not self.modelTrained:
            raise RuntimeError("Model not yet trainede")

        if not self.isBinary:
            logging.info("Converting labels to categorical")
            trainLabels = to_categorical(trainLabels)
            testLabels = to_categorical(testLabels)

        self.modelEvaluationTest = self.network.evaluate(testData, testLabels)

        print(self.modelEvaluationTest)

        self.getInputWeights(variables, outputFolder)

        
        stopEpoch, stopEpochMetrics = self.StopValues
        lines = "Evaluation:\nStopped at epoch {0}\n".format(stopEpoch)
        for iMetric, metric in enumerate(self.network.metrics_names):
            thisLine = "{0} - Train {1:.3f} | Val {2:.3f} | Test {3:.3f} -- Ratio Train {4:.3f} | Ratio Val {5:.3f}".format(
                metric,
                stopEpochMetrics[metric],
                stopEpochMetrics["val_"+metric],
                self.modelEvaluationTest[iMetric],
                stopEpochMetrics[metric]/self.modelEvaluationTest[iMetric],
                stopEpochMetrics["val_"+metric]/self.modelEvaluationTest[iMetric]
            )
            logging.info(thisLine)
            lines += thisLine+"\n"

        with open("{0}/evalMetrics.txt".format(outputFolder), "w") as f:
            f.write(lines)
        
        preditionTest = self.network.predict(testData)
        preditionTrain = self.network.predict(trainData)

        if plotMetics:
            logging.info("Saving epoch metrics")
            metricList = [x for x in self.trainedModel.__dict__["params"]['metrics'] if not x.startswith("val_")]
            for metric in metricList:
                logging.debug("Plotting metric %s", metric)
                metricTrain = np.array(self.trainedModel.__dict__["history"][metric])
                metricVal = np.array(self.trainedModel.__dict__["history"]["val_"+metric])
                xAxis = np.array(self.trainedModel.__dict__["epoch"])
                plotVals = [(xAxis, metricTrain), (xAxis, metricVal)]
                
                plotting.plotUtils.make1DPlot(plotVals,
                                              output = outputFolder+"/"+self.name+"_metrics_"+metric+plotPostFix,
                                              xAxisName = "Epochs",
                                              yAxisName = metric,
                                              legendEntries = ["Training Sample", "Validation Sample"])

        from sklearn.metrics import roc_auc_score, roc_curve, auc
        if not self.isBinary:
            predictedTestClasses = np.argmax( preditionTest, axis = 1)
            predictedTrainClasses = np.argmax( preditionTrain, axis = 1)

            roc_auc_score = roc_auc_score(testLabels, preditionTest)

            logging.info("ROC score: %s", roc_auc_score)
            
        else:
            AUCMetrics = {}
            ROCMetrics = {}
            ROCMetrics["DNN"], AUCMetrics["DNN"] = getROCs(testLabels, preditionTest[:,0], testWeights)
            for metric, values in addROCMetrics:
                logging.info("Making ROC curve for %s", metric)
                ROCMetrics[metric], AUCMetrics[metric] = getROCs(testLabels, values, testWeights)

            plotting.plotUtils.makeROCPlot(ROCMetrics, AUCMetrics,
                                           output = outputFolder+"/"+self.name+"_ROC"+plotPostFix)

            legendTest = []
            legendTrain = []
            for sample in classes:
                legendTest.append("Test sample - {0} (ID: {1})".format(sample, classes[sample]))
                legendTrain.append("Train sample - {0} (ID: {1})".format(sample, classes[sample]))

                
            plotting.plotUtils.make1DHistoPlot(getSigBkgArrays(testLabels, preditionTest[:,0]),
                                               getSigBkgArrays(testLabels, testWeights),
                                               output = outputFolder+"/"+self.name+"_TestBvS"+plotPostFix,
                                               varAxisName = "DNN prediction",
                                               legendEntries = legendTest,
                                               nBins = 30,
                                               binRange = (0,1),
                                               normalized = True)

            plotting.plotUtils.make1DHistoPlot(getSigBkgArrays(trainLabels, preditionTrain[:,0]),
                                               getSigBkgArrays(trainLabels, trainWeights),
                                               output = outputFolder+"/"+self.name+"_TrainBvS"+plotPostFix,
                                               varAxisName = "DNN prediction",
                                               legendEntries = legendTrain,
                                               nBins = 30,
                                               binRange = (0,1),
                                               normalized = True)

        if saveData:
            data2Pickle = {"variable" : variables,
                           "classes" : classes,
                           "trainInputData" : trainData,
                           "trainInputWeight" : trainWeights,
                           "trainInputLabels" : trainLabels if self.isBinary else predictedTrainClasses,
                           "trainPredictionData" : preditionTrain,
                           "testInputData" : testData,
                           "testInputWeight" : testWeights,
                           "testInputLabels" : testLabels if self.isBinary else predictedTestClasses,
                           "testPredictionData" : preditionTest,

            }
            with open("{0}/testDataArrays.pkl".format(outputFolder), "wb") as pickleOut:
                 pickle.dump(data2Pickle, pickleOut)
                 
        return preditionTest, preditionTrain
            
    def saveModel(self, outputFolder, transfromations=None):
        """ Function for saving the model and additional information """
        fileNameModel = "trainedModel.h5py"
        logging.info("Saving model at %s/%s", outputFolder, fileNameModel)
        self.network.save("%s/%s"%(outputFolder, fileNameModel))

        fileNameWeights = "trainedModel_weights.h5"
        logging.info("Saving model weights at %s/%s", outputFolder, fileNameWeights)
        self.network.save_weights("%s/%s"%(outputFolder, fileNameWeights))

        infos = self.getInfoDict()
        print(infos)
        fileNameJSON = "network_attributes.json"
        logging.info("Saveing class attributes in json file %s", fileNameJSON)
        with open("%s/%s"%(outputFolder, fileNameJSON), "w") as f:
            json.dump(infos, f, indent = 2, separators = (",", ": "))

        fileNameReport = "network_report.txt"
        logging.info("Saving summary to %s/%s", outputFolder, fileNameReport)
        with open("%s/%s"%(outputFolder, fileNameReport),'w') as fh:
            self.network.summary(print_fn=lambda x: fh.write(x + '\n'))
    
        if transfromations is not None:
            fileNameTransfromations = "network_inputTransformation.json"
            logging.info("Saving transofmration factors for inputvariables at: %s/%s",outputFolder, fileNameTransfromations)
            with open("%s/%s"%(outputFolder, fileNameTransfromations), "w") as f:
                json.dump(transfromations, f, indent=2,  separators=(",", ": "))

    def loadModel(self, inputFolder):
        """ Loads a model created with the class """
        self.network = load_model("{0}/trainedModel.h5py".format(inputFolder))
        self.modelTrained = True

    def getPrediction(self, inputData):
        return self.network.predict(inputData)

    def getInputWeights(self, variables, outputFolder):

        firstLayer = self.network.layers[1]
        weights = firstLayer.get_weights()[0]

        self.varWeights = {}
        for outWeights, variable in zip(weights, variables):
            wSum = np.sum(np.abs(outWeights))
            self.varWeights[variable] = float(wSum)

        
        fileNameRank = "variableRanking.txt"
        logging.info("Saving ranking at: %s/%s",outputFolder, fileNameRank)
        logging.info("================== Ranking ==================")
        with open("%s/%s"%(outputFolder, fileNameRank), "w") as f:
            for key, val in sorted(self.varWeights.items(), key = lambda item: item[1], reverse=True):
                line = "  {0:>20} : {1:.3f}".format(key, val)
                logging.info(line)
                f.write(line+"\n")
        logging.info("=============================================")
