import inspect
import logging
import json
import os

import numpy as np


from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import regularizers, losses, callbacks, optimizers, metrics, initializers
from keras.utils import plot_model, print_summary, to_categorical
from tensorflow import Session, device
from keras.callbacks import EarlyStopping, ModelCheckpoint

from scipy.integrate import simps

from utils.utils import reduceArray, getSigBkgArrays
from training.trainUtils import r_square
from training.networkBase import NetWork
import plotting.plotUtils

class DNN(NetWork):
    """
    Class for setting up and building a neural net for classification. Models wit hat leas on layer are supported.
    """
    def __init__(self, identifier, inputDim, layerDims, weightDecay=False, activation="relu",
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
        self.weightDecayLambda = 1e-5 # Change "by hand" if needed

        if self.nLayer == 0:
            raise RuntimeError("At least one layer must be defines")
        
        self.loss = loss

        self.activation = activation
        self.outputActivation = outputActivation
        self.metrics = metric
        self.optimizer = None
        self.batchSize = batchSize
        
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
            earlyStoppingLoss = EarlyStopping(monitor='val_loss', verbose=1, patience=patience, restore_best_weights=True)
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
        
        return True

    def evalModel(self, testData, testWeights, testLabels,
                  trainData, trainWeights, trainLabels,
                  variables, outputFolder, plotMetics=False, saveData=False, plotPostFix="", addROCMetrics = []):
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

        
        self.modelEvaluation = self.network.evaluate(testData, testLabels)

        print(self.modelEvaluation)

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
            reduction = 10
            AUCMetrics = {}
            ROCMetrics = {}
            ROCMetrics["DNN"] = roc_curve(testLabels, preditionTest[:,0], sample_weight=testWeights)
            #AUC seems not to work because it thinks the rates are not monotonous
            fpr ,tpr ,_ = ROCMetrics["DNN"]
            AUCMetrics["DNN"] = np.trapz(tpr, fpr)
            for metric, values in addROCMetrics:
                logging.info("Making ROC curve for %s", metric)
                ROCMetrics[metric] = roc_curve(testLabels, values, sample_weight=testWeights)
                fpr ,tpr ,_ = ROCMetrics[metric]
                AUCMetrics[metric] = np.trapz(tpr, fpr)

                if AUCMetrics[metric] < 0.5:
                    logging.warning("Inverting ROC for %s", metric)
                    invLabels = []
                    for label in testLabels:
                        invLabels.append(0 if label == 1 else 1)
                    invLabels = np.array(invLabels)
                    ROCMetrics[metric] = roc_curve(invLabels, values, sample_weight=testWeights)
                    fpr ,tpr ,_ = ROCMetrics[metric]
                    AUCMetrics[metric] = np.trapz(tpr, fpr)
            plotVals = []
            legendEntries = []
            for key in ROCMetrics:
                fpr, tpr, _ = ROCMetrics[key]
                plotVals.append((fpr, tpr))
                    
                legendEntries.append("{0} - AUC = {1:.2f}".format(key, AUCMetrics[key]))
                
            plotting.plotUtils.make1DPlot(plotVals,
                                          output = outputFolder+"/"+self.name+"_ROC"+plotPostFix,
                                          xAxisName = "False Postive Rate",
                                          yAxisName = "True Postive Rate",
                                          legendEntries = legendEntries)

            plotting.plotUtils.make1DHistoPlot(getSigBkgArrays(testLabels, preditionTest[:,0]),
                                               getSigBkgArrays(testLabels, testWeights),
                                               output = outputFolder+"/"+self.name+"_TestBvS"+plotPostFix,
                                               varAxisName = "DNN prediction",
                                               legendEntries = ["Background test sample", "Signal test sample"],
                                               nBins = 30,
                                               binRange = (0,1),
                                               normalized = True)

            plotting.plotUtils.make1DHistoPlot(getSigBkgArrays(trainLabels, preditionTrain[:,0]),
                                               getSigBkgArrays(trainLabels, trainWeights),
                                               output = outputFolder+"/"+self.name+"_TrainBvS"+plotPostFix,
                                               varAxisName = "DNN prediction",
                                               legendEntries = ["Background train sample", "Signal train sample"],
                                               nBins = 30,
                                               binRange = (0,1),
                                               normalized = True)
            
            
                
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
