import inspect
import logging
import json
import os

import numpy as np

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import regularizers, losses, callbacks, optimizers, metrics, initializers
from keras.utils import plot_model, print_summary
from tensorflow import Session, device
from keras.callbacks import EarlyStopping, ModelCheckpoint

from training.trainUtils import r_square
from training.networkBase import NetWork

import plotting.plotUtils

class Autoencoder(NetWork):
    """
    Class for setting up and building the autoencoder (AE) model. This supports shallow AEs (if hiddenLayerDim is apassed as empty list) and deep AEs by passing 
    a list of dientions for hidden layers. The argument encoderDim is the dimention of the center layer of the network. The systematic for deep AE is the following:
    Passsing a hiddenLayerDim a list with one int, will lead to one hidden encoder and one hidden decoder layer with the passed dimention. Graphically this means that 
    a hidden player is inserted between input and the encoder layer in the encoder and between the encoder and the reconstruction layer in the decoder. 
    Passing a longer list will always increase encoder and decoder. Therefore this is a implementation of a symmetric autoencoder. Furthermore it supports different 
    activation function per layer by passing lists to the encoderActivation and decoderActivation arguments. 
    The activation functions (if passed as list) will be used in the following order: 
    - 1st element will be used for 1st hidden layer encoder and last element will be the activation of the encoder layer (in-between layer will be set accordingly)
    - 1st element will be used for 1st hidden layer after the encoder layer and last element will be used in reconstruction layer

    Args:
      identifier (str) : Name of the Autoencoder
      inputDim (int) : Dimention of the input dataset
      encoderDim (int) : Dimention of the encoder layer (center of symmetric network)
      hiddenLayerDim (list) : Dimention of addition hidden layers. Will be added symmetrically
      weightDecay (bool) : If True, weight decay will be used for regularization
      robust (bool) : If True, the corrEntropy will be used as loss function
      encoderActivation/decoderActivation (str or list) : Activation function for all layer or specific layers
      loss (str) : Internal name for a loss function. Check supportedLosses attribute
      metric (list) : List of metrics
    """
    def __init__(self, identifier, inputDim, encoderDim, hiddenLayerDim = [], weightDecay=False, robust=False,
                 encoderActivation="tanh", decoderActivation="tanh", loss = "MSE", metric = None, batchSize = 128):
        ################### Exceptions ##########################
        if len(hiddenLayerDim) == 0 and not (isinstance(encoderActivation, str) and isinstance(decoderActivation, str)):
            raise TypeError("From hidderLayerDim this is supposed to be a shallow AE. Only str type activation are supported.")
        if type(encoderActivation) != type(decoderActivation):
            raise TypeError("Passed encoder- and decoderactivation are required to be of same type."
                            "Encoder type : %s | decoder type : %s"%(type(encoderActivation),type(decoderActivation)))
        if isinstance(encoderActivation, list) and isinstance(decoderActivation, list):
            if len(encoderActivation) != len(decoderActivation):
                raise RuntimeError("Only symmetric AE supported. encoder- and decoderactivation have different lens")
            else:
                if len(encoderActivation) > len(hiddenLayerDim)+1:
                    raise RuntimeError("Passed more activations than layer. Expected 1 + %s (len(hiddenLayerDim)) but got %s"%(len(hiddenLayerDim),
                                                                                                                              len(encoderActivation)))
        #########################################################
        super().__init__()
        self.inputDimention = inputDim
        self.reconstructionDimention = self.inputDimention
        self.encoderDimention = encoderDim

        self.nHidden = 0
        self.hiddenLayerDimention = []
        for layerDim in hiddenLayerDim:
            self.nHidden += 1
            self.hiddenLayerDimention.append(layerDim)

        
            
        self.useWeightDecay = weightDecay
        self.weightDecayLambda = 0.02 # Change "by hand" if needed
        self.robustAutoencoder = robust

        self.hiddenLayerEncoderActivation = []
        self.hiddenLayerDecoderActivation = []

        self.passedEncoderActivation = encoderActivation
        self.passedDecoderActivation = decoderActivation
        
        if isinstance(encoderActivation, str) and isinstance(decoderActivation, str):
            self.encoderActivation = encoderActivation
            self.decoderActivation = decoderActivation
            for layerDim in hiddenLayerDim:
                self.hiddenLayerEncoderActivation.append(encoderActivation)
                self.hiddenLayerDecoderActivation.append(decoderActivation)
        else:
            self.encoderActivation = encoderActivation[-1]
            self.decoderActivation = decoderActivation[-1]
            for i in range(self.nHidden):
                self.hiddenLayerEncoderActivation.append(encoderActivation[i])
                self.hiddenLayerDecoderActivation.append(decoderActivation[i])

        self.supportedLosses = ["MSE", "LOGCOSH", "MEA", "MSLE"]
        self.loss = loss
        self.lossFunction = self._getLossInstanceFromName(loss)
        self.metrics = metric
        self.optimizer = None
        self.batchSize = batchSize

        self.autoencoder = None     
        
        self.name = "%s"%identifier
        self.name += "_wDecay" if self.useWeightDecay else ""
        self.name += "_robust" if self.robustAutoencoder else ""

    def _getLossInstanceFromName(self, lossName):
        """ 
        Helper function for getting the keras loss instance from a name
        Technically not necessary since compile takes strings but this enables custom
        loss funcitons to be treated to the same porcedure.
        """
        if lossName not in self.supportedLosses:
            raise NameError("Loss %s not supported. Supperted losses are %s"%(lossName, self.supportedLosses))

        if lossName == "MSE":
            return losses.mean_squared_error
        if lossName == "LOGCOSH":
            return losses.logcosh
        if lossName == "MAE":
            return losses.mean_absolute_error
        if lossName == "MSLE":
            return losses.mean_squared_logarithmic_error
       
    def buildModel(self, plot=False):
        """ Building the network """
        kernelRegulizer = regularizers.l2(self.weightDecayLambda) if self.useWeightDecay else regularizers.l1(0.)
        
        inputLayer = Input(shape=(self.inputDimention,))

        hiddenEncoderLayers = {}
        layersSet = 0
        for iLayer in range(self.nHidden):
            hiddenEncoderLayers[iLayer] =  Dense(self.hiddenLayerDimention[iLayer],
                                                 kernel_initializer=self.LayerInitializerKernel,
                                                 bias_initializer=self.LayerInitializerBias,
                                                 activation=self.hiddenLayerEncoderActivation[iLayer],
                                                 kernel_regularizer=kernelRegulizer,
                                                 name = "hiddenEncoderLayer_"+str(iLayer))
            logging.info("Setting hidden encoder layer %s with", layersSet)
            logging.info("  Dimention %s | Activation %s",
                         self.hiddenLayerDimention[iLayer],
                         self.hiddenLayerEncoderActivation[iLayer])
            logging.info("  regualizer %s | name %s", kernelRegulizer,"hiddenEncoderLayer_"+str(iLayer))
            layersSet += 1
        encoderLayer = Dense(self.encoderDimention,
                             kernel_initializer=self.LayerInitializerKernel,
                             bias_initializer=self.LayerInitializerBias,                    
                             activation=self.encoderActivation,
                             kernel_regularizer=kernelRegulizer,
                             name = "encoderLayer")
        logging.info("Setting encoder layer %s with", layersSet)
        logging.info("  Dimention %s | Activation %s",
                     self.encoderDimention,
                     self.encoderActivation)
        logging.info("  regualizer %s | name %s", kernelRegulizer,"encoderLayer")
        layersSet += 1
        hiddenDecoderLayers = {}
        for iLayer in range(self.nHidden):
            hiddenDecoderLayers[iLayer] = Dense(self.hiddenLayerDimention[::-1][iLayer],
                                                kernel_initializer=self.LayerInitializerKernel,
                                                bias_initializer=self.LayerInitializerBias,
                                                activation=self.hiddenLayerDecoderActivation[iLayer],
                                                kernel_regularizer=kernelRegulizer,
                                                name = "hiddenDecoderLayer_"+str(iLayer))
            logging.info("Setting hidden decoder layer %s with", layersSet)
            logging.info("  Dimention %s | Activation %s",
                         self.hiddenLayerDimention[::-1][iLayer],
                         self.hiddenLayerDecoderActivation[iLayer])
            logging.info("  regualizer %s | name %s", kernelRegulizer,"hiddenDecoderLayer_"+str(iLayer))
            layersSet += 1
        decoderLayer = Dense(self.reconstructionDimention,
                             kernel_initializer=self.LayerInitializerKernel,
                             bias_initializer=self.LayerInitializerBias,
                             activation=self.decoderActivation,
                             kernel_regularizer=kernelRegulizer,
                             name = "decoderLayer")
        logging.info("Setting decoder layer %s with", layersSet)
        logging.info("  Dimention %s | Activation %s",
                     self.reconstructionDimention,
                     self.decoderActivation)
        logging.info("  regualizer %s | name %s", kernelRegulizer,"decoderLayer")
        layersSet += 1
        
        #Build DeepNetwork
        if self.nHidden > 0:
            encoder = hiddenEncoderLayers[0](inputLayer)
            logging.debug("Added encoder layer %s", encoder)
            for layer in hiddenEncoderLayers.keys():
                if layer == 0:
                    continue
                encoder = hiddenEncoderLayers[layer](encoder)
                logging.debug("Added encoder layer %s", encoder)
            encoder = encoderLayer(encoder)
            logging.debug("Added bottlenck layer %s", encoder)
            decoder = hiddenDecoderLayers[0](encoder)
            logging.debug("Added decoder layer %s", decoder)
            for layer in hiddenDecoderLayers.keys():
                if layer == 0:
                    continue
                decoder = hiddenDecoderLayers[layer](decoder)
                logging.debug("Added decoder layer %s", decoder)
            decoder = decoderLayer(decoder)
            logging.debug("Added final decoder layer %s", decoder)
        #Build shallow network
        else:
            encoder = encoderLayer(inputLayer)
            decoder = decoderLayer(encoder)

        logging.debug("Setting autoencoder")
        self.autoencoder = Model(inputLayer, decoder)
        
        logging.debug("Setting encoder")
        self.encoder = Model(inputLayer, encoder)
        
        encodedInput = Input(shape=(self.encoderDimention,))
        # print(self.autoencoder.layers)
        if self.nHidden > 0:
            logging.debug("Setting decoder")
            inputSet = False
            for i in list(range(1,self.nHidden+2))[::-1]:
                if not inputSet:
                    logging.debug("Adding layer %s - %s as decoder input", -i, self.autoencoder.layers[-i])
                    decoderLayer_ = self.autoencoder.layers[-i](encodedInput)
                    inputSet = True
                    logging.debug("Set decoderLayer_ %s", decoderLayer_)
                else:
                    decoderLayer_ = self.autoencoder.layers[-i](decoderLayer_)
                    logging.debug("Set decoderLayer_ %s", decoderLayer_)
            self.decoder = Model(encodedInput, decoderLayer_)
        else:
            decoderLayer_ = self.autoencoder.layers[-1](encodedInput)
            self.decoder = Model(encodedInput, decoderLayer_)
        
        if plot:
            plot_model(self.autoencoder, to_file="autoencoder_model.png", show_shapes=True)
            plot_model(self.encoder, to_file="encoder_model.png", show_shapes=True)
            plot_model(self.decoder, to_file="decoder_model.png", show_shapes=True) 
            

        self.modelBuilt = True
        return True

    def compileModel(self, writeyml=False, outdir="."):
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set")
        if not self.modelBuilt:
            raise RuntimeError("Model not built")

        logging.info("Will compile model with")
        logging.info("  Optimiizer: %s", self.optimizer)
        logging.info("  Loss function: %s", self.lossFunction)
        logging.info("  Metrics: %s", self.metrics)
        print(r_square)        
        self.autoencoder.compile(
            optimizer=self.optimizer,
            #optimizer="adam",
            #loss="mse",
            loss=self.lossFunction,
            metrics=self.metrics
        )

        self.modelCompiled = True
        
        if writeyml:
            ymlModel = self.autoencoder.to_yaml()
            with open("{0}/{1}_model_summary.yml".format(outdir, self.name), "w") as f:
                f.write(ymlModel)

        return True

    def trainModel(self, trainingData, trainingWeights, outputFolder, epochs=100, valSplit=0.25, thisDevice="/device:CPU:0", earlyStopping=False, patience=0):
        """ Train model with setting set by attributes and argements """
        if not self.modelCompiled:
            raise RuntimeError("Model not compiled")
        if not isinstance(trainingData, np.ndarray):
            raise TypeError("trainingdata should be np.ndarray but is %s"%type(trainingData))
        if not isinstance(trainingWeights, np.ndarray):
            raise TypeError("trainingdata should be np.ndarray but is %s"%type(trainingWeights))
        logging.info("Starting training of the autoencoder %s", self.autoencoder)
        logging.warning("Will start training on %s", thisDevice)
        
        allCallbacks = []
        #checkpoint = ModelCheckpoint("{0}/best_model.h5py".format(outputFolder), monitor='val_loss', mode='min', verbose=1)
        #allCallbacks.append(checkpoint)
        if earlyStopping:
            logging.warning("Adding early stopping by validation loss")
            logging.debug("Variable paramters: Patience : %s", patience)
            earlyStoppingLoss = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, restore_best_weights=True)
            allCallbacks.append(earlyStoppingLoss)        	
        if not allCallbacks:
            allCallbacks = None
            
        self.trainedModel = self.autoencoder.fit(trainingData, trainingData,
                                                 batch_size = self.batchSize,
                                                 epochs = epochs,
                                                 shuffle = True,                                                     
                                                 validation_split = valSplit,
                                                 sample_weight = trainingWeights,
                                                 callbacks = allCallbacks)

        #logging.debug("Loading best model: %s/best_model.h5py", outputFolder)
        #self.autoencoder = load_model("{0}/best_model.h5py".format(outputFolder))
        #logging.debug("Removing file for best model: %s/best_model.h5py", outputFolder)
        #os.remove("{0}/best_model.h5py".format(outputFolder))
        self.modelTrained = True
        
        return True

    def evalModel(self, testData, testWeights, variables, outputFolder, plotPrediction=False, plotMetics=False, splitNetwork=True, saveData=False, plotPostFix=""):
        """ Evaluate trained model """
        if not self.modelTrained:
            raise RuntimeError("Model not yet trainede")

        self.modelEvaluation = self.autoencoder.evaluate(testData, testData)

        #print(self.modelEvaluation)
        if splitNetwork:
            predictEncoder = self.encoder.predict(testData)
            predictDecoder = self.decoder.predict(predictEncoder)
            logging.info("Mean activations: %s",predictEncoder.mean())
        else:
            predictDecoder = self.autoencoder.predict(testData)
        # print("Input test data")
        # print(testData)
        # print("Prediction encoder layer")
        # print(predictEncoder)
        # print("Prediction decoder layer")
        # print(predictDecoder)

        for iVar, var in enumerate(variables):
            logging.info("Mean %s - input %s | prediction %s",var, testData[:,iVar].mean(), predictDecoder[:,iVar].mean())
            logging.info("Std %s - input %s | prediction %s",var, testData[:,iVar].std(), predictDecoder[:,iVar].std())

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

        
        if plotPrediction:
            thisLegend = ["Test Input", "Decoder prediction"]
            if plotPostFix != "":
                thisLegend = [x+" ("+plotPostFix.replace("_","")+")" for x in thisLegend]
            for iVar, var in enumerate(variables):
                plotting.plotUtils.make1DHistoPlot([testData[:,iVar], predictDecoder[:,iVar]],
                                                   [testWeights, testWeights],
                                                   outputFolder+"/"+self.name+"_EvalOutput_"+var+plotPostFix,
                                                   nBins = 60,
                                                   binRange = (-10, 10),
                                                   varAxisName = var,
                                                   legendEntries = thisLegend)
        return predictDecoder
    
    def saveModel(self, outputFolder, transfromations=None):
        """ Function for saving the model and additional information """
        fileNameModel = "trainedModel.h5py"
        logging.info("Saving model at %s/%s", outputFolder, fileNameModel)
        self.autoencoder.save("%s/%s"%(outputFolder, fileNameModel))

        fileNameWeights = "trainedModel_weights.h5"
        logging.info("Saving model weights at %s/%s", outputFolder, fileNameWeights)
        self.autoencoder.save_weights("%s/%s"%(outputFolder, fileNameWeights))

        infos = self.getInfoDict()
        print(infos)
        fileNameJSON = "autoencoder_attributes.json"
        logging.info("Saveing class attributes in json file %s", fileNameJSON)
        with open("%s/%s"%(outputFolder, fileNameJSON), "w") as f:
            json.dump(infos, f, indent = 2, separators = (",", ": "))

        fileNameReport = "autoencoder_report.txt"
        logging.info("Saving summary to %s/%s", outputFolder, fileNameReport)
        with open("%s/%s"%(outputFolder, fileNameReport),'w') as fh:
            self.autoencoder.summary(print_fn=lambda x: fh.write(x + '\n'))
    
        if transfromations is not None:
            fileNameTransfromations = "auoencoder_inputTransformation.json"
            logging.info("Saving transofmration factors for inputvariables at: %s/%s",outputFolder, fileNameTransfromations)
            with open("%s/%s"%(outputFolder, fileNameTransfromations), "w") as f:
                json.dump(transfromations, f, indent=2,  separators=(",", ": "))

    def loadModel(self, inputFolder):
        """ Loads a model created with the class """
        self.autoencoder = load_model("{0}/trainedModel.h5py".format(inputFolder))
        self.modelTrained = True

    def _getAutoencoderPrediction(self, inputData):
        return self.autoencoder.predict(inputData)
    
    def getReconstructionErr(self, inputData, evalMetric=None):
        """ Prdiction of autoencoder for given input dataset. Returns the reconstruction error """
        if not isinstance(inputData, np.ndarray):
            raise TypeError("Passed inputData is expected to be np.ndarray but is %s"%type(inputData))
        if not self.modelTrained:
            raise RuntimeError("No model trained or loaded")
        if evalMetric is None: # If None is passed use loss function fefined in setup
            evalMetric = self.loss    

        reconstruction = self._getAutoencoderPrediction(inputData)
        
        metricFunction = getattr(metrics, evalMetric)

        reconstructionError = Session().run(metricFunction(inputData, reconstruction))

        return evalMetric, reconstructionError
                
   
