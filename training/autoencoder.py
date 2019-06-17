from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers, losses, callbacks
from keras.utils import plot_model

#from training.trainUtils import correntropyLoss

class Autoencoder:
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
    """
    def __init__(self, identifier, inputDim, encoderDim, hiddenLayerDim = [], weightDecay=False, robust=False, encoderActivation="tanh", decoderActivation="tanh"):
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

            
        self.name = "%s"%identifier
        self.name += "_wDecay" if self.useWeightDecay else ""
        self.name += "_robust" if self.robustAutoencoder else ""

    def buildNetwork(self, plot=False):
        """ Building the network """
        kernelRegulizer = regularizers.l2(self.weightDecayLambda) if self.useWeightDecay else regularizers.l1(0.)

        inputLayer = Input(shape=(self.inputDimention,))

        hiddenEncoderLayers = {}
        for iLayer in range(self.nHidden):
            hiddenEncoderLayers[iLayer] =  Dense(self.hiddenLayerDimention[iLayer],
                                                 activation=self.hiddenLayerEncoderActivation[iLayer],
                                                 kernel_regularizer=kernelRegulizer,
                                                 name = "hiddenEncoderLayer_"+str(iLayer))
        
        encoderLayer = Dense(self.encoderDimention,
                             activation=self.encoderActivation,
                             kernel_regularizer=kernelRegulizer,
                             name = "encoderLayer")

       
        hiddenDecoderLayers = {}
        for iLayer in range(self.nHidden):
            hiddenDecoderLayers[iLayer] = Dense(self.hiddenLayerDimention[::-1][iLayer],
                                                 activation=self.hiddenLayerDecoderActivation[iLayer],
                                                 kernel_regularizer=kernelRegulizer,
                                                 name = "hiddenDecoderLayer_"+str(iLayer))

            
        decoderLayer = Dense(self.reconstructionDimention,
                             activation=self.decoderActivation,
                             kernel_regularizer=kernelRegulizer,
                             name = "decoderLayer")


        #Build DeepNetwork
        if self.nHidden > 0:
            encoder = hiddenEncoderLayers[0](inputLayer)
            for layer in hiddenEncoderLayers.keys():
                if layer == 0:
                    continue
                encoder = hiddenEncoderLayers[layer](encoder)
            encoder = encoderLayer(encoder)
            decoder = hiddenDecoderLayers[0](encoder)
            for layer in hiddenDecoderLayers.keys():
                if layer == 0:
                    continue
                decoder = hiddenEncoderLayers[layer](decoder)
            decoder = decoderLayer(decoder)
        #Build shallow network
        else:
            encoder = encoderLayer(inputLayer)
            decoder = decoderLayer(encoder)

        self.autoencoder = Model(inputLayer, decoder)

        self.encoder = Model(inputLayer, encoder)
        
        encodedInput = Input(shape=(self.encoderDimention,))
        # print(self.autoencoder.layers)
        if self.nHidden > 0:
            decoderLayer_ = self.autoencoder.layers[-2](encodedInput)
            decoderLayer_ = self.autoencoder.layers[-1](decoderLayer_)
            self.decoder = Model(encodedInput, decoderLayer_)
        else:
            decoderLayer_ = self.autoencoder.layers[-1](encodedInput)
            self.decoder = Model(encodedInput, decoderLayer_)
        
        if plot:
            plot_model(self.autoencoder, to_file="autoencoder_model.png", show_shapes=True)
            plot_model(self.encoder, to_file="encoder_model.png", show_shapes=True)
            plot_model(self.decoder, to_file="decoder_model.png", show_shapes=True) 
            
        
        return True
