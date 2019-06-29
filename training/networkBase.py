"""
Base network class for the Framework

K. Schweiger, 2019

TODO: Rename self.autoencoder in Autoencoder class to self.network. Then compileModel can be moved to the base class
"""

import inspect
import logging

from keras import optimizers

class NetWork:
    """ Base class for networks"""
    def __init__(self):
        self.modelBuilt = False
        self.modelCompiled = False
        self.modelTrained = False

        self.trainedModel = None

        self.LayerInitializerKernel = "random_uniform"
        self.LayerInitializerBias = "zeros"


    def setOptimizer(self, **kwargs):
        """ 
        Function for setting the optimizer. Passing correctly names names arguments will chnage the default setting of the oprtimizer.
        Check in KERAS documentation https://keras.io/optimizers/ for details
        """
        logging.debug("Optimizer to set: %s", kwargs["optimizerName"])
        addkwargs = [k for k in kwargs.keys() if k != "optimizerName"]
        logging.debug("Additional arguments : %s", addkwargs)
        if kwargs["optimizerName"] == "rmsprop":
            optimizerFunc = optimizers.RMSprop
        elif kwargs["optimizerName"] == "adagrad":
            optimizerFunc = optimizers.Adagrad
        elif kwargs["optimizerName"] == "adam":
            optimizerFunc = optimizers.Adam
        else:
            raise NotImplementedError("Optiizer %s not implemented"%kwargs["optimizerName"])
        defaultSig = self._parseSignature(inspect.signature(optimizerFunc))
        useSig = {}
        for arg in defaultSig:
            if arg not in kwargs:
                useSig[arg] = defaultSig[arg]
            else:
                useSig[arg] = kwargs[arg]
        logging.debug("Default signature: %s",defaultSig)
        logging.debug("Will use signature: %s",useSig)
        for arg in addkwargs:
            if arg not in useSig.keys():
                raise RuntimeError("Passed argument %s no valid argument for optimizer %s"%(arg, kwargs["optimizerName"]))

        if kwargs["optimizerName"] == "rmsprop":
            self.optimizer = optimizers.RMSprop(lr=useSig["lr"], rho=useSig["rho"], epsilon=useSig["epsilon"], decay=useSig["decay"])
        elif kwargs["optimizerName"] == "adagrad":
            self.optimizer =optimizers.Adagrad(lr=useSig["lr"], epsilon=useSig["epsilon"], decay=useSig["decay"])
        elif kwargs["optimizerName"] == "adam":
            self.optimizer =optimizers.Adam(lr=useSig["lr"], beta_1=useSig["beta_1"], beta_2=useSig["beta_2"],
                                            epsilon=useSig["epsilon"], decay=useSig["decay"], amsgrad=useSig["amsgrad"])
        else:
            raise RuntimeError("This should certainly not happen!")

    @staticmethod
    def _parseSignature(signature):
        """ Parses the inspect.signature object and returns a dict with argument, default pairs """
        defaultSignature = {}
        for k,v in signature.parameters.items():
            if v.default is not inspect.Parameter.empty:
                defaultSignature[k] = v.default
        return defaultSignature


    def getInfoDict(self):
        """ Save attributies in dict """
        info = {}
        for attribute in inspect.getmembers(self):
            if attribute[0].startswith("__"):
                continue
            if isinstance(attribute[1], (int, float, list, str, dict)) or attribute[1] is None:
                key, attribute = attribute
                if key == "metrics":
                    if isinstance(attribute, list):
                        attribute = [x.__name__ if callable(x) else x for x in attribute]
                info[key] = attribute         
                
        return info
