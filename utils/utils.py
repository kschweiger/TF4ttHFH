"""
Collection of util functions (and classes)

K. Schweiger, 2019
"""
import coloredlogs, logging
import os
import sys

import numpy as np

from scipy.integrate import simps
from sklearn.metrics import roc_auc_score, roc_curve, auc

def initLogging(thisLevel):
    """
    Helper function for setting up python logging
    """
    log_format = ('[%(asctime)s] %(funcName)-20s %(levelname)-8s %(message)s')
    if thisLevel == 20:
        thisLevel = logging.INFO
    elif thisLevel == 10:
        thisLevel = logging.DEBUG
    elif thisLevel == 30:
        thisLevel = logging.WARNING
    elif thisLevel == 40:
        thisLevel = logging.ERROR
    elif thisLevel == 50:
        thisLevel = logging.CRITICAL
    else:
        thisLevel = logging.NOTSET


    # logging.basicConfig(
    #     format=log_format,
    #     level=thisLevel,        
    # )

    coloredlogs.install(
        level=thisLevel,
        fmt = log_format
    )
    return True

def checkNcreateFolder(path, onlyFolder=False):
    outpath = path.split("/")
    if len(outpath) > 1: #if no / in string output will be saved in current dir
        outpath = "/".join(outpath[0:-1])
        if not os.path.exists(outpath):
            logging.warning("Creating direcotries %s", outpath)
            os.makedirs(outpath)

    if onlyFolder:
        if not os.path.exists(path):
            logging.warning("Creating direcotries %s", path)
            os.makedirs(path)
        
def reduceArray(inputArray, level):
    """
    Reduces to size of an array by averaging over mulitlpe subentries.
    
    Args:
      inputArray (np.array) : INput array
      level (int) : Level of reduction. Will split input array into subarrays of len level and calc. the average for all
    """
    outArray = []
    splitArray = []
    lastIndex = 0
    for i in range(len(inputArray)):
        if i%level == 0 and i != 0:
            splitArray.append(np.array(inputArray[lastIndex:i]))
            lastIndex = i
    splitArray.append(np.array(inputArray[lastIndex::]))
    for split in splitArray:
        outArray.append(np.average(split))

    return  np.array(outArray)


def getSigBkgArrays(inputlabel, inputArray):
    """Splits inputArray according to inputlables in separate arrays """
    nOutputArrays = inputlabel.max()
    if nOutputArrays == 0:
        raise RuntimeError("Maximum label 0. No splitting possible")
    outLists = [[] for i in range(nOutputArrays+1)]
    for iElem, elem in enumerate(inputArray):
        outLists[inputlabel[iElem]].append(elem)

    return outLists

def getROCs(labels, classifier, weights):
    ROC, AUC = None, None
    ROC = roc_curve(labels, classifier, sample_weight=weights)
    fpr ,tpr ,_ = ROC
    AUC = np.trapz(tpr, fpr)

    if AUC < 0.5:
        logging.warning("Inverting ROC")
        invLabels = []
        for label in labels:
            invLabels.append(0 if label == 1 else 1)
        ROC = roc_curve(invLabels, classifier, sample_weight=weights)
        fpr ,tpr ,_ = ROC
        AUC = np.trapz(tpr, fpr)

    return ROC, AUC

