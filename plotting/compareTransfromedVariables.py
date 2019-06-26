import sys
import os
import logging
import json

import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('../'))

from plotting.plotUtils import getColors, make1DHistoPlot
from utils.utils import initLogging, checkNcreateFolder



def main(args):
    checkNcreateFolder(args.output, onlyFolder=True)
    evalData = {}
    for iData, pickledData in enumerate(args.input):
        with open(pickledData, "rb") as f:
            evalData[args.inputID[iData]] = pickle.load(f)

    assert evalData[args.inputID[0]]["variables"] == evalData[args.inputID[1]]["variables"]
    
    variables = evalData[args.inputID[0]]["variables"]
    datasets = evalData[args.inputID[0]]["datasets"]
    if args.skipDataset is not None:
        for ds in args.skipDataset:
            datasets.remove(ds)
    
    inputData = {}
    predictionData = {}
    for name in args.inputID:
        inputData[name] = {}
        predictionData[name] = {}
        for iVar, var in enumerate(variables):
            logging.debug("Adding array for output %s, variable %s", name, var)
            dataList =  [evalData[name]["inputData"][i][:, iVar] for i in range(len(evalData[name]["inputData"]))]
            dataListPred =  [evalData[name]["predictionData"][i][:, iVar] for i in range(len(evalData[name]["predictionData"]))]
            inputData[name][var] = {}
            predictionData[name][var] = {}
            for iDataset, dataset in enumerate(datasets):
                inputData[name][var][dataset] = dataList[iDataset]
                logging.debug("Added input data for %s with len %s", dataset, len(inputData[name][var][dataset]))
                predictionData[name][var][dataset] = dataListPred[iDataset]
                logging.debug("Added prediction data for %s with len %s", dataset, len(inputData[name][var][dataset]))
        

    if args.transform:
        transtransforms = {}
        for iName, name in enumerate(args.inputID):
            with open(args.useTransform,"r") as f:
                transtransforms[name] = json.load(f)
            for iVar, var in enumerate(variables):
                for iDataset, dataset in enumerate(datasets):
                    inputData[name][var][dataset] = transform("Gauss", inputData[name][var][dataset], [var], transtransforms[name])
                    predictionData[name][var][dataset] = transform("Gauss", predictionData[name][var][dataset], [var], transtransforms[name])
        
                
    logging.info("Starting plotting")
    for var in variables:
        logging.info("Plotting var %s", var)
        for dataset in datasets:
            dataSetCompInput = []
            dataSetCompPred = []
            dataSetCompAll = []
            legendInput = []
            legendPrediction = []
            legendAll = []
            for name in args.inputID:
                dataSetCompInput.append(inputData[name][var][dataset])
                dataSetCompPred.append(predictionData[name][var][dataset])
                dataSetCompAll.append(inputData[name][var][dataset])
                dataSetCompAll.append(predictionData[name][var][dataset])
                thisTextInput = "%s Input data (%s)"%(name, dataset)
                thisTextPrediction = "%s Predicted data (%s)"%(name, dataset)
                legendInput.append(thisTextInput)
                legendPrediction.append(thisTextPrediction)
                legendAll.append(thisTextInput)
                legendAll.append(thisTextPrediction)
            make1DHistoPlot(dataSetCompInput,
                            None,
                            "{0}/comp_input_{1}_{2}".format(args.output, var, dataset),
                            nBins = 40,
                            binRange = (-4, 4),
                            varAxisName = var,
                            legendEntries = legendInput,
                            normalized=True)
            make1DHistoPlot(dataSetCompPred,
                            None,
                            "{0}/comp_prediction_{1}_{2}".format(args.output, var, dataset),
                            nBins = 40,
                            binRange = (-4, 4),
                            varAxisName = var,
                            legendEntries = legendPrediction,
                            normalized=True)
            make1DHistoPlot(dataSetCompAll,
                            None,
                            "{0}/comp_all_{1}_{2}".format(args.output, var, dataset),
                            nBins = 40,
                            binRange = (-4, 4),
                            varAxisName = var,
                            legendEntries = legendAll,
                            normalized=True)

def transform(method, data, variables, transformation):
    if method == "Gauss":
        std = np.array([transformation["std"][var] for var in variables])
        mu = np.array([transformation["mu"][var] for var in variables])
        return (data*std)+mu
    else:
        raise NotImplementedError("Method %s not implemented"%method)
            

def parseArgs(args):
    import argparse
    argumentparser = argparse.ArgumentParser(
        description='Script for plotting variables used training output (actually the settings from it ;))',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argumentparser.add_argument(
        "--log",
        action="store",
        type=int,
        default=20,
        help="Define logging level: CRITICAL - 50, ERROR - 40, WARNING - 30, INFO - 20, DEBUG - 10, \
        NOTSET - 0 \nSet to 0 to activate ROOT root messages"
    )
    argumentparser.add_argument(
        "--input",
        action="store",
        type=str,
        required=True,
        nargs="+",
        help="Help...",
    )
    argumentparser.add_argument(
        "--inputID",
        action="store",
        type=str,
        required=True,
        nargs="+",
        help="Help...",
    )
    argumentparser.add_argument(
        "--output",
        action="store",
        type=str,
        required=True,
        help="Output file",
    )
    argumentparser.add_argument(
        "--skipDataset",
        action="store",
        type=str,
        nargs="+",
        help="Datasets to be excluded",
        default=None
    )
    argumentparser.add_argument(
        "--transform",
        action="store_true",
        help="Transfrom the variabls back to physics distributions",
    )
    argumentparser.add_argument(
        "--useTransform",
        action="store",
        type=str,
        default=None,
        help="Path to transfroamtions used in the training",
    )
    
    

    return argumentparser.parse_args(args)

if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])

    initLogging(args.log)

    main(args)
