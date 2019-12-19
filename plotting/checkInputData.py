import sys
import os
import logging

import matplotlib.pyplot as plt
import pandas as pd
import math
import copy

from itertools import permutations
sys.path.insert(0, os.path.abspath('../'))

from utils.utils import initLogging, checkNcreateFolder
from plotting.style import StyleConfig
from plotting.plotUtils import getColors, make1DHistoPlot
#plt.style.use('seaborn') #This can lead to a crash. To review all available styles use `print(plt.style.available)

def transformDataframe(dataframe, variables):
    """ Transforms all passed variables in the dataframe to to distibutions with mean=0 and std=1"""
    transformedDataframe = dataframe.copy()
    for variable in list(variables):
        transformedDataframe[variable] = ((dataframe[variable] - dataframe[variable].mean())/(dataframe[variable].std()))

    return transformedDataframe

def getWeights(dataframe, addWeights=[]):
    thisPUWeight = None
    if "puWeight" in dataframe:
        thisPUWeight = "puWeight"
    elif "weightPURecalc" in dataframe:
        thisPUWeight = "weightPURecalc"
    elif "weight_pu" in dataframe:
        thisPUWeight = "weight_pu"
    else:
        raise KeyError("No valid PU weight found")
    
    standardWeights = [thisPUWeight, "genWeight", "btagWeight_shape", "weight_CRCorr", "triggerWeight"]
    retWeight = None
    for weight in standardWeights+addWeights:
        if retWeight is None:
            retWeight = dataframe.loc[:, [weight]]
            retWeight.rename(columns={weight : "EventWeight"}, inplace=True)
            logging.debug("Mean Weight %s for %s",retWeight["EventWeight"].mean(), weight)
        else:
            tmpWeight = dataframe.loc[:, [weight]]
            if tmpWeight[weight].mean() <= 0: #Temp fix for bug in preprocessing. Only effect data
                logging.warning("Skipping weight %s becuase mean is %s", weight, tmpWeight[weight].mean())
                continue
            tmpWeight.rename(columns={weight : "EventWeight"}, inplace=True)
            logging.debug("Mean Weight %s for %s",tmpWeight["EventWeight"].mean(), weight)
            retWeight *= tmpWeight
        
    return retWeight

def generateVariableList(dataframe, whitelist, blacklist):
    """
    Function for generating the variable list. Blacklist also accepts wildcard "*"'s but only if they are used as first of last char. 
    
    Args:
      dataframe (pd.Dataframe) : Input dataframe
      whitelist (list) : Variable list - Will be check against dataframe. Passing ["All"] will get all columns form dataframe
      blacklist (list) : Variable list to be exlcuded

    Returns:
      variables (list) : List of variables that are in the white list but not in the blacklist
    """
    logging.debug("Whitelist = %s", whitelist)
    logging.debug("Blacklist = %s", blacklist)
    if "All" in whitelist:
        whitelist = list(dataframe.columns)
    else:
        allColumns = list(dataframe.columns)
        for var in whitelist:
            if var not in allColumns:
                raise KeyError("Variable %s not in dataframe columns"%var)


    _whitelist = copy.copy(whitelist)
    for blackVar in blacklist:
        if blackVar in _whitelist:
            whitelist.remove(blackVar)
            continue
        if "*" in blackVar:
            if blackVar.count("*") == 1:
                if blackVar.endswith("*"):
                    query = blackVar.replace("*","")
                    for whiteVar in _whitelist:
                        if whiteVar.startswith(query):
                            whitelist.remove(whiteVar)
                elif blackVar.startswith("*"):
                    query = blackVar.replace("*","")
                    #logging.debug("blackVar %s, Query %s", blackVar, query)
                    for whiteVar in _whitelist:
                        if whiteVar.endswith(query):
                            #logging.debug("Removing: %s", whiteVar)
                            whitelist.remove(whiteVar)
                else:
                    raise RuntimeError("Only Wildcards in the beginning or end of the string are supported")
            elif blackVar.count("*") == 2:
                if blackVar.startswith("*") and blackVar.endswith("*"):
                    query = blackVar.replace("*","")
                    for whiteVar in _whitelist:
                        if query in whiteVar:
                            whitelist.remove(whiteVar)
                else:
                    raise RuntimeError("Only 2 wildcard blacklist items are supported that start and end with *")    
            else:
                raise RuntimeError("Only blacklist wildcards with less than 3 are supported")
    

    logging.debug("Resulting whitelist = %s", whitelist)
    return whitelist

def plotDataframeVars(dataframes, output, variable, dfNames, nBins, binRange, varAxisName, normalized=False, transform=False, savePDF=True, drawStats=True, lumi=None):
    """ Function for plotting the passed variable from all passed dataframes """
    # TODO: Implement weights
    logging.info("Entering plotDataframeVars")
    assert len(dfNames) == len(dataframes)
    logging.info("Getting fig and base")
    fig, base = plt.subplots(dpi=150)
    logging.info("Got fig and base")
    if transform:
        binRange = (-4, 4)

    listOfValues = []
    listOfWeights = []
    texts = []
    for idf, fullDF in enumerate(dataframes):
        logging.info("Getting values form df %s", idf)
        df = fullDF.loc[:, [variable]]
        if transform:
            df = transformDataframe(df, [variable])
        var = df.loc[:, [variable]].to_numpy()[:, 0]
        #listOfValues.append(var)
        weights = getWeights(fullDF)
        weights = weights.to_numpy()
        weights = weights[:, 0]
        logging.debug("Mean weight : %s (%s)", weights.mean(), dfNames[idf] )
        listOfWeights.append(weights)
        listOfValues.append(var)
        texts.append("$\mu$: {0:.3f}, $\sigma$: {1:.3f}".format((var*weights).mean(), (var*weights).std()))
    legendStuff = []
    if drawStats:
        #print(dfNames, texts)
        for iName in range(len(dfNames)):
            legendStuff.append("{0}\n{1}".format(dfNames[iName], texts[iName]))
    else:
        legendStuff = dfNames
        
    #listOfWeights = None
    return make1DHistoPlot(listOfValues, listOfWeights,
                           output=output,
                           nBins=nBins,
                           binRange=binRange,
                           varAxisName=varAxisName,
                           legendEntries=legendStuff,
                           normalized=normalized,
                           #text=texts,
                           #xtextStart=0.25,
                           drawCMS = "Preliminary",
                           drawLumi = lumi,
                           savePDF=savePDF)
    
def plotCorrelation(dataframe, output, variable1, nBins1, binRange1, var1AxisTitle, variable2, nBins2, binRange2, var2AxisTitle, transform=False, savePDF=True):
    """ Function for plotting correlation between 2 variables of the passed dataframe """
    # TODO: Implement weights
    fig, base = plt.subplots(dpi=150)
    fig.subplots_adjust(bottom = 0.16, right = 0.88, left = 0.11, top = 0.92)
    # For some reason pandas only can do hexagonal binned 2d plots  ¯\_(ツ)_/¯
    # So we convert the columns of interest to numpy arrays and used to standard
    # pyplot implementation -.-
    if transform:
        dataframe = transformDataframe(dataframe, [variable1, variable2])
  
    var1Data = dataframe[variable1].to_numpy()
    var2Data = dataframe[variable2].to_numpy()

    weights = getWeights(dataframe)
    weights = weights.to_numpy()
    weights = weights[:, 0]
    #print(weights)
    
    if transform:
        binRange1 = (-4, 4)
        binRange2 = (-4, 4)
        
    h = base.hist2d(
        var1Data, var2Data,
        bins=[nBins1, nBins2],
        range=[binRange1, binRange2],
        weights=weights
    )
    
    plt.colorbar(h[3], ax=base)

    base.set_xlabel(var1AxisTitle)
    base.set_ylabel(var2AxisTitle)
    
    # Get Correaltion
    var1Series = dataframe.loc[:, variable1]
    var2Series = dataframe.loc[:, variable2]
    
    corr = var1Series.corr(var2Series)
    base.text(0, 1.02, "Correlation: {0:0.4f}".format(corr), transform=base.transAxes)


    logging.info("Saving file: {0}".format(output+".pdf"))
    if savePDF:
        plt.savefig(output+".pdf")
    
    plt.close(fig)

    return True

    

def getCorrealtions(styleConfig, dataframe, outputPath, processVars, transform=False):
    """ 
    Function for plotting 2D distribution for all passed variables in passed dataframe 
    
    Args:
      dataframe (pd.Dataframe) : Input dataframe
      outputPath (str) : Output file name
      processVars (list) : List of variables to be plotted
    """
    logging.info("Getting 2D plots with correlation")
    allCombinations = list(permutations(processVars, 2))
    logging.info("Got %s variables. Will make %s plots", len(processVars), len(allCombinations))
    for combination in list(allCombinations):
        logging.debug("Processing combination %s", combination)

        thisOutput = outputPath+("_".join(combination))
        var1 = combination[0]
        var2 = combination[1]

        plotCorrelation(
            dataframe,
            thisOutput,
            var1,
            styleConfig.style[var1].nBins,
            styleConfig.style[var1].binRange,
            styleConfig.style[var1].axisName,
            var2,
            styleConfig.style[var2].nBins,
            styleConfig.style[var2].binRange,
            styleConfig.style[var2].axisName,
            transform = transform,
            savePDF = True
        )
    logging.info("Exiting function")
    return True

def getDistributions(styleConfig, inputDataframes, outputPath, processVars, inputNames=None, drawNormalized=False, transform=False, lumi=None):
    """ 
    Function for plotting 1D distribution comparsions for all passed Dataframes 
    
    Args:
      inputDataframes (list) : List of input dataframes
      outputPath (str) : Output file name
      processVars (list) : List of variables to be plotted
    """
    logging.info("Getting 1D dsitbutions for different dataframes")
    logging.debug("Dataframes passed: %s", len(inputDataframes))
    if inputNames is None:
        inputNames = ["DF"+str(i) for i in range(len(inputDataframes))]
    assert len(inputNames) == len(inputDataframes)
    for i in range(len(inputDataframes)):
        inputNames[i] = "{0} ({1})".format(inputNames[i], inputDataframes[i].shape[0])
    #print(inputNames)
    for variable in processVars:
        logging.info("Processing variable %s", variable)
        plotDataframeVars(
            dataframes = inputDataframes,
            output = outputPath+"_"+variable,
            variable = variable,
            dfNames = inputNames,
            nBins = styleConfig.style[variable].nBins,
            binRange = styleConfig.style[variable].binRange,
            varAxisName = styleConfig.style[variable].axisName,
            normalized = drawNormalized,
            transform = transform,
            savePDF = True,
            lumi = lumi,
        )
        
    logging.info("Exiting function")
    return True

def parseArgs(args):
    import argparse
    argumentparser = argparse.ArgumentParser(
        description='Script for plotting the input variables forectly from the input dataset',
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
        help="Input file (h5 format)",
    )
    argumentparser.add_argument(
        "--output",
        action="store",
        type=str,
        required=True,
        help="Output file",
    )
    argumentparser.add_argument(
        "--plotVars",
        action="store",
        type=str,
        nargs="+",
        help="Output file",
        default = ["All"]
    )
    argumentparser.add_argument(
        "--excludeVars",
        action="store",
        type=str,
        nargs="+",
        help="Output file",
        default = []
    )
    argumentparser.add_argument(
        "--inputNames",
        action="store",
        type=str,
        nargs="+",
        help="Legend names for the input dataset",
        default = None
    )
    argumentparser.add_argument(
        "--style",
        action="store",
        type=str,
        help="Styleconfig path",
        default = "../data/plotStyle.cfg"
    )
    argumentparser.add_argument(
        "--plotCorr",
        action="store_true",
        help="Do correlation plots for all combination of passed variables",
    )
    argumentparser.add_argument(
        "--plotDist",
        action="store_true",
        help="Do distributions for all combination of passed variables",
    )
    argumentparser.add_argument(
        "--normalized",
        action="store_true",
        help="Do distributions for all combination of passed variables",
    )
    argumentparser.add_argument(
        "--transform",
        action="store_true",
        help="Transforms the variable to distributions with mean=0 and sigma=1",
    )
    argumentparser.add_argument(
        "--merge",
        action="store",
        type=str,
        nargs="+",
        help="Styleconfig path",
        default = None
    )
    argumentparser.add_argument(
        "--xsec",
        action="store",
        type=float,
        nargs="+",
        help="Styleconfig path",
        default = None
    )
    argumentparser.add_argument(
        "--nGen",
        action="store",
        type=float,
        nargs="+",
        help="Styleconfig path",
        default = None
    )
    argumentparser.add_argument(
        "--lumi",
        action="store",
        type=float,
        help="Lumi using in the plot as lumi label",
        default = None
    )
    
    return argumentparser.parse_args(args)

def mergeDatasets(args, variables, lumi=41.5):
    assert len(args.merge) == len(args.xsec)
    assert len(args.merge) == len(args.nGen)
    
    mergedDatasetName = [filename for filename in args.input if filename.startswith("merge") ][0]
    mergedDatasetName = mergedDatasetName.replace("merge","")

    allDataFrames = getDataframes(args.merge)
    retDataFrames = []
    mergedDataFrame = None
    for iDataset, dataFrame in enumerate(allDataFrames):
        thisXSec = args.xsec[iDataset]
        thisNGen = args.nGen[iDataset]
        sf = (lumi*1000*thisXSec*1)/thisNGen
        logging.debug("Rewighting dataframe %s with %s entries", iDataset, dataFrame.shape[0])
        logging.debug("Weighting df %s with xsec %s and nGen %s -- SF=%s",iDataset, thisXSec, thisNGen, sf)

        dataFrame[variables] = dataFrame[variables] * sf
        retDataFrames.append(dataFrame)
        if mergedDataFrame is None:
            mergedDataFrame = dataFrame.copy()
        else:
            mergedDataFrame = mergedDataFrame.append(dataFrame)
            
    return mergedDatasetName, mergedDataFrame, retDataFrames
    
def getDataframes(filesList):
    return [pd.read_hdf(fileName) for fileName in filesList]

def process(args, styleConfig):
    inputDFs = getDataframes([filename for filename in args.input if not filename.startswith("merge") ])
    vars2Process = generateVariableList(inputDFs[0], args.plotVars, args.excludeVars)
    if len([filename for filename in args.input if filename.startswith("merge") ]) == 1:
        mergedName, mergedDF, weightedDFs = mergeDatasets(args, vars2Process)

        #print(mergedDF)
        inputDFs = inputDFs + [mergedDF]

        if args.inputNames is not None:
            args.inputNames = args.inputNames + [mergedName]
        
    if args.inputNames is not None:
        assert len(args.inputNames) == len(inputDFs)
        
    
    if args.plotCorr:
        logging.info("Will do correlation plots")
        for iDF, inputDF in enumerate(inputDFs):
            logging.info("Processing file %s",inputDFs)
            outpath = args.output.split("/")
            if len(outpath) > 1: #In this case we want to insert the folder for the dataframe
                folders = outpath[0:-1]
                folders.append("DF-"+str(iDF) if args.inputNames is None else args.inputNames[iDF])
                thisOutput = "/".join(folders+[outpath[-1]])
                thisOutput += "_corr2D_"
                checkNcreateFolder(thisOutput)
            else:
                thisOutput = args.output+"_corr2D_"+(str(iDF) if args.inputNames is None else args.inputNames[iDF])
            
            getCorrealtions(styleConfig, inputDF, thisOutput, vars2Process, args.transform)

    if args.plotDist:
        logging.info("Will plot distributions")
        thisOutput = args.output+"_dist_"
        checkNcreateFolder(thisOutput)
        getDistributions(styleConfig, inputDFs, thisOutput, vars2Process, args.inputNames, args.normalized, args.transform, args.lumi)
            
if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])

    initLogging(args.log)

    styleConfig = StyleConfig(args.style)

    process(args, styleConfig)

    output_ = "/".join(args.output.split("/")[0:-1])
    
    with open("{0}/plotCommand.txt".format(output_),"w") as f:
        f.write("python "+" ".join(sys.argv))

    
