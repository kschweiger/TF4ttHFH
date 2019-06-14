import sys
import os
import logging

import matplotlib.pyplot as plt
import pandas as pd

from itertools import permutations
sys.path.insert(0, os.path.abspath('../'))

from utils.utils import initLogging
from plotting.style import StyleConfig

#plt.style.use('seaborn') #This can lead to a crash. To review all available styles use `print(plt.style.available)

def getColors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def generateVariableList(dataframe, whitelist, blacklist):
    """
    Function for generating the variable list.
    
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

    for var in blacklist:
        if var in whitelist:
            whitelist.remove(var)

    logging.debug("Resulting whitelist = %s", whitelist)
    return whitelist

def plotDataframeVars(dataframes, output, variable, dfNames, nBins, binRange, varAxisName, normalized=False, savePDF=True):
    """ Function for plotting the passed variable from all passed dataframes """
    # TODO: Implement normalization
    # TODO: Implement weights
    assert len(dfNames) == len(dataframes)
    fig, base = plt.subplots(dpi=150)
    for idf, fullDF in enumerate(dataframes):
        df = fullDF.loc[:, [variable]]
        df.plot.hist(
            ax = base,
            color = getColors()[idf],
            histtype="step",
            bins = nBins,
            range = binRange,
            linewidth = 2
        )

    
    base.set_xlabel(varAxisName)
    if normalized:
        base.set_ylabel("Normalized Units")
    else:
        base.set_ylabel("Events")
    base.set_xlim(binRange)
    base.grid(False)
    base.legend(dfNames)
    logging.info("Saving file: {0}".format(output+".pdf"))
    if savePDF:
        plt.savefig(output+".pdf")

    plt.close(fig)
        
    return True
    
def plotCorrelation(dataframe, output, variable1, nBins1, binRange1, var1AxisTitle, variable2, nBins2, binRange2, var2AxisTitle, savePDF=True):
    """ Function for plotting correlation between 2 variables of the passed dataframe """
    # TODO: Implement weights
    fig, base = plt.subplots(dpi=150)
    fig.subplots_adjust(bottom = 0.16, right = 0.88, left = 0.11, top = 0.92)
    # For some reason pandas only kan do hexagonal binned 2d plots  ¯\_(ツ)_/¯
    # So we convert the columns of interest to numpy arrays and used to standard
    # pyplot implementation -.-
    var1Data = dataframe[variable1].to_numpy()
    var2Data = dataframe[variable2].to_numpy()

    h = base.hist2d(var1Data, var2Data, bins=[nBins1, nBins2], range=[binRange1, binRange2])
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

    

def getCorrealtions(styleConfig, dataframe, outputPath, processVars):
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
            styleConfig.style[var2].nBins,
            styleConfig.style[var2].binRange,
            styleConfig.style[var2].axisName,
            savePDF = True
        )
    logging.info("Exiting function")
    return True

def getDistributions(styleConfig, inputDataframes, outputPath, processVars, inputNames=None, drawNormalized=False):
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
            savePDF = True
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
    
    return argumentparser.parse_args(args)

def getDataframes(filesList):
    return [pd.read_hdf(fileName) for fileName in filesList]
    
def process(args, styleConfig):
    inputDFs = getDataframes(args.input)
    if args.inputNames is not None:
        assert len(args.inputNames) == len(inputDFs)
    
    vars2Process = generateVariableList(inputDFs[0], args.plotVars, args.excludeVars)
    if args.plotCorr:
        logging.info("Will do correlation plots")
        for iDF, inputDF in enumerate(inputDFs):
            logging.info("Processing file %s",inputDFs)
            thisoutput = args.output+"_corr2D_"+(str(iDF) if args.inputNames is None else args.inputNames[iDF])
            getCorrealtions(styleConfig, inputDF, thisoutput, vars2Process)

    if args.plotDist:
        logging.info("Will plot distributions")
        thisOutput = args.output+"_dist_"
        getDistributions(styleConfig, inputDFs, thisOutput, vars2Process, args.inputNames, args.normalized)
            
if __name__ == "__main__":
    args = parseArgs(sys.argv[1:])

    initLogging(args.log)

    styleConfig = StyleConfig(args.style)

    process(args, styleConfig)
