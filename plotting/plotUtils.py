import sys
import os
import logging

import matplotlib.pyplot as plt

import numpy as np

def getColors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def make1DHistoPlot(listOfValues, listOfWeights, output, nBins, binRange, varAxisName, legendEntries, normalized=False, log=False, text=None, xtextStart=0, savePDF=True, drawCMS = "Preliminary", drawLumi = 10.00):
    fig, base = plt.subplots(dpi=150)
    for iVal, values in enumerate(listOfValues):
        if listOfWeights is not None:
            thisWeight = listOfWeights[iVal]
        else:
            thisWeight = None
        h = base.hist(values,
                      bins=nBins,
                      range=binRange,
                      linewidth=2,
                      density=normalized,
                      weights=thisWeight,
                      color = getColors()[iVal],
                      log=log,
                      histtype='step')

    base.set_xlabel(varAxisName)
    if normalized:
        base.set_ylabel("Normalized Units")
    else:
        base.set_ylabel("Events")
    base.set_xlim(binRange)
    base.grid(False)
    base.legend(legendEntries)
    if drawCMS is not None:
        if isinstance(drawCMS, str):
            base.text(0.0, 1.01, "CMS", transform=base.transAxes,
                      fontsize="xx-large", fontweight = "bold")
            base.text(0.12, 1.01, drawCMS, transform=base.transAxes,
                      fontsize="large", fontstyle = "italic") 
        else:
            raise TypeError("Pass string for drawCMS")
    if drawLumi is not None:
        if isinstance(drawLumi, float):
            base.text(0.87
                      , 1.01, "{0:.1f}".format(drawLumi)+" fb$^{-1}$",
                      transform=base.transAxes,
                      fontsize="medium")
        else:
            raise TypeError("Pass float for drawLumi")
    if text is not None:
        if isinstance(text, str):
            base.text(0, 1.02, text, transform=base.transAxes)
        elif isinstance(text, list):
            ypos = 1.08
            for it, t in enumerate(text):
                logging.debug("Putting %s as label on plot", t)
                base.text(xtextStart, ypos, t, transform=base.transAxes, color=getColors()[it])
                ypos -= 0.06
        else:
            raise NotImplementedError("Pass list or strings as text")

    logging.info("Saving file: {0}".format(output+".pdf"))
    if savePDF:
        plt.savefig(output+".pdf")


    plt.close('all')
    
    return True


def make1DPlot(listOfValueTupless, output, xAxisName, yAxisName, legendEntries, savePDF=True):
    fig, base = plt.subplots(dpi=150)
    
    for iVal, values in enumerate(listOfValueTupless):
        xData, yData = values
        p = base.plot(xData, yData,
                      color = getColors()[iVal])
    base.set_xlabel(xAxisName)
    base.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    base.set_ylabel(yAxisName, labelpad=base.yaxis.labelpad*1.6)
    base.get_yaxis().get_major_formatter().set_powerlimits((-4, 4))
    base.grid(False)
    base.legend(legendEntries)
    logging.info("Saving file: {0}".format(output+".pdf"))
    if savePDF:
        plt.savefig(output+".pdf")
    
    plt.close(fig)

    return True
        
def makeROCPlot(ROCs, AUCs, output, passedLegend=None, colorOffset=0):
    """
    Plot ROCs passed to function. 

    Args:
      ROCs (dict) : Dict with output from sklearn.metrics.roc_curve 
      AUCs (dict) : Dict with AUC
    """
    assert set(ROCs.keys()) == set(AUCs.keys())
    fig, base = plt.subplots(dpi=150)

    plotVals = []
    legendEntries = []
    for key in ROCs:
        fpr, tpr, _ = ROCs[key]
        plotVals.append((fpr, tpr))
        if passedLegend is None:
            legendEntries.append("{0} - AUC = {1:.2f}".format(key, AUCs[key]))


    for iVal, values in enumerate(plotVals):
        xData, yData = values
        p = base.plot(xData, yData,
                      color = getColors()[iVal+colorOffset])
    
    p = base.plot(np.array([0,1]), np.array([0,1]),
                  color='black',
                  linestyle='dashed')

    base.set_xlabel("True Postive Rate")
    base.set_ylabel("False Postive Rate")

    base.grid(False)
    if passedLegend is None:
        base.legend(legendEntries)
    else:
        base.legend(passedLegend)
    logging.info("Saving file: {0}".format(output+".pdf"))
    plt.savefig(output+".pdf")

    plt.close(fig)

    return True
