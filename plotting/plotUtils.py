import sys
import os
import logging

import matplotlib.pyplot as plt

import numpy as np

def getColors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def make1DHistoPlot(listOfValues, listOfWeights, output, nBins, binRange, varAxisName, legendEntries, normalized=False, log=False, text=None, xtextStart=0, savePDF=True):
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


    plt.close(fig)

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
        
