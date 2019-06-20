import sys
import os
import logging

import matplotlib.pyplot as plt
import numpy as np

def getColors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def make1DPlot(listOfValues, listOfWeights, output, nBins, binRange, varAxisName, legendEntries, normalized=False, savePDF=True):
    fig, base = plt.subplots(dpi=150)
    for iVal, values in enumerate(listOfValues):
        h = base.hist(values,
                      bins=nBins,
                      range=binRange,
                      linewidth=2,
                      density=normalized,
                      weights=listOfWeights[iVal],
                      color = getColors()[iVal],
                      histtype='step')

    base.set_xlabel(varAxisName)
    if normalized:
        base.set_ylabel("Normalized Units")
    else:
        base.set_ylabel("Events")
    base.set_xlim(binRange)
    base.grid(False)
    base.legend(legendEntries)
    logging.info("Saving file: {0}".format(output+".pdf"))
    if savePDF:
        plt.savefig(output+".pdf")

    plt.close(fig)

    return True
