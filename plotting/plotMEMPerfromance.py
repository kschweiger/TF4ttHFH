import sys,os
import logging

import numpy as np

sys.path.insert(0, os.path.abspath('../'))
from utils.utils import initLogging, checkNcreateFolder, getROCs

from plotting.style import StyleConfig

import checkInputData
import plotUtils

basePath = os.path.expanduser("~/Code/TF4ttHFH")
subfolder = "Private/"
inFileBase = "/Volumes/Korbinian Backup/Data/LegacyRun2 DNN Training/h5/"

plots = {
    "7J" : {
        "style" : basePath+"/data/plotStyle.cfg",
        "inputs" : [
            inFileBase+"/files_Run2Legacy_v3/Combined/preprocData_Run2Legacy_Comb_SR_v1p1_ttHbb_7J.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_ttSplit_ttHad_7J_ttbb.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_ttSplit_ttHad_7J_ttcc.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_ttSplit_ttHad_7J_ttlf.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_QCD_QCD_7J.h5"
        ],
        "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/v2/TF4ttHFH/LegacyRun2_v3/MEMPerfromance/7J/"+subfolder+"MEM_performance_7J_SR",
        "normalized" : True,
        "Legend" : ["ttH(bb)", "tt+bb", "tt+cc", "tt+lf", "QCD multijet"],
        "Colors" : ["#1d00c6","#640104","#c90313","#fc6769","#159a23"],
        "catLabel" :  "SR with 7 jets",
        "yScale" : 1.5,
        "lumi" : 137.1,
        #"CMSString" : "Work in progress (Simulation)",        
        #"CMSString" : "Private work (Simulation)",        
        "CMSString" : "Private work",        
        #"CMSString" : "Work in progress",        
    },
    "8J" : {
        "style" : basePath+"/data/plotStyle.cfg",
        "inputs" : [
            inFileBase+"/files_Run2Legacy_v3/Combined/preprocData_Run2Legacy_Comb_SR_v1p1_ttHbb_8J.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_ttSplit_ttHad_8J_ttbb.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_ttSplit_ttHad_8J_ttcc.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_ttSplit_ttHad_8J_ttlf.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_QCD_QCD_8J.h5"
        ],
        "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/v2/TF4ttHFH/LegacyRun2_v3/MEMPerfromance/8J/"+subfolder+"MEM_performance_8J_SR",
        "normalized" : True,
        "Legend" : ["ttH(bb)", "tt+bb", "tt+cc", "tt+lf", "QCD multijet"],
        "Colors" : ["#1d00c6","#640104","#c90313","#fc6769","#159a23"],
        "catLabel" :  "SR with 8 jets",
        "yScale" : 1.5,
        "lumi" : 137.1,
        #"CMSString" : "Work in progress (Simulation)",        
        #"CMSString" : "Private work (Simulation)",        
        "CMSString" : "Private work",        
        #"CMSString" : "Work in progress",        
    },
    "9J" : {
        "style" : basePath+"/data/plotStyle.cfg",
        "inputs" : [
            inFileBase+"/files_Run2Legacy_v3/Combined/preprocData_Run2Legacy_Comb_SR_v1p1_ttHbb_9J.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_ttSplit_ttHad_9J_ttbb.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_ttSplit_ttHad_9J_ttcc.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_ttSplit_ttHad_9J_ttlf.h5",
            inFileBase+"/files_Run2Legacy_v4/Combined/preprocData_Run2Legacy_Comb_SR_v1p2_QCD_QCD_9J.h5"
        ],
        "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/v2/TF4ttHFH/LegacyRun2_v3/MEMPerfromance/9J/"+subfolder+"MEM_performance_9J_SR",
        "normalized" : True,
        "Legend" : ["ttH(bb)", "tt+bb", "tt+cc", "tt+lf", "QCD multijet"],
        "Colors" : ["#1d00c6","#640104","#c90313","#fc6769","#159a23"],
        "catLabel" :  "SR with $\geq 9$ jets",
        "yScale" : 1.5,
        "lumi" : 137.1,
        #"CMSString" : "Work in progress (Simulation)",        
        #"CMSString" : "Private work (Simulation)",        
        "CMSString" : "Private work",        
        #"CMSString" : "Work in progress",        
    }
    }


if __name__ == "__main__":

    print(plots)
    initLogging(20)
    for plot in plots:
        logging.warning("Plotting %s", plot)
        plotSettings = plots[plot]

        if "CMSString" in plotSettings.keys():
            thisCMSString = plotSettings["CMSString"]
        else:
            thisCMSString = "Work in progress"
        if "Colors" in plotSettings.keys():
            theseColors = plotSettings["Colors"]
        else:
            theseColors = None
        if "catLabel" in plotSettings.keys():
            thisCatLabel = plotSettings["catLabel"]
        else:
            thisCatLabel = None
        if "yScale" in plotSettings.keys():
            thisYScale = plotSettings["yScale"]
        else:
            thisYScale = 1.25


        #thisCMSString = None
            
        styleConfig = StyleConfig(plotSettings["style"])

        inputDFs = checkInputData.getDataframes([filename for filename in plotSettings["inputs"]])
        
        checkNcreateFolder(os.path.expanduser(plotSettings["output"]))


        checkInputData.getDistributions(
            styleConfig,
            inputDFs,
            os.path.expanduser(plotSettings["output"]),
            ["MEM"],
            plotSettings["Legend"],
            plotSettings["normalized"],
            False,
            plotSettings["lumi"],
            thisCMSString,
            theseColors,
            thisCatLabel,
            thisYScale,
            addYields=False,
            #labelStartY=1.02,
            addStats=False
        ) 


        ROCPlotvals = {}
        AUCPlotvals = {}
        ROCPlotLabels = []
        for i, df in enumerate(inputDFs):
            if i == 0:
                logging.info("Skipping first DF (will be used as reference)")
                continue
            logging.info("Getting ROCs for sample %s", i)
            nSignal = len(inputDFs[0].MEM)
            nBackgorund = len(df.MEM)
            logging.info("nSignal = %s | nBkg = %s", nSignal, nBackgorund)
            ROCPlotvals[plotSettings["Legend"][i]], AUCPlotvals[plotSettings["Legend"][i]] = getROCs(
                np.append(np.array(nSignal*[0]), np.array(nBackgorund*[1])),
                np.append(
                    inputDFs[0].MEM.values,
                    df.MEM.values,
                    ),
                np.append(
                    checkInputData.getWeights(inputDFs[0]).to_numpy()[:, 0],
                    checkInputData.getWeights(df).to_numpy()[:, 0]
                )
            )
            ROCPlotLabels.append("{0} vs {1} - AUC {2:.2f}".format(plotSettings["Legend"][0], plotSettings["Legend"][i], AUCPlotvals[plotSettings["Legend"][i]]))
        output_file_name = "{0}_{1}".format(os.path.expanduser(plotSettings["output"]), "ROCs")
        plotUtils.makeROCPlot(ROCPlotvals, AUCPlotvals,
                              output = output_file_name,
                              passedLegend = ROCPlotLabels,
                              forceColor = theseColors[1:],
                              drawLumi = plotSettings["lumi"],
                              drawCMS = thisCMSString,
                              catLabel = thisCatLabel,
                              #labelStartY=1.02,
        )
        
            
            
                              
            # ROCPlotvals[bkg+addDisc], AUCPlotvals[bkg+addDisc] = getROCs(np.append(np.array(nSignal*[0]),
            #                                                                        np.array(nBackgorund*[1])),
            #                                                              np.append(data[config.signalSampleGroup].getTestData(asMatrix=False)[addDisc].values,
            #                                                                        data[bkg].getTestData(asMatrix=False)[addDisc].values),
            #                                                              np.append(weights[config.signalSampleGroup],
            #                                                                        weights[bkg]))
            
        
        # for bkg in bkgs:
        #     logging.info("Getting ROCs for %s", bkg)
        #     nSignal = len(predictions[config.signalSampleGroup][:,0])
        #     nBackgorund = len(predictions[bkg][:,0])
        #     ROCPlotvals[bkg+"DNN"], AUCPlotvals[bkg+"DNN"] = getROCs(np.append(np.array(nSignal*[0]),
        #                                                                        np.array(nBackgorund*[1])),
        #                                                              np.append(predictions[config.signalSampleGroup][:,0],
        #                                                                        predictions[bkg][:,0]),
        #                                                              np.append(weights[config.signalSampleGroup],
        #                                                                        weights[bkg]))

        #     ROCPlotLabels.append("DNN : {0} vs {1} - AUC {2:.2f}".format(config.signalSampleGroup, bkg, AUCPlotvals[bkg+"DNN"]))
        #     for addDisc in config.plotAdditionalDisc:
        #         ROCPlotvals[bkg+addDisc], AUCPlotvals[bkg+addDisc] = getROCs(np.append(np.array(nSignal*[0]),
        #                                                                                np.array(nBackgorund*[1])),
        #                                                                      np.append(data[config.signalSampleGroup].getTestData(asMatrix=False)[addDisc].values,
        #                                                                                data[bkg].getTestData(asMatrix=False)[addDisc].values),
        #                                                                      np.append(weights[config.signalSampleGroup],
        #                                                                                weights[bkg]))


        #         ROCPlotLabels.append("{3} : {0} vs {1} - AUC {2:.2f}".format(config.signalSampleGroup, bkg, AUCPlotvals[bkg+addDisc], addDisc))


        # ROCColors = []
        # for c in config.forceColors[1:]:
        #     ROCColors.append(c)
        #     if alternateDashed:
        #         ROCColors.append(c)            

        # output_file_name = "{0}/{1}_{2}".format(config.plottingOutput, config.plottingPrefix, "ROCs")
        # if saveVals:
        #     logging.info("Saving ROC vals to %s.plk", output_file_name)
        #     pickle.dump( {"ROCs" : ROCPlotvals, "AUCs" : AUCPlotvals, "labels" : ROCPlotLabels}, open( output_file_name+".pkl", "wb" ) )

        # makeROCPlot(ROCPlotvals, AUCPlotvals,
        #             output = output_file_name,
        #             passedLegend = ROCPlotLabels,
        #             forceColor = ROCColors,
        #             alternateDash = alternateDashed)

