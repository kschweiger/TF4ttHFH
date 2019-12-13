import sys,os
import logging

sys.path.insert(0, os.path.abspath('../'))
from utils.utils import initLogging, checkNcreateFolder

from plotting.style import StyleConfig

import checkInputData

basePath = os.path.expanduser("~/Code/TF4ttHFH")

plots = {
    #################################################################################################################################
    ################################################### Sample comp VR  #############################################################
    #################################################################################################################################
    # "2016_VR_7J" : {
    #     "style" : basePath+"/data/plotStyle.cfg",
    #     "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_ttHbb_7J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_ttHad_7J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_Data_7J.h5"],
    #     "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/ttH_vs_ttHad_vs_Data_VR/7J/comp_7J",
    #     "plotType" : "plotDist",
    #     "normalized" : True,
    #     "Legend" : ["ttH(bb)", "tt+Jets (FH)", "Data/QCD"],
    #     "excludeVars" : ["bJets*", "*weight*", "*Weight*", "njets", "nPVs", "nBDeepFlav*"],
    #     "lumi" : 35.9,
    # },
    # "2016_VR_8J" : {
    #     "style" : basePath+"/data/plotStyle.cfg",
    #     "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_ttHbb_8J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_ttHad_8J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_Data_8J.h5"],
    #     "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/ttH_vs_ttHad_vs_Data_VR/8J/comp_8J",
    #     "plotType" : "plotDist",
    #     "normalized" : True,
    #     "Legend" : ["ttH(bb)", "tt+Jets (FH)", "Data/QCD"],
    #     "excludeVars" : ["bJets*", "*weight*", "*Weight*", "njets", "nPVs", "nBDeepFlav*"],
    #     "lumi" : 35.9,
    # },
    # "2016_VR_9J" : {
    #     "style" : basePath+"/data/plotStyle.cfg",
    #     "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_ttHbb_9J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_ttHad_9J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_Data_9J.h5"],
    #     "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/ttH_vs_ttHad_vs_Data_VR/9J/comp_9J",
    #     "plotType" : "plotDist",
    #     "normalized" : True,
    #     "Legend" : ["ttH(bb)", "tt+Jets (FH)", "Data/QCD"],
    #     "excludeVars" : ["bJets*", "*weight*", "*Weight*", "njets", "nPVs", "nBDeepFlav*"],
    #     "lumi" : 35.9,
    # },
    #################################################################################################################################
    ########################################### Signal regions comp - Same sample ###################################################
    #################################################################################################################################
    # "2016_Data_regionCompare_7J" : {
    #     "style" : basePath+"/data/plotStyle.cfg",
    #     "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR_v1_Data_7J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR_v1_Data_7J.h5"],
    #     "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/Data_regionCompare/7J/regionComp_7J",
    #     "plotType" : "plotDist",
    #     "normalized" : True,
    #     "Legend" : ["Data/QCD TR", "Data/QCD CR"],
    #     "excludeVars" : [],
    #     "lumi" : 35.9,
    # },
    # "2016_Data_regionCompare_8J" : {
    #     "style" : basePath+"/data/plotStyle.cfg",
    #     "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR_v1_Data_8J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR_v1_Data_8J.h5"],
    #     "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/Data_regionCompare/8J/regionComp_8J",
    #     "plotType" : "plotDist",
    #     "normalized" : True,
    #     "Legend" : ["Data/QCD TR", "Data/QCD CR"],
    #     "excludeVars" : [],
    #     "lumi" : 35.9,
    # },
    # "2016_Data_regionCompare_9J" : {
    #     "style" : basePath+"/data/plotStyle.cfg",
    #     "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR_v1_Data_9J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR_v1_Data_9J.h5"],
    #     "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/Data_regionCompare/9J/regionComp_9J",
    #     "plotType" : "plotDist",
    #     "normalized" : True,
    #     "Legend" : ["Data/QCD TR", "Data/QCD CR"],
    #     "excludeVars" : [],
    #     "lumi" : 35.9,
    # },
    # "2016_ttHbb_regionCompare_7J" : {
    #     "style" : basePath+"/data/plotStyle.cfg",
    #     "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR_v1_ttHbb_7J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR_v1_ttHbb_7J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_SR_v1_ttHbb_7J.h5"],
    #     "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/ttHbb_regionCompare/7J/regionComp_7J",
    #     "plotType" : "plotDist",
    #     "normalized" : True,
    #     "Legend" : ["ttH(bb) TR", "ttH(bb) CR", "ttH(bb) SR"],
    #     "excludeVars" : ["MEM"],
    #     "lumi" : 35.9,
    # },
    # "2016_ttHbb_regionCompare_8J" : {
    #     "style" : basePath+"/data/plotStyle.cfg",
    #     "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR_v1_ttHbb_8J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR_v1_ttHbb_8J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_SR_v1_ttHbb_8J.h5"],
    #     "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/ttHbb_regionCompare/8J/regionComp_8J",
    #     "plotType" : "plotDist",
    #     "normalized" : True,
    #     "Legend" : ["ttH(bb) TR", "ttH(bb) CR", "ttH(bb) SR"],
    #     "excludeVars" : ["MEM"],
    #     "lumi" : 35.9,
    # },
    # "2016_ttHbb_regionCompare_9J" : {
    #     "style" : basePath+"/data/plotStyle.cfg",
    #     "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR_v1_ttHbb_9J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR_v1_ttHbb_9J.h5",
    #                 basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_SR_v1_ttHbb_9J.h5"],
    #     "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/ttHbb_regionCompare/9J/regionComp_9J",
    #     "plotType" : "plotDist",
    #     "normalized" : True,
    #     "Legend" : ["ttH(bb) TR", "ttH(bb) CR", "ttH(bb) SR"],
    #     "excludeVars" : ["MEM"],
    #     "lumi" : 35.9,
    # },
    #################################################################################################################################
    ############################################ Validation regions comp - Same sample  #############################################
    #################################################################################################################################
    "2016_Data_regionCompare_VR_7J" : {
        "style" : basePath+"/data/plotStyle.cfg",
        "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR2_v1_Data_7J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR2_v1_Data_7J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_Data_7J.h5"],
        "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/Data_regionCompare_VR/7J/regionComp_7J",
        "plotType" : "plotDist",
        "normalized" : True,
        "Legend" : ["Data/QCD ValTR", "Data/QCD ValCR", "Data/QCD VR"],
        "excludeVars" : [],
        "lumi" : 35.9,
    },
    "2016_Data_regionCompare_VR_8J" : {
        "style" : basePath+"/data/plotStyle.cfg",
        "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR2_v1_Data_8J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR2_v1_Data_8J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_Data_8J.h5"],
        "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/Data_regionCompare_VR/8J/regionComp_8J",
        "plotType" : "plotDist",
        "normalized" : True,
        "Legend" : ["Data/QCD ValTR", "Data/QCD ValCR", "Data/QCD VR"],
        "excludeVars" : [],
        "lumi" : 35.9,
    },
    "2016_Data_regionCompare_VR_9J" : {
        "style" : basePath+"/data/plotStyle.cfg",
        "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR2_v1_Data_9J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR2_v1_Data_9J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_Data_9J.h5"],
        "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/Data_regionCompare_VR/9J/regionComp_9J",
        "plotType" : "plotDist",
        "normalized" : True,
        "Legend" : ["Data/QCD ValTR", "Data/QCD ValCR", "Data/QCD VR"],
        "excludeVars" : [],
        "lumi" : 35.9,
    },
    "2016_ttHbb_regionCompare_VR_7J" : {
        "style" : basePath+"/data/plotStyle.cfg",
        "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR2_v1_ttHbb_7J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR2_v1_ttHbb_7J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_ttHbb_7J.h5"],
        "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/ttHbb_regionCompare_VR/7J/regionComp_7J",
        "plotType" : "plotDist",
        "normalized" : True,
        "Legend" : ["ttH(bb) ValTR", "ttH(bb) ValCR", "ttH(bb) VR"],
        "excludeVars" : ["MEM"],
        "lumi" : 35.9,
    },
    "2016_ttHbb_regionCompare_VR_8J" : {
        "style" : basePath+"/data/plotStyle.cfg",
        "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR2_v1_ttHbb_8J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR2_v1_ttHbb_8J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_ttHbb_8J.h5"],
        "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/ttHbb_regionCompare_VR/8J/regionComp_8J",
        "plotType" : "plotDist",
        "normalized" : True,
        "Legend" : ["ttH(bb) ValTR", "ttH(bb) ValCR", "ttH(bb) VR"],
        "excludeVars" : ["MEM"],
        "lumi" : 35.9,
    },
    "2016_ttHbb_regionCompare_VR_9J" : {
        "style" : basePath+"/data/plotStyle.cfg",
        "inputs" : [basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_TR2_v1_ttHbb_9J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_CR2_v1_ttHbb_9J.h5",
                    basePath+"/files_Run2Legacy_v3/2016/preprocData_Run2Legacy_2016_VR_v1_ttHbb_9J.h5"],
        "output" : "~/Documents/PhD/LegacyRun2/LegacyRun2_AH_v3/TF4ttHFH/LegacyRun2_v3/2016/ttHbb_regionCompare_VR/9J/regionComp_9J",
        "plotType" : "plotDist",
        "normalized" : True,
        "Legend" : ["ttH(bb) ValTR", "ttH(bb) ValCR", "ttH(bb) VR"],
        "excludeVars" : ["MEM"],
        "lumi" : 35.9,
    },
}

if __name__ == "__main__":
    initLogging(30)
    for plot in plots:
        logging.warning("Plotting %s", plot)
        plotSettings = plots[plot]


        styleConfig = StyleConfig(plotSettings["style"])

        inputDFs = checkInputData.getDataframes([filename for filename in plotSettings["inputs"]])

        vars2Process = checkInputData.generateVariableList(inputDFs[0], ["All"], plotSettings["excludeVars"])

        checkNcreateFolder(os.path.expanduser(plotSettings["output"]))
        
        if plotSettings["plotType"] == "plotDist":
            checkInputData.getDistributions(
                styleConfig,
                inputDFs,
                os.path.expanduser(plotSettings["output"]),
                vars2Process,
                plotSettings["Legend"],
                plotSettings["normalized"],
                False,
                plotSettings["lumi"],
            ) 
        else:
            raise NotImplementedError
        




    
