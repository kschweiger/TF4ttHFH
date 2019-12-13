#!/bin/bash

COMMAND="python checkInputData.py --normalized --plotDist "
#VARS="met_pt ht30 jets_pt_0 jets_pt_1 jets_pt_2 jets_pt_3 jets_pt_4 jets_pt_5 jets_pt_6 jets_eta_0 jets_eta_1 jets_eta_2 jets_eta_3 jets_eta_4 jets_eta_5 jets_eta_6 Detaj_5 jets_btagDeepCSV_0 jets_btagDeepCSV_1 jets_btagDeepCSV_2 jets_btagDeepCSV_3 jets_btagDeepCSV_4 jets_btagDeepCSV_5 jets_btagDeepCSV_6 bJets_aplanarity bJets_sphericity btagNorm MEM selJets_Mass_jj_min selJets_Max_bVal selJets_DeltaR_jj_min selJets_Avg_bVal selJets_DeltaR_jj_max selJets_Mass_DeltaR_jj_min selJets_HT selJets_Mass_jj_max selJets_Mass_jj_avg selJets_DeltaR_jj_avg selJets_Sq_diffAvgbVal selJets_Min_bVal selJets_Mass_DeltaR_jj_H H_6 H_7 H_4 H_5 H_2 H_3 H_0 H_1 R_1 R_2 R_3 R_4 R_5 R_6 R_7 sphericity aplanarity jets_bsort_pt_0 jets_bsort_pt_1 jets_bsort_btag_0 jets_bsort_btag_1 jets_bsort_eta_1 jets_bsort_eta_0 looseSelJets_Mass_jj_avg looseSelJets_Mass_jj_min looseSelJets_Mass_jj_max looseSelJets_Min_bVal looseSelJets_Sq_diffAvgbVal looseSelJets_DeltaR_jj_max looseSelJets_Max_bVal looseSelJets_Mass_DeltaR_jj_H looseSelJets_HT looseSelJets_DeltaR_jj_avg looseSelJets_Avg_bVal looseSelJets_Mass_DeltaR_jj_min looseSelJets_DeltaR_jj_min bJets_Mass_DeltaR_jj_H bJets_HT bJets_Mass_jj_max bJets_Avg_bVal bJets_Mass_DeltaR_jj_min bJets_DeltaR_jj_avg bJets_DeltaR_jj_min bJets_Mass_jj_min bJets_Max_bVal bJets_Min_bVal bJets_DeltaR_jj_max bJets_Mass_jj_avg bJets_Sq_diffAvgbVal"
VARS='aplanarity jets_pt_4 selJets_DeltaR_jj_min H_0 MEM R_1 R_4 Detaj_5 Detaj_6 Detaj_7 sphericity selJets_DeltaR_jj_max met_pt jets_eta_2 jets_eta_3 selJets_DeltaR_jj_avg jets_eta_1 H_5 jets_bsort_eta_1 jets_bsort_eta_0'
OUTPUTBASE=$1
FILEFOLDER=$2

for NJETS in 7 8 9
do
    ${COMMAND} --output ${OUTPUTBASE}/inputs/${NJETS}J/ttHbb/RegionComp --input ${FILEFOLDER}/preprocData_Run2Dev_TR_v1_ttHbb_${NJETS}J.h5  ${FILEFOLDER}/preprocData_Run2Dev_CR_v1_ttHbb_${NJETS}J.h5 ${FILEFOLDER}/preprocData_Run2Dev_SR_v1_ttHbb_${NJETS}J.h5 --inputNames TR CR SR --plotVars ${VARS}
    ${COMMAND} --output ${OUTPUTBASE}/inputs/${NJETS}J/ttHad/RegionComp --input ${FILEFOLDER}/preprocData_Run2Dev_TR_v1_ttHad_${NJETS}J.h5  ${FILEFOLDER}/preprocData_Run2Dev_CR_v1_ttHad_${NJETS}J.h5 ${FILEFOLDER}/preprocData_Run2Dev_SR_v1_ttHad_${NJETS}J.h5 --inputNames TR CR SR --plotVars ${VARS}
    ${COMMAND} --output ${OUTPUTBASE}/inputs/${NJETS}J/Data/RegionComp --input ${FILEFOLDER}/preprocData_Run2Dev_TR_v1_Data_${NJETS}J.h5  ${FILEFOLDER}/preprocData_Run2Dev_CR_v1_Data_${NJETS}J.h5  --inputNames TR CR --plotVars ${VARS}
done




