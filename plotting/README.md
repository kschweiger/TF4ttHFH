# Plotting scripts

## checkInputData
Plot 1D distributions overlaying passed variables (`--plotVars`) for mulitple dataset (`--input`) when `--plotDist` is passed.
Plot 2D distributions with calculated correlation for each unique combination of passed variabels (`--plotVars`) for each dataset (`--input`) when `--plotCorr` is passed

Example calls
```bash
# 1D distributions for TR, CR and SR dataset of of ttbar FH
python checkInputData.py --input ../file_Run2Dev_v1/preprocData_Run2Dev_TR_v1_ttHad_7J.h5 ../file_Run2Dev_v1/preprocData_Run2Dev_CR_v1_ttHad_7J.h5 ../file_Run2Dev_v1/preprocData_Run2Dev_SR_v1_ttHad_7J.h5 --output minimalTestInputs/test_ttHad --plotVars jets_pt_0 jets_pt_1 jets_eta_0 jets_eta_1 --inputNames ttHad ttHad-CR ttHad-SR --plotDist --normalized
```

## compareTransfromedVariables
Example calls:
```bash
python compareTransfromedVariables.py --input ../minimalTesting/eval/50Epochs/evalDataArrays.pkl ../minimalTesting/eval/SR_50Epochs/evalDataArrays.pkl --inputID TR SR --output ../minimalTesting/eval/transfromComp
```
