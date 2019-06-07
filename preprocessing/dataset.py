"""
Classes for the preporcessing step

K.Schweiger, 2019
"""
import numpy as np
import pandas as pd
import uproot as root

import logging


class Dataset:
    """
    Intermediate object for dataset. This is user exposed and has the exception handling
    
    Args:
      outputName (str) : Name of the output file
      treeName (str) : Name of the ROOT::TTree in the files
    """
    def __init__(self, outputName, outputdir=".", treeName="tree"):
        self.outputName = outputName
        self.outputdir = outputdir
        self.treeName = treeName
        self.encoding = "UTF-8"

        self.selection = ""
        self.sampleSelection = ""
        
        #Input 
        self.filesAdded = False
        self.files = None
        self.branches = []

        #Output
        self.outputBranchesSet = False
        self.outputBranches = []
        self.outputIndex = None

    def process(self, maxEvents=999999, skipOutput = False):
        """
        Function that start the processing of all files passed.

        Raises:
          RuntimeError : If files are not set or output branches not set
        """        
        if not self.filesAdded:
            raise RuntimeError("Files not set")
        if not self.outputBranchesSet:
            raise RuntimeError("Output branches not set")

        outputDF = pd.DataFrame()

        
        evProcessed = 0
        for iFile, f in enumerate(self.files):
            logging.info("Processing file %s", iFile)
            with root.open(f) as rFile:
                #print(rFile)
                try:
                    tree = rFile[self.treeName]
                except:
                    logging.warning("Could not open tree %s in file %s. Skipping",self.treeName, f)
                    continue
            df = self.getSelectedDataframe(tree)

            evProcessed += len(df)
            if self.outputIndex is not None:
                df.set_index(self.outputIndex, inplace=True)
            #Add to final dataframe
            if outputDF.empty:
                outputDF = df
            else:
                outputDF = pd.concat([outputDF,df])
                
            if evProcessed > maxEvents:
                logging.info("*"*30)
                logging.info("maxEvents reached")
                break

        #Massage output dataframe
        branchesToRemove = list(set(self.branches).difference(set(self.outputBranches)))
        for br in branchesToRemove:
            logging.debug("Will remove branch %s", br)
        #print(branchesToRemove)
        outputDF.drop(columns=branchesToRemove, inplace=True) # Remove unnecessary branches
        
        #Output
        if skipOutput:
            self.makeOutput(outputDF)

        return outputDF

    def makeOutput(self, outputDF):
        output = self.outputdir+"/"+self.outputName+".h5"
        logging.info("Creating output file: %s", output)
        with pd.HDFStore(output, "a") as store:
            store.append("data", outputDF, index = False)
        
    def getBranchesFromFile(self, infile):
        """ Opens tree in a file with uproot and returns the branches (as strings!) """
        tree = root.open(infile)[self.treeName]
        return [br.decode(self.encoding) for br in tree.keys()]
        
    def addFiles(self, fileList):
        """ Function for adding files. For each file it is checked if the branches are the same in all """
        if not self.filesAdded:
            self.branches = self.getBranchesFromFile(fileList[0])
            self.files = []
            self.filesAdded = True
            
        for f in fileList:
            if self.branches == self.getBranchesFromFile(f):
                self.files.append(f)
            else:
                raise RuntimeError("All files should have the same branches. Failed for file %s"%f)

    def setOutputBranches(self, branchList):
        """ Sets the branches that should be present in the output file """
        if not self.filesAdded:
            raise RuntimeError("Set at least one input file (and therefor the valid branches) before running this")
        
        for br in branchList:
            if br not in self.branches:
                raise KeyError("Branch %s not in tree branches"%br)
            else:
                self.outputBranches.append(br)

        self.outputBranchesSet = True

    def getSelectedDataframe(self, tree):
        """ Function for getting the tree entries as dataframe and apply the selection """
        fullDF = tree.pandas.df()
        logging.info("Entries in dataframe: %s", len(fullDF))
        if self.sampleSelection != "":
            selectedDF = fullDF.query(self.sampleSelection)
            logging.info("Entries in dataframe after sample selection: %s", len(selectedDF))
        else:
            selectedDF = fullDF
            logging.info("No sample selection applied")

        if self.selection != "":
            selectedDF = selectedDF.query(self.selection)
            logging.info("Entries in dataframe after selection: %s", len(selectedDF))
        else:
            logging.info("No sample selection applied")
        
        return selectedDF

    
