"""
Classes for the preporcessing step

K.Schweiger, 2019
"""
import logging
import re

#import numpy as np
import pandas as pd
import uproot as root

class Dataset:
    """
    Intermediate object for dataset. This is user exposed and has the exception handling.
    The process() method will start the loop over the files and comverion to HDf5. Before
    running this run:
    1.) addFiles()
    2.) setOutputBranches()


    Args:
      outputName (str) : Name of the output file
      treeName (str) : Name of the ROOT::TTree in the files
    """
    def __init__(self, outputName, outputdir=".", treeName="tree"):
        self.outputName = outputName
        logging.info("Initializing Dataset with output Name: %s", outputName)
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

    def process(self, maxEvents=999999, skipOutput=False):
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
                    logging.warning("Could not open tree %s in file %s. Skipping", self.treeName, f)
                    continue
            df = self.getSelectedDataframe(tree)

            evProcessed += len(df)
            logging.debug("File %s - Processed events %s", iFile, evProcessed)

            if self.outputIndex is not None:
                df.set_index(self.outputIndex, inplace=True)
            #Add to final dataframe
            if outputDF.empty:
                logging.debug("Initial output dataframe")
                outputDF = df
            else:
                outputDF = pd.concat([outputDF, df])

            if evProcessed > maxEvents:
                logging.info("*"*20)
                logging.info("maxEvents reached")
                break

        #Massage output dataframe
        branchesToRemove = list(set(self.branches).difference(set(self.outputBranches)))
        for br in branchesToRemove:
            logging.debug("Will remove branch %s", br)
        #print(branchesToRemove)
        outputDF.drop(columns=branchesToRemove, inplace=True) # Remove unnecessary branches

        #Output
        if not skipOutput:
            self.makeOutput(outputDF)
        else:
            logging.warning("Skipping output")

        return outputDF

    def makeOutput(self, outputDF):
        """ Write DF to .h5 file """
        output = self.outputdir+"/"+self.outputName+".h5"
        logging.info("Creating output file: %s", output)
        with pd.HDFStore(output, "a") as store:
            store.append("data", outputDF, index=False)

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
            logging.debug("Adding file: %s", f)
            if self.branches == self.getBranchesFromFile(f):
                self.files.append(f)
            else:
                raise RuntimeError("All files should have the same branches. Failed for file %s"%f)

    def setOutputBranches(self, branchList):
        """ Sets the branches that should be present in the output file """
        if isinstance(branchList, str):
            branchList = [branchList]
        if not self.filesAdded:
            raise RuntimeError("Set at least one input file (and therefor the valid branches) before running this")

        wildcards = list(filter(lambda x : "*" in x, branchList))

        expandedWildcard = []
        for selector in wildcards:
            branchList.remove(selector)
            if selector == "*":
                expandedWildcard += self.branches
            else:
                expandedWildcard += self._resolveWildcardBranch(selector)

        if expandedWildcard:
            branchList += expandedWildcard

        branchList = list(set(branchList)) # remove duplicated

        for br in branchList:
            logging.debug("Adding output branch: %s", br)
            if br not in self.branches:
                raise KeyError("Branch %s not in tree branches"%br)
            else:
                self.outputBranches.append(br)

        self.outputBranchesSet = True

    def _resolveWildcardBranch(self, selector):
        if not "*" in selector:
            raise RuntimeError("No wildcard - * - in passed selector %s"%selector)

        if selector.count("*") == 1:
            if selector.startswith("*"):
                thisRE = selector.replace("*", "")+"$"
            elif selector.endswith("*"):
                thisRE = "^"+selector.replace("*", "")
            else:
                raise NotImplementedError("Wildcards like a*b not supported")
        else:
            if selector.startswith("*") and selector.endswith("*"):
                thisRE = selector.replace("*", "")
            else:
                raise NotImplementedError("Wildcards like a*b* not supported")

        retList = []
        for br in self.branches:
            if re.search(thisRE, br) is not None:
                retList.append(br)
                print(thisRE, br)

        return retList

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
