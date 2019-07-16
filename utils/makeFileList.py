import sys
import os
import glob
import logging

from utils.utils import initLogging, checkNcreateFolder

def main(base, output):
    logging.info("Got base: %s", base)
    logging.info("Got output: %s", output)

    checkNcreateFolder(output, onlyFolder=True)
    
    for folder in glob.glob(base+"/*"):
        foldername = folder.split("/")[-1]
        outName = "{0}/{1}.txt".format(output, foldername)
        logging.info("Will create file: %s", outName)
        with open(outName, "w") as f:
            for _file in glob.glob(folder+"/*.root"):
                logging.debug("Found file %s", _file)
                f.write(_file+"\n")

if __name__ == "__main__":
    basePath = sys.argv[1]
    outPath = sys.argv[2]
    initLogging(20)
    main(base = basePath, output = outPath)
