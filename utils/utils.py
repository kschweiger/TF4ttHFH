"""
Collection of util functions (and classes)

K. Schweiger, 2019
"""
import logging
import os
import sys

def initLogging(thisLevel):
    """
    Helper function for setting up python logging
    """
    log_format = ('[%(asctime)s] %(funcName)-20s %(levelname)-8s %(message)s')
    if thisLevel == 20:
        thisLevel = logging.INFO
    elif thisLevel == 10:
        thisLevel = logging.DEBUG
    elif thisLevel == 30:
        thisLevel = logging.WARNING
    elif thisLevel == 40:
        thisLevel = logging.ERROR
    elif thisLevel == 50:
        thisLevel = logging.CRITICAL
    else:
        thisLevel = logging.NOTSET
    
    logging.basicConfig(
        format=log_format,
        level=thisLevel,
    )

    return True

def checkNcreateFolder(path, onlyFolder=False):
    outpath = path.split("/")
    if len(outpath) > 1: #if no / in string output will be saved in current dir
        outpath = "/".join(outpath[0:-1])
        if not os.path.exists(outpath):
            logging.warning("Creating direcotries %s", outpath)
            os.makedirs(outpath)

    if onlyFolder:
        if not os.path.exists(path):
            logging.warning("Creating direcotries %s", path)
            os.makedirs(path)
        
