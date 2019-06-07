import logging

def initLogging(thisLevel):
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
