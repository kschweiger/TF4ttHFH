import configparser
import logging

class ConfigReaderBase:
    """
    Base class of a config reader
    """
    def __init__(self, pathtoConfig):
        logging.info("Loading config %s", pathtoConfig)
        thisconfig = self.readConfig(pathtoConfig)
        self.readConfig = thisconfig
        self.path = pathtoConfig

    def readConfig(self, pathtoConfig):
        """
        Create and load the configparser object. Move the separate function for unit testability
        """
        thisconfig = configparser.ConfigParser()
        thisconfig.optionxform = str #Use this so the section names keep Uppercase letters
        thisconfig.read(pathtoConfig)

        return thisconfig
        
    def readMulitlineOption(self, section, thisOption, optionType):
        ret = {}
        option = self.readConfig.get(section, thisOption)
        for elem in option.split("\n"):
            if elem == "":
                continue
            if optionType == "Single":
                name, value = elem.split(" : ")
            elif optionType == "List":
                name, value = elem.split(" : ")
                value = self.getList(value)
            else:
                raise RuntimeError
            logging.debug("Found: %s = %s", name, value)
            ret[name] = value

        return ret
    
    @staticmethod
    def getList(value):
        value = value.replace(" ", "") 
        return value.split(",")
