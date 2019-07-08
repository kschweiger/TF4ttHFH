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
        
    def readMulitlineOption(self, section, thisOption, optionType, sep=" : "):
        ret = {}
        option = self.readConfig.get(section, thisOption)
        for elem in option.split("\n"):
            if elem == "":
                continue
            if optionType == "Single":
                name, value = elem.split(sep)
            elif optionType == "List":
                name, value = elem.split(sep)
                value = self.getList(value)
            else:
                raise RuntimeError
            logging.debug("Found: %s = %s", name, value)
            ret[name] = value

        return ret

    def setOptionWithDefault(self, section, option, default, getterType="str"):
        if self.readConfig.has_option(section, option):
            if getterType == "float":
                return self.readConfig.getfloat(section, option)
            elif getterType == "int":
                return self.readConfig.getint(section, option)
            elif getterType == "bool":
                return self.readConfig.getboolean(section, option)
            elif getterType == "intlist":
                return [int(x) for x in self.getList(self.readConfig.get(section, option))]
            else:
                if default is None and self.readConfig.get(section, option) == "None":
                    return None
                else:
                    return self.readConfig.get(section, option)
        else:
            return default

    
    @staticmethod
    def getList(value):
        value = value.replace(" ", "") 
        return value.split(",")
