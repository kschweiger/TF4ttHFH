"""
Implemenetation of plotting style configuration

K. Schweiger, 2019
"""
import os
import logging

from collections import namedtuple, defaultdict

from utils.ConfigReader import ConfigReaderBase

class CustomDefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory:
            dict.__setitem__(self, key, self.default_factory(key))
            return self[key]
        else:
            defaultdict.__missing__(self, key)


class StyleConfig(ConfigReaderBase):
    """
    Style configuration for plotting
    """
    def __init__(self, path):
        super(StyleConfig, self).__init__(path)

        self.styleDefiniton = namedtuple("styleDefiniton", ["nBins", "binRange", "axisName"])
        self.defaultnBins = self.readConfig.getint("General", "defaultnBins")
        self.defaultbinRange = (self.readConfig.getfloat("General", "defaultRangeMin"),
                                self.readConfig.getfloat("General", "defaultRangeMax"))

        styleFactory = lambda name : self.styleDefiniton(self.defaultnBins, self.defaultbinRange, name)
        
        self.style = CustomDefaultdict(styleFactory)
        
        for section in self.readConfig.sections():
            if section == "General":
                continue
            nBins = self.readConfig.getint(section, "nBins")
            binRangeMin = self.readConfig.getfloat(section, "binRangeMin")
            binRangeMax = self.readConfig.getfloat(section, "binRangeMax")
            axisName = self.readConfig.get(section, "axisName")
            self.style[section] = self.styleDefiniton(nBins, (binRangeMin, binRangeMax), axisName)
