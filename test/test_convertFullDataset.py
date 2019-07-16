"""
Test for the classes used in the preprocessing scripts

K. Schweiger, 2019
"""
import sys
import os
from types import SimpleNamespace
import configparser
import argparse

import uproot as root
import pandas as pd
import copy

import convertFullDataSet

import pytest

def test_argParse():
    args = convertFullDataSet.parseArgs(["--output","path/to/output","--input","Input1, Input2", "--name", "NameOfOutputFile" ])
    assert isinstance(args, argparse.Namespace)
