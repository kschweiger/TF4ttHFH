pyenv virtualenv 3.6.8 TFCPU
pyenv local TFCPU
pip install --upgrade pip
pip install tensorflow==1.13.1 keras==2.2.4 matplotlib pandas awkward pytest pytest-mock pytest-cov tables ConfigParser pylint coloredlogs
pip install uproot uproot-methods
