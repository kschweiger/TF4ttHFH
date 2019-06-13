![Coverage](https://img.shields.io/badge/pytest--cov-83%25-yellow.svg?longCache=true&style=flat-square)
![Tests](https://img.shields.io/badge/Test%20passing-true-green.svg?longCache=true&style=flat-square)
# TF4ttHFH

Framework for classification (using TensorFlow) for ttH(bb) in the fully-hadronic channel


# Prequisites
This framework is designed to be **indepdent** on CMSSW and ROOT (except IO). To ensure operation on systems w/o root access it is assumed that **pyenv** is setup.

An informative guide can be found e.g. [here](https://realpython.com/intro-to-pyenv)

## tl;dr (and some additional tricks :wink:): 
Installing is best doen using the [pyenv-installer-project](https://github.com/pyenv/pyenv-installer):

```bash
curl https://pyenv.run | bash
```

After the setup the following need to be added to the file loaded on loging (`.bash_profile` etc.)

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi
```

This can be either be loaded when needed (by putting it in a function) or loaded on default. Initializing CMSSW afterwards should be no problem because it overwrites the path variables and the CMSSW-python version will be used

After that one can install multiple **independent** python version with their **own packages** etc. In principle python version are insalled with 

```bash
pyenv install X.X.X
```

If this is setup on a shared system it is possible that one can not write to the `/tmp` folder. In this case, a different tmp folder can be set at installation time with:
(using `PYTHON_CONFIGURE_OPTS` also might be necessary and should not hurt)

```bash
env PYTHON_CONFIGURE_OPTS="--enable-shared" TMPDIR="/path/to/custom/tmp" pyenv install X.X.X
```

# Setup

Run `setup.sh` for automated setup after cloning the repo. (See special instructions for T3@PSI)

Running 
```bash
soruce init.sh
```
is required every time.

To varify that the installation works run 
```bash
python test/test_keras.py
```

## Important note for cluster use
Since TensorFlow will not be built from source is requires glibc > 1.17. This is not the case on all cluster installation (e.g. SL6!). One can check this with `ldd --version`. On SL7 the version is new enough!       
Works on: *lxplus7* or t3ui07      
Does **not** work on: *lxplus6* or *t3ui01..03*

### READ if running on T3@PSI
GPU operation on T3@PSI is currently only working with anaconda (since it get shipped with required libs). Use the setup `setup_t3PSI.sh` in this case. This will install 
- Anaconda 2018.12 and create venv (and downgrade to python 3.6.8)
- A CPU `TFCPU` and GPU `TFGPU` enabled venv will be initialized with TF and keras (only the former will install all packages required for preprocessing)

## Current versions

| package    | version |
| ---------- | ------- |
| python     | 3.6.8   |
| TensorFlow | 1.13.1  |
| KERAS      | 2.2.4   |


# Operating the framework

## Preprocessing
Since all processing (aside form this) is done within CMSSW the first step is preprocessing a **flat** ROOT::TTree (see note). To be independent on ROOT the uproot package is used. 
The Dataset class implements the routines required to convert the flat tree to a .h5 file that can be used in the training. Using the Dataset.outputBranches paramter will set the branches that are written to the output file. In order to make further selection used the Dataset.selection and  Dataset.sampleSelection. All variables in the selection are required to be present in the input tree and normal binary operations (and, or) can be used (check pandas.dataframe.query for more information).    
Run the preprocessing with the `convertTree.py` script. It requires a configuration file defining input files, ouput paramters and selections. See `data/testPreprocessor.cfg` for an example.

### Note
Flattening of ROOT::TTrees or adding of variables is not part of the proprocesser on purpos in order to avoid dependencies on ROOT itself. The main motivation for this is to have the possibility to used python/TF/Keras version that are not native to the root/CMSSW environment on a cluster.      
In an environment with root access it is not problem to compile a root version against the python version used for TF. Example instructions for this can be found [here](https://gitlab.cern.ch/koschwei/Documentation/blob/master/Notes/04_root.md#43-compiling-on-macos-with-pyenv)(CMS-internal link)      
For the ttH(FH) UZH group a flattener can be found [here](https://gitlab.cern.ch/Zurich_ttH/FHDownstream/blob/FullRunII_dev/nTupleProcessing/classification/flatNprocess.py) (CMS-internal link)

## The SLURM cluster (T3@PSI)
In order to run on GPUs from the T3@PSI the SLURM cluser needs to be used. See [README](slurm/README.md) for instructions and tests.

# Mics

## Tests
All tests are located in the test folder. Run tests with
```bash
python -m pytest test/
```
or run coverage with
```bash
python -m pytest --cov-report=html --cov=. test/
```

## Linting
Style defintions are located in .pylintrc. Run with
```bash
python -m pylint [folder/file]
```
