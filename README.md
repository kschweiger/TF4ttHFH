![Coverage](https://img.shields.io/badge/pytest--cov-64%25-yellow.svg?longCache=true&style=flat-square)
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

## Training
For the training of the network script(s) are available in the root directory of the repo. For setting the paramters of the net, the input files, output path etc. configuration files are used that are passed as argument when calling the top-level training script.

### Dataprocessing
The proprocessed data is is loaded in the beginning of the training. Currently the default setting for all variables is to be transformed with a so-called "Gauss" method. After transformation all input variables have a mean = 0 and a std. deviation = 1. The mean and std used for the training are stored in the autoencoder class and saved to disc in after training. For ecaluation these values have to be used to transform the input variables.

### Autoencoder
In the config the follow parameters of the network can be set:

- Regularization with weight decay - `useWeightDecay`
- The optimizer used in the training - `optimizer` (currently supported are `rmsprop`, `adamadagrad`)
- Loss function - `loss` (currently supported are `MSE`, `LOGCOSH`, `MEA`) 
- General parameters like: `epochs`, `batchSize`, `validationSplit`
- Default settings for activations can be set with `defaultActivationEncoder` and `defaultActivationDecoder`
- This can be overwritten for the encoder, decoder and each pair of hidden layers
- It is required to set an encoder dimention (the "bottleneck") 
- EarlyStopping by minimum loss can be activated with `doEarlyStopping` and the patience can be set with the `patience` parameter
- If hidden layers are set in the network configuration a section `[HiddenLayer_X]` (where `x` is the hidden startign at 0) for each pair of layers is required in the config. There the dimention has to be set and activation functions can be set
- For each sample set in the general part of the config a section with the same name is expected. It has to set `input`, `label` and `datatype` and can set `xsec` and `nGen`.

Optimizers and loss functions can be "added" by modifiying the `supportedLosses` attribute and `_getLossInstanceFromName` and `setOptimizer` methods in `training/autoencoder/autoencoder.py`.

#### Training
The training for autoencoders is implemented in `train_autoencoder.py`. It only requires a config to be passed with the `--config` flag. For example configs check `data/autoencoder_example.cfg`, `data/autoencoder_example_minimal.cfg` or the test script `test/test_train_autoencoder.py`

The output folder set in the config will be created if not present on the FS and will overwrite existing files if existent. In the output folder the following files will be created:

- Input information like the passed configuration file (`usedConfig.cfg`), applied transformations (`auoencoder_inputTransformation.json`), set attributes of the `Autoencoder` class (`autoencoder_attributes.json`)
- Model summary (`autoencoder_report.txt`)
- Model (`trainedModel.h5py`) and weights (`trainedModel_weights.h5`)
- Plots for all set metrics and the loss per epoch as pdf
- Plots comparing autoencoder input and prediction for each variable as pdf


#### Evaluation
Evaluating a training done with `train_autoencoder.py` can be done with the `eval_autoencoder.py` script. The evaluation script requires some additionl files saved during training. It requires only a config to be passed with the `--config` flag.

### Classification DNN
In the config the follow parameters of the network can be set:

- Regularization with weight decay - `useWeightDecay`
- The optimizer used in the training - `optimizer` (currently supported are `rmsprop`, `adamadagrad`) - Setting the `optimizer` directly and not running `setOptiomizer()` can be used to run all KERAS optimizers by str.
- Loss function - `loss`
- General parameters like: `epochs`, `batchSize`, `validationSplit`
- Activation can be set with `activation` (for all layers except the last one) and `outputActivation`
- EarlyStopping by minimum loss can be activated with `doEarlyStopping` and the patience can be set with the `patience` parameter
- Layers are set with the `layerDimentions` option as comma separted list
- For each sample set in the general part of the config a section with the same name is expected. It has to set `input`, `label` and `datatype` and can set `xsec` and `nGen`.

#### Training
The training for autoencoders is implemented in `train_dnn.py`. It only requires a config to be passed with the `--config` flag. The training can be run in a **binary** or **categorical** mode by setting the `loss` to *binary_crossentropy* or *categorical_crossentropy* as well as setting the output activation accordingly (*sigmoid* vs. *softmax*)

The output folder set in the config will be created if not present on the FS and will overwrite existing files if existent. In the output folder the following files will be created:

- Input information like the passed configuration file (`usedConfig.cfg`), applied transformations (`network_inputTransformation.json`), set attributes of the `DNN` class (`network_attributes.json`)
- Model summary (`network_report.txt`)
- Model (`trainedModel.h5py`) and weights (`trainedModel_weights.h5`)
- Plots for all set metrics and the loss per epoch as pdf
- ROC curve of the DNN ouput and comparisons to other metrics (== input variables) set in the trainin script **(Only in binary classification)**

#### Evaluation
Evaluating a training done with `train_dnn.py` can be done with the `eval_dnn.py` script. The evaluation script requires some additionl files saved during training. It requires only a config to be passed with the `--config` flag. In the config a "signal" sample need to be set in order to calculate ROC curves. When categorical classification was used ROCs will be produced for all samples respective to the set signal samples. Check `data/evalDNN_testing.cfg` and `data/evalDNN_testing_multi.cfg` for example configurations.

## Plotting
Several plot scripts are provided to check input dataframes and training output.   All scripts are expected to be executed from **within** the plotting folder

- For validation of the input dataset `plotting/checkcheckInputData.py` can be used.
- `compareTransfromedVariables.py` can be used to compare the output of multiple runs of `eval_autoencoder.py`. Point with `--input` to the `evalDataArrays.pkl` files in the output folder of the evaluation script. Will produce plot comparing *Input Dataset*, *Predictions* and both for each dataset processed in the eval script. 

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
