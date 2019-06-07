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

Run `setup.sh` for automated setup after cloning the repo.

Running 
```bash
export KERAS_BACKEND=tensorflow
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


## Current versions

| package    | version |
| ---------- | ------- |
| python     | 3.6.8   |
| TensorFlow | 1.13.1  |
| KERAS      | 2.2.4   |


# Operating the framework

## Preprocessing
Since all processing (aside form this) is done within CMSSW the first step is preprocessing a **flat** ROOT::TTree. To be independent on ROOT the uproot package is used
