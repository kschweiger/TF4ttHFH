pyenv install anaconda3-2018.12
mkdir /work/$USER/.pyenv.ext
cd ~/.pyenv/versions/
#This cna now take a while since anaconda rather large
mv anaconda3-2018.12 /work/$USER/.pyenv.ext/
## wait....
ln -s /work/$USER/.pyenv.ext/anaconda3-2018.12/ anaconda3-2018.12
pyenv virtualenv anaconda3-2018.12 TFGPU
source activate TFGPU
# Requires input - Downgrade python <= 3.7.1 in order to run on T3 
conda install python=3.6.8 # 3.7.1
# Requires input
conda install keras tensorflow-gpu scikit-learn
pip install matplotlib pandas awkward pytest pytest-mock pytest-cov tables ConfigParser pylint uproot uproot-methods scikit-learn pydot coloredlogs
pyenv virtualenv anaconda3-2018.12 TFCPU
source activate TFCPU
# Requires input - Downgrade python <= 3.7.1 in order to run on T3 
conda install python=3.6.8 # 3.7.1
# Requires input
conda install keras tensorflow
pip install matplotlib pandas awkward pytest pytest-mock pytest-cov tables ConfigParser pylint uproot uproot-methods scikit-learn pydot coloredlogs
pyenv local TFCPU

