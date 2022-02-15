# CRITICAL
Plasticity in a spiking Reservoir with a regulation based on critical branching factors. Also inspired from astrocytes, illustration on speech.

## Paper to cite
Simon Brodeur and Jean Rouat, Regulation toward Self-Organized Criticality in a Recurrent Spiking Neural Reservoir, int. Conf. on Artificial Neural Networks and Machine Learning, pp. 547-554, 2012
http://dx.doi.org/10.1007/978-3-642-33269-2_69,
preprint: https://www.gel.usherbrooke.ca/rouat/publications/paper_ICANN_2012SimonBrodeurJeanRouatSherbrooke.pdf

![Capture d’écran, le 2022-02-15 à 16 44 28](https://user-images.githubusercontent.com/16075468/154154553-7c1daba5-6608-42b8-848e-093e134111b8.png)


## Dependencies

Main requirements:
- Python 3.6 wit Numpy, Scipy and Matplotlib
- Brian2

## Setup Anaconda environment
```
conda create -n critical python=3.6 pip
source activate critical

conda install matplotlib nose
conda install -c conda-forge brian2
```

## Downloading the code

Download the source code from the git repository:
```
mkdir -p $HOME/work
cd $HOME/work
git clone https://github.com/NECOTIS/CRITICAL.git
```

## Running unit tests

Make sure the project is in the Python path:
```
export PYTHONPATH=$HOME/work/CRITICAL:$HOME/work/CRITICAL/test:$PYTHONPATH
```
This can also be added at the end of $HOME/.bashrc

To ensure all libraries where correctly installed, it is advised to run the test suite:
```
cd $HOME/work/CRITICAL/test
./run_tests.sh
```
Note that this can take a little while.

## Running experiments

Make sure the project is in the Python path:
```
export PYTHONPATH=$HOME/work/CRITICAL:$HOME/work/CRITICAL/test:$PYTHONPATH
```
This can also be added at the end of $HOME/.bashrc

```
cd $HOME/work/CRITICAL/scripts/experiments/poisson
python main_poisson_input.py
```
