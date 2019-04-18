# CRITICAL
Spiking Reservoir with a critical branching factor

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
