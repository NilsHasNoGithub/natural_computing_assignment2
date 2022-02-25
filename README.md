# Repository for assignment 2 Natural Computing

## Preliminaries

<!-- ### Install rust
See: [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install) -->

### Install anaconda (or miniconda)
See: [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html)

### (Recommended) install mamba for faster virtual environment installations
See: [https://github.com/mamba-org/mamba](https://github.com/mamba-org/mamba)

### Install and activate the virtual environment:
```bash
<mamba/conda> env create -f conda_environment.yml && \
conda activate natural_computing
```

## Reproducing results

### Exercise 1
All hyperparameters are defined in the scripts, so results can be reproduced with:
```bash
python ex1.py
```
The output is printed to the terminal.

### Exercise 2
All hyperparameters used to generate results are in the script, except for `num_steps`. Results can be generated using:
```bash
python ex2.py --num-runs 100 && \
python ex2.py --num-runs 1000 && \
python ex2.py --num-runs 10000
```
The resulting plots can be found in the folder `results`

### Exercise 3
All hyperparameters are in the script, the [iris data](https://archive.ics.uci.edu/ml/datasets/iris) is included in the repository. Run:
```bash
python ex3.py
```
Resulting plots can be found in the `results` folder. Quantitization errors are printed in the terminal.
