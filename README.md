# Financial disclosures

## Requirements:

- Python>=3.4
- miniconda/Anaconda

## How to get started

It is recommended to use _miniconda_ (or `conda`) as opposed to Python only default `pip` as library dependecy manager and to use the local environment of this repository for reproducibility.

1. [Install miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) if `conda` is not yet installed on the system through Anaconda
2. Setup local environment: `conda env create --prefix ./envs -f environment.yml`
3. Activate the project environment by running `conda activate ./envs`
4. Start JupyterLab server: `jupyter lab`
5. To deactivate the environment, run `conda deactivate`

## Cheat Sheet:

- Update environment file if a package was added: `conda env export -f environment.yml`
- Update environment: `conda env update --prefix ./env --file environment.yml --prune`

## Additional Resources:

- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf)
