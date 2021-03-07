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

### How to run the Labeler

To run the labeler locally, additionally the following needs to be done (need to restart the JupyterLab server afterwards):

6. Enable Jupyter Widgets: `jupyter labextension install @jupyter-widgets/jupyterlab-manager`
7. Load the spacy language model: `python -m spacy download en_core_web_md`

## Data

Raw annual reports, sustainability reports and if available 20-F's of the Euro STOXX 50 for the years 1999-2019 can be [found here](https://drive.google.com/drive/u/0/folders/1wn8nY1QkkquzRzYjb58SdApM0JKD5xi5).

## Docker for containerization

- Install Docker Desktop [Install Docker](https://www.docker.com/get-started)
- The `docker-compose.yml` in the root directory contains the container definitions and sets up networking
- The `Dockerfile` in the `./data` directory contains the container config for the pdf mining tasks
- Start tmux:
  `tmux` or `tmux new -s myname`
- Connect to container: `docker ps` and `docker exec -it pdf-mining bash`
- Name the session accordingly by first send the prefix `Ctrl` + `b` and then `$`
- Start long running process and leave/detach the session with prefix `Ctrl` + `b` and then `d`

- Later, when you want to attach to the session:
  `tmux list-sessions`
  `tmux attach-session -t 0`

## Cheat Sheet:

- Update environment file if a package was added: `conda env export -f environment.yml --no-builds` or `conda env export --no-builds > environment.yml`. NOTE: This will overwrite everything and adds unecessary packages, making cross-platform compatibility difficult. So better to update manually...

## Additional Resources:

- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf)
