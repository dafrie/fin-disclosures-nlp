FROM continuumio/miniconda3
COPY environment.yml /code/
WORKDIR /code
RUN conda config --add channels conda-forge && conda update --all && conda env create -f environment.yml

# Make subsequent run commands use the new environment
# SHELL ["conda", "run", "-n", "fin-disclosures-nlp", "/bin/bash", "-c"]
ENV PATH /opt/conda/envs/fin-disclosures-nlp/bin:$PATH
RUN python -m spacy download en_core_web_md

# Copy over rest of the files. This is needed to allow pip
# TODO: Replace with sudo apt install git + git pull
COPY data /code/
# CMD python pipeline.py /files