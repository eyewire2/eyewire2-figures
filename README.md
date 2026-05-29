# Content

This repository collects the code that is necessary to reproduce the figures from the Eyewire II resource paper.

# Setup

## Python

Run the following command to setup the python environment:
```
uv sync
```


## Data

Download the data from: https://huggingface.co/datasets/eulerlab/eyewire2-data/tree/main and copy it to [data](./data). Some notebooks will require that you unpack a zip file first (see [data README](./data/README.md)). 

All functional response data can be found [here](https://github.com/eyewire2/eyewire2-functional-analysis).

# Notebooks

Run the notebooks in [notebooks](./notebooks) to create figure panels from the paper. For all analyses and plots related to functional responses, refer to the notebooks in [this folder](https://github.com/eyewire2/eyewire2-functional-analysis/tree/main/notebooks/analysis/manuscript) in our separate repository for functional data.
