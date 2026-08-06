# Eyewire II - Figures

Code to reproduce the figures from the Eyewire II resource paper.
For more information about the Eyewire II dataset, see [eyewire.ai](https://eyewire.ai/).

**Preprint:** https://www.biorxiv.org/content/10.64898/2026.05.28.727403v1

**Citation:**
```
Stroeh, S., Ebert, S., Fadjukov, J., Lause, J., Oesterle, J., Franke, K., ... & Eyewire II Consortium. (2026).
Eyewire II-A connectomic resource for resolving cell types and circuits of the mouse retina. bioRxiv, 2026-05.
```

**License:** [MIT](LICENSE)

## Setup

Clone this repository, then follow the steps below.
The code has been tested on Ubuntu 24.04.3 LTS and Windows 11 Pro.

### Python

Requires Python 3.13. If you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run:

```bash
uv sync
```

Alternatively, set up the environment from [requirements.txt](requirements.txt).

With uv, the install time is only a few seconds.

### Data

This repo does not keep its own copy of the data — it reads from a single shared `eyewire2-data` folder that is also used by [eyewire2-functional-analysis](https://github.com/eyewire2/eyewire2-functional-analysis), so both repos stay in sync with one download.

1. Download the data from [huggingface.co/datasets/eulerlab/eyewire2-data](https://huggingface.co/datasets/eulerlab/eyewire2-data/tree/main).
2. Place it in a folder named `eyewire2-data`. By default, [data_config.yaml](data_config.yaml) expects it at `../huggingface/eyewire2-data` relative to this repo's root — i.e. inside a `huggingface/` folder that sits next to this repo (and next to `eyewire2-functional-analysis`), laid out like:

    ```text
    <parent>/
    ├── eyewire2-figures/            <- this repo
    ├── eyewire2-functional-analysis/
    └── huggingface/
        └── eyewire2-data/
            ├── data-2p/
            ├── images/
            ├── spreadsheets/
            ├── swc/
            └── ...
    ```

3. Unpack `track_proofreading.zip` to `eyewire2-data/track_proofreading/`.
4. Unpack `swc-examples.zip` and copy all `.swc` files to `eyewire2-data/swc/`. This archive contains all skeletons shown in the figures.
5. *(Optional)* Unpack `swc.zip` to get the skeletons for all cells used in the paper. This is not required to reproduce the figures.
6. Copy the `df_all_neurons_*.parquet` (and any other top-level analysis parquet files) into `eyewire2-data/` directly.

If your `eyewire2-data` folder lives somewhere else, update the paths in [data_config.yaml](data_config.yaml) (at the repo root) to match — every notebook loads its data paths from there via `data_io.get_data_config()`.

## Notebooks

The notebooks in [notebooks/](notebooks/) reproduce the figure panels from the paper. Start Jupyter with:

```bash
uv run --with jupyter jupyter lab
```

| Notebook | Figure |
|---|---|
| [plot_examples.ipynb](notebooks/plot_examples.ipynb) | Figure 1: Example cell morphologies |
| [plot_proofreading-stats.ipynb](notebooks/plot_proofreading-stats.ipynb) | Figure 2: Proofreading statistics |
| [plot_cell_counts.ipynb](notebooks/plot_cell_counts.ipynb) | Figure 2: Cell type counts and statistics |
| [plot_cellclass_embedding.ipynb](notebooks/plot_cellclass_embedding.ipynb) | Figure 3: Examples and embedding of all neurons |
| [plot_BCs.ipynb](notebooks/plot_BCs.ipynb) | Figure 4: Bipolar cell types: examples, embeddings and mosaics |
| [plot_RGCs.ipynb](notebooks/plot_RGCs.ipynb) | Figure 5: Retinal ganglion cell types: examples and mosaics |
| [plot_RGC_mapping.ipynb](notebooks/plot_RGC_mapping.ipynb) | Figure 6 panel A-G: Mapping from EM to 2p of example RGC |
| [plot_RGC_examples.ipynb](notebooks/plot_RGC_examples.ipynb) | Figure 6 panel H, I: Retinal ganglion cell example type responses |
| [plot_response_overview.ipynb](notebooks/plot_response_overview.ipynb) | Figure 6 panel J: Overview of all functional responses |
| [plot_SACs.ipynb](notebooks/plot_SACs.ipynb) | Figure 7 A, B: Starburst amacrine cells: examples and mosaics |
| [plot_rbccircuit_examples.ipynb](notebooks/plot_rbccircuit_examples.ipynb) | Figure 8: Rod bipolar cell circuit cell examples |
| [plot_rbcciruit_fig8_panelD.ipynb](notebooks/plot_rbcciruit_fig8_panelD.ipynb) | Figure 8 panel D: Rod bipolar cell circuit analysis |
| [plot_ribbons_fig8_panelF.ipynb](notebooks/plot_ribbons_fig8_panelF.ipynb) | Figure 8 panel F: ribbons per BC type |
| [plot_ribbons_fig8_panelG.ipynb](notebooks/plot_ribbons_fig8_panelG.ipynb) | Figure 8 panel G: ribbons in the IPL |


For all deeper analyses of functional responses, refer to the notebooks in [this folder](https://github.com/eyewire2/eyewire2-functional-analysis/tree/main/notebooks/analysis/) in our separate repository for functional data.
