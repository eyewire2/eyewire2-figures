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

### Python

Requires Python 3.13. If you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run:

```bash
uv sync
```

Alternatively, set up the environment from [requirements.txt](requirements.txt).

### Data

1. Download the data from [huggingface.co/datasets/eulerlab/eyewire2-data](https://huggingface.co/datasets/eulerlab/eyewire2-data/tree/main).
2. Copy all parquet files to [data/](data/).
3. Unpack `swc-examples.zip` and copy all `.swc` files to [data/swc/](data/swc/). This archive contains all skeletons shown in the figures.
4. *(Optional)* Unpack `swc.zip` to get the skeletons for all cells used in the paper. This is not required to reproduce the figures.

The resulting directory structure should look like:

```
data/
├── swc/
│   ├── 720575940537001651.swc
│   ├── 720575940537038003.swc
│   └── ...
├── df_all_neurons_2026-05-15-15h.parquet
└── ...
```

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
| [plot_rbccircuit_examples.ipynb](notebooks/plot_rbccircuit_examples.ipynb) | Figure 8: Rod bipolar cell circuit cell examples |
| [plot_rbcciruit_fig8_panelD.ipynb](notebooks/plot_rbcciruit_fig8_panelD.ipynb) | Figure 8 panel D: Rod bipolar cell circuit analysis |
| [plot_ribbons_fig8_panelF.ipynb](notebooks/plot_ribbons_fig8_panelF.ipynb) | Figure 8 panel F: ribbons per BC type |
| [plot_ribbons_fig8_panelG.ipynb](notebooks/plot_ribbons_fig8_panelG.ipynb) | Figure 8 panel G: ribbons in the IPL |

For all analyses and plots related to functional responses, refer to the notebooks in [this folder](https://github.com/eyewire2/eyewire2-functional-analysis/tree/main/notebooks/analysis/manuscript) in our separate repository for functional data.
