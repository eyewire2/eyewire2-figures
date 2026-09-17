# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Code that reproduces the figures for the Eyewire II resource paper (preprint:
https://www.biorxiv.org/content/10.64898/2026.05.28.727403v1). Each notebook in
[notebooks/](notebooks/) regenerates one or more figure panels from data pulled out of a
shared, separately-downloaded dataset (`eyewire2-data`).

## Setup

```bash
uv sync
```

Requires Python >=3.13. On Windows, `pywarper` is pinned to a git branch (`windows-fix`) in
`pyproject.toml`'s `[tool.uv.sources]` because the PyPI release crashes on a non-ASCII console
warning under cp1252 — don't "fix" this by pointing it back at PyPI.

### Data dependency

This repo has no local data copy. It reads from a sibling `eyewire2-data` folder (downloaded
from huggingface.co/datasets/eulerlab/eyewire2-data) that is shared with the
`eyewire2-functional-analysis` repo, expected at `../../huggingface/eyewire2-data` relative to
repo root, i.e.:

```
<parent>/
├── eyewire2-figures/            <- this repo
├── eyewire2-functional-analysis/
└── huggingface/eyewire2-data/
```

All paths (and dataset version strings / file prefixes) are declared in
[data_config.yaml](data_config.yaml) at repo root, loaded via `utils/data_io.get_data_config()`
— never hardcode a data path in a notebook, read it from the config. `data_config.yaml` also
pins per-dataset `version`/`file_prefix` fields (e.g. `version`, `version_ribbons`, `version_rb`)
that select which dated parquet snapshot to load; bump these, not the code, when a new data
export lands.

## Running the notebooks

Notebooks are stored as `.py` files in **jupytext "percent" format** (see the `# ---
jupyter: ...` header cell), not `.ipynb` — this keeps diffs reviewable in git. Open them in
Jupyter or VS Code as notebooks, or run directly with `python notebooks/<script>.py`:

```bash
uv run --with jupyter jupyter lab
```

Each notebook follows the same boilerplate: `sys.path.append("../utils")` then `import
data_io`, `data_io.get_data_config()` to load paths/versions, then asserts that the resolved
data dir/file exists before doing any plotting. `utils/style.set_rc_params()` applies the
shared matplotlib style (`utils/paper.mplstyle`) before any figure is drawn.

There is a table in [README.md](README.md) mapping each notebook to the paper figure/panel it
produces — check it before hunting for the right notebook to modify.

## Output layout (figures are committed)

Rendered figures ARE checked into git (unlike raw data). Every notebook writes into
`../figures/<category>/v{version}/` (category matches the notebook's subject: `All`, `BC`,
`RGC`, `examples`, `counts`, `proofreading`, `connectivity`, `calcium_data`), where `version`
comes from `data_config.yaml`. Each figure is saved twice, as `.svg` and `.png` (often
`dpi=600`). Some notebooks also write CSVs into `website/v{version}/` for the public
Eyewire II website. When re-running a notebook after a data version bump, a new
`v{version}` directory is created alongside old ones rather than overwriting them — old
version directories are historical and generally should not be deleted.

## `utils/` module map

Notebooks all `sys.path.append("../utils")` and import directly (no package install) from:

- `data_io.py` — config loading (`get_data_config`, `get_file_path`) and numpy-array
  serialization helpers for round-tripping array-valued columns through parquet
  (`serialize_numpy_arrays`/`restore_numpy_arrays`), plus dtype-aware array/scalar comparison
  (`safe_compare_arrays`).
- `colors.py` — the canonical color palette and `cellclass2color` mapping used across every
  figure; change colors here, not per-notebook, to keep figures consistent.
- `style.py` / `paper.mplstyle` — shared matplotlib rcParams for paper-consistent figures.
- `plot.py`, `plot_cells.py` — the bulk of shared plotting logic (largest modules).
- `skeleton.py` — loading/plotting cell skeletons (`.swc`, via `skeliner`).
- `mosaics.py`, `dendrogram.py`, `embedding.py` — cell-type mosaic, dendrogram, and
  tSNE/embedding plotting (embedding recompute is optional — see notebook flags).
- `labeling.py`, `ew1.py`, `baden16_utils.py`, `data_2p_loader.py` — cell-type labeling
  utilities, Eyewire I comparison data, Baden 2016 reference-dataset helpers, and 2p functional
  data loading, respectively.

## `notebooks/light_exposure/`

A self-contained sub-project (own README) moved here from `eyewire2-functional-analysis`
because its output figures are used in the paper. It vendors a copy of
`eyewire2_functional_analysis`'s `stimulus`/`scanm` modules under `stim_utils/` rather than
depending on that repo — see the workspace-level cross-repo policy below. Two of its four
scripts require large stimulus-movie pickle files that aren't checked into any repo and must be
placed manually; see its README for exactly which outputs are expected to fail without them.

## Cross-repo conventions (from the workspace root CLAUDE.md)

This repo lives inside a multi-repo workspace of independent `eyewire2-*` repos. Before writing
new utility code, check whether equivalent code already exists in a sibling repo (notably
`eyewire2-functional-analysis`). Do not add a shared package dependency between repos to avoid
duplication — each repo must stay independently installable/runnable; duplicate small
utility code instead (as `notebooks/light_exposure/stim_utils/` does).
