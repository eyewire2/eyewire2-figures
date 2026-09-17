# Light exposure / stimulus history

Moved here from `eyewire2-functional-analysis/scripts/analysis/light_exposure` because the
output figures are used in the paper. These are [jupytext](https://jupytext.readthedocs.io/)
"percent format" scripts — open them in Jupyter/VS Code as notebooks, or run with
`python <script>.py`.

## Scripts

- **`EW2_stim_history.py`** — figure-generating script for this sub-project's own output.
  Reads the consolidated experiment overview (`experiment-overview_consolidated.csv`) and
  plots:
  1. `stimulus_presentation_map.svg` — recording field positions and stimulus outlines
     (moving bar / chirp / mouse-cam), colored by time. Only needs the consolidated CSV.
  2. `exposure_spatial.svg` / `exposure_temporal.svg` — per-field light exposure computed by
     replaying the actual QDSpy stimulus movies, loaded from the shared `eyewire2-data`
     dataset (see "Stimulus movie files" below).
- **`load_recording_log.py`** — preprocessing: parses the raw QDSpy `.log` file and the
  ScanM `.smh` recording headers into `stims.csv` / `smhs.csv` and a recording-position
  figure (`smh_positions_by_time.svg`). Not a dependency of `EW2_stim_history.py` (which
  reads the already-consolidated CSV instead) — this is upstream/exploratory.
- **`load_stim_movie.py`** — exploratory notebook for inspecting a single QDSpy stimulus
  movie (frame viewer, intensity traces). Uses the same shared movie files as
  `EW2_stim_history.py` part 2.
- **`inspect_natural_movie_stimulus.py`** (formerly `NaturalStimuli.py`) — explains the
  natural-movie stimulus format used elsewhere (not in this recording) and how to
  reconstruct the exact sequence shown for a given `scan_sequence_index`. Requires manually
  downloading a stimulus file from Hugging Face into `~/Downloads` (see the notebook's
  markdown) — not part of any repo's data. Its final, optional cell (playing the
  reconstructed movie as a video) additionally requires `uv pip install openretina`, which
  is not a `pyproject.toml` dependency of this repo.

The paper-figure notebook that consumes this sub-project's code, **`../plot_stim_history.py`**
(one level up, in `notebooks/`), regenerates the actual paper panels
(`exposure_spatial.svg`, `exposure_temporal.svg`, `exposure_summary.svg`) using the same
stimulus-movie loading logic as `EW2_stim_history.py`. It writes to `../figures/light_exposure/`
like the rest of this repo's paper-figure notebooks, but unversioned (no `v{version}`
subdirectory) since these figures depend on the manually curated spreadsheets here, not on
`data_config.yaml`'s data version.

## Shared `stim_utils/` code

These scripts originally imported from the `eyewire2_functional_analysis` package
(`eyewire2-functional-analysis/eyewire2_functional_analysis/{stimulus,scanm}/`). Per this
workspace's convention of keeping the `eyewire2-*` repos independently installable, this
repo carries a duplicated (not shared-package) copy of exactly the modules these scripts
need, now at [`../../utils/stim_utils/`](../../utils/stim_utils/) (`stimulus/*` and
`scanm/*`) alongside the rest of `utils/` — it moved there from a `stim_utils/` folder
local to this directory so all notebooks that need it (including `../plot_stim_history.py`)
can reach it via the `utils/` path every notebook already adds to `sys.path`. If the
original modules change in meaningful ways, this copy won't pick that up automatically.

Similarly, `experiment-overview_consolidated.csv` is repo-local metadata in
eyewire2-functional-analysis (not part of the shared `eyewire2-data` Hugging Face download),
so a copy was placed here rather than reaching across repos for it.

## Stimulus movie files

`EW2_stim_history.py` part 2, `load_stim_movie.py`, and `../plot_stim_history.py` all load
the full QDSpy movie-as-numpy-array dumps (`DS`, `Chirp`, `MouseCam`) from
`data-2p/stimuli-as-movies/*.npz` in the shared `eyewire2-data` download (path resolved via
`data_config.yaml`'s `data_2p_dir`, same as every other notebook in this repo). These used
to require manually-placed, non-shared `.pickle` files; that gap is closed now that
compressed `.npz` versions are part of the shared dataset.

## Test results (this repo's `.venv`)

| Script | Result |
|---|---|
| `EW2_stim_history.py` | Runs end-to-end, reproducing `stimulus_presentation_map.svg`, `exposure_spatial.svg`, `exposure_temporal.svg`. |
| `load_recording_log.py` | Runs end-to-end and regenerates `stims.csv`, `smhs.csv`, `smh_positions_by_time.svg`. |
| `load_stim_movie.py` | Runs end-to-end. |
| `inspect_natural_movie_stimulus.py` | Runs through loading and movie-sequence reconstruction; fails only at the final, explicitly-optional video-visualization cell without `openretina` installed. Requires the manual Hugging Face download described in its own markdown. |
| `../plot_stim_history.py` | Runs end-to-end. |
