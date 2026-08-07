# Light exposure / stimulus history

Moved here from `eyewire2-functional-analysis/scripts/analysis/light_exposure` because the
output figures are used in the paper. These are [jupytext](https://jupytext.readthedocs.io/)
"percent format" scripts — open them in Jupyter/VS Code as notebooks, or run with
`python <script>.py`.

## Scripts

- **`EW2_stim_history.py`** — the main figure-generating script. Reads the consolidated
  experiment overview (`experiment-overview_consolidated.csv`) and plots:
  1. `stimulus_presentation_map.pdf` — recording field positions and stimulus outlines
     (moving bar / chirp / mouse-cam), colored by time. Only needs the consolidated CSV.
  2. `exposure_spatial.svg` / `exposure_temporal.svg` — per-field light exposure computed by
     replaying the actual QDSpy stimulus movies. Needs the full movie-as-pickle files (see
     "Known gap" below).
- **`load_recording_log.py`** — preprocessing: parses the raw QDSpy `.log` file and the
  ScanM `.smh` recording headers into `stims.csv` / `smhs.csv` and a recording-position
  figure (`smh_positions_by_time.pdf`). Not a dependency of `EW2_stim_history.py` (which
  reads the already-consolidated CSV instead) — this is upstream/exploratory.
- **`load_stim_movie.py`** — exploratory notebook for inspecting a single QDSpy stimulus
  movie (frame viewer, intensity traces). Needs the same movie-as-pickle files as
  `EW2_stim_history.py` part 2.
- **`NaturalStimuli.py`** — explains the natural-movie stimulus format used elsewhere (not
  in this recording) and how to reconstruct the exact sequence shown for a given
  `scan_sequence_index`. Requires manually downloading a stimulus file from Hugging Face
  into `~/Downloads` (see the notebook's markdown) — not part of any repo's data.

## Local copies instead of a cross-repo dependency

These scripts originally imported from the `eyewire2_functional_analysis` package
(`eyewire2-functional-analysis/eyewire2_functional_analysis/{stimulus,scanm}/`). Per this
workspace's convention of keeping the `eyewire2-*` repos independently installable,
`stim_utils/` here is a duplicated (not shared-package) copy of exactly the modules these
scripts need — `stim_utils/stimulus/*` and `stim_utils/scanm/*`. If the original modules
change in meaningful ways, these copies won't pick that up automatically.

Similarly, `experiment-overview_consolidated.csv` is repo-local metadata in
eyewire2-functional-analysis (not part of the shared `eyewire2-data` Hugging Face download),
so a copy was placed here rather than reaching across repos for it.

## Test results (this repo's `.venv`)

| Script | Result |
|---|---|
| `EW2_stim_history.py` | Part 1 (`stimulus_presentation_map.pdf`) runs and reproduces the figure. Part 2 fails with `FileNotFoundError` on the movie pickles — expected, see below. |
| `load_recording_log.py` | Runs end-to-end and regenerates `stims.csv`, `smhs.csv`, `smh_positions_by_time.pdf` — the raw `.log`/`.smh` files turned out to be part of the shared `eyewire2-data` download after all. |
| `load_stim_movie.py` | Fails with `FileNotFoundError` on the movie pickles — expected. |
| `NaturalStimuli.py` | Fails at the `assert os.path.exists(...)` check — expected, needs the manual Hugging Face download. |

**Known gap:** `EW2_stim_history.py` part 2 and `load_stim_movie.py` both load
`stimuli-as-movies/{RGC_MovingBar,RGC_Chirp,MouseCam_Left}.pickle` — full QDSpy
movie-as-numpy-array dumps. These aren't in the shared `eyewire2-data` download (which has
same-named but much smaller pickles under `data-2p/stimuli/QDSpy/` that turned out to hold
only a single int each, not movie arrays — not a drop-in replacement) and aren't checked
into either repo, presumably because of file size. `STIM_MOV_PATH` in both scripts points at
a `stimuli-as-movies/` folder next to the script; drop the real pickles there if you get
hold of them.
