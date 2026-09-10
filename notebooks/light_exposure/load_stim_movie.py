# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: eyewire2-figures (3.13.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Load stimuli saved with QDSpy
#
# Requires the `.pickle`, `.npy`, or `.npz` files saved with QDSpy, which are in the [`eyewire2-data`](https://huggingface.co/datasets/eulerlab/eyewire2-data/tree/main) dataset downloaded from huggingface.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

HERE = os.getcwd()
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "utils"))
from data_io import get_data_config, REPO_ROOT
from stim_utils.stimulus import stim_movies

# %%
# Get paths from `data_config.yaml` paths 
DATA_2P = (Path(REPO_ROOT) / "notebooks" / get_data_config()["data_2p_dir"]).resolve()

# Use compressed .npz files for stimuli-as-movies
STIM_MOV_EXT = ".npz"
STIM_MOV_PATH = DATA_2P / "stimuli-as-movies"

# %% [markdown]
# ### Load stimulus movie files into numpy array
#
# ... and zero blue channel, as it was not used (green=G, red=UV?)

# %%
# Load movie files and zero blue channel
tmp_path = Path.joinpath(STIM_MOV_PATH, "DS" +STIM_MOV_EXT)
mov_DS = stim_movies.load_qdspy_movie(tmp_path)
mov_DS[:,:,:,2] = 0

tmp_path = Path.joinpath(STIM_MOV_PATH, "Chirp" +STIM_MOV_EXT)
mov_Chirp = stim_movies.load_qdspy_movie(tmp_path)
mov_Chirp[:,:,:,2] = 0

tmp_path = Path.joinpath(STIM_MOV_PATH, "MouseCam_Left" +STIM_MOV_EXT)
mov_MouseCamLeft = stim_movies.load_qdspy_movie(tmp_path)
mov_MouseCamLeft[:,:,:,2] = 0

# Define spatial and temporal scaling
# (pixel size from moving bar width / bar pixels in movies)
_, dx, dy, _ = mov_DS.shape
px_um = 300 /7
params = dict({
    "pix_size_um": 300 /7,  # moving bar width / bar pixels in movies
    "mov_dxy": [dx, dy],
    "mov_dxy_um": [px_um *dx, px_um *dy],
    "dt_fr_s": 1 /60,
    "nCh": 2
})

# %% [markdown]
# ### Inspect stimulus movie

# %%
_mov = mov_DS # or mov_Chirp or mov_MouseCamLeft

nCh = 3
nFr = _mov.shape[0]
vmin = _mov.min()
vmax = _mov.max()

def show_frame(frame):
    fig, axes = plt.subplots(1, nCh, figsize=(10, 3))
    for ch in range(nCh):
        axes[ch].imshow(_mov[frame,:,:,ch], cmap='gray', vmin=vmin, vmax=vmax)
        axes[ch].axis('off')
        axes[ch].set_title(f'Channel {ch}')
    fig.suptitle(f'Frame {frame}/{nFr-1}')
    plt.tight_layout()
    plt.show()

# Create interactive slider
interact(show_frame, frame=IntSlider(min=0, max=nFr-1, step=1, value=0, description='Frame:'))

# %% [markdown]
# ### Calculate intensity traces for an area within the movie ...
#
# ... using the DS stimulus (`mov_DS`) as example. Range `[0, -1]` means use complete movie, the field for the intensity estimate is centred (`_field_xy_um=[0,0]`) and twice the recording field (`_field_size_um=[95*2.0, 95*2.0]`) 

# %%
DS_intens, DS_intens_cumul = stim_movies.calc_intensity_trace(
    mov_DS, params, _range_s=[0, -1], _field_xy_um=[0,0], _field_size_um=[95*2.0, 95*2.0],
    _plot=True, _verbose=True
)

# %% [markdown]
# ### Flattening a movie ...
#
# ... using the DS stimulus (`mov_DS`) as example.

# %%
mov_flat = stim_movies.flatten_movie(mov_DS, params, _range_s=[0, 25], _plot=True)

