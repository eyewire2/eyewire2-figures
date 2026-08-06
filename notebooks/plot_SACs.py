# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: eyewire2-figures
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

import numpy as np
from matplotlib import pyplot as plt

# %% [markdown]
# # Data

# %%
# %%time
import sys

sys.path.append("../utils")

# %%
# %%time
import data_2p_loader as data_loader


df_rois, df_fields, df_outline = data_loader.load_all_dfs()
df = data_loader.load_df_rois_morph(df_rois=df_rois)

# %%
# %%time
import data_io

config = data_io.get_data_config()
data_root = config.get("root")
image_root = config.get("image_root")

# %%
df_sacs = df[df['Cell Type'] == 'ON SAC'].copy()

# %%
# All confirmed as "both_strong"
print(df.loc[df['Cell Type'] == 'ON SAC', 'Latest NucID'].values)

# %%
for field, df_field in df_sacs.groupby('field'):
    print(field)
    print(np.sort(df_field['Latest SegID'].values.astype(int)))

# %%
# Version in Overleaf
# GCL0
# [720575940572885792 720575940566710278 720575940559774600
#  720575940562511187 720575940557921829 720575940561027498
#  720575940565165807 720575940559558491 720575940545899781
#  720575940552838981 720575940581510152 720575940576356549
#  720575940569052267 720575940556421607]
# GCL1
# [720575940545926661 720575940572328807 720575940578433226
#  720575940558682534 720575940558094727 720575940563145273
#  720575940547940040 720575940549776224 720575940562796239
#  720575940569689625]
# GCL2
# [720575940579256464 720575940553426083 720575940570227253
#  720575940556765443 720575940567191371 720575940585113878
#  720575940554919918 720575940545311219]
# GCL3
# [720575940568381112 720575940556288487 720575940554749466
#  720575940567864205 720575940567518130 720575940563685101
#  720575940563685101 720575940569910418 720575940570574881]
# GCL4
# [720575940559596123 720575940573708617 720575940553072535
#  720575940567200587 720575940566539482 720575940557124706
#  720575940573958960 720575940560424382 720575940554439194
#  720575940551719027 720575940573959984]

# %% [markdown]
# # Plot

# %%
# %%time
from style import set_rc_params

set_rc_params()

# %%
# %%time
fig_dir = f'../figures/calcium_data'
os.makedirs(fig_dir, exist_ok=True)


# %%
def plot_mean_and_sd(ax, traces, time, color='black', alt_color='dimgray', facealpha=0.2, offset=0.0):
    if traces.shape[0] <= 2:
        ax.plot(time, traces[0] - np.mean(traces[0]) + offset, color=color)
        if len(traces) == 2:
            ax.plot(time, traces[1] - np.mean(traces[1]) + offset, color=alt_color)
    else:
        mu = np.mean(traces, axis=0)
        mu = mu - np.mean(mu) + offset
        sd = np.std(traces, axis=0)

        ax.plot(time, mu, color=color)
        ax.fill_between(time, mu - sd, mu + sd, color=color, alpha=facealpha)

    ax.axis('off')


# %%
def plot_traces(ax, traces, time, dy=-1.5, **kwargs):
    for i, trace in enumerate(traces):
        offset = i * dy
        ax.plot(time, trace + offset, **kwargs)


# %%
from scipy import ndimage

def downsample_uniform_filter(data, n):
    """Downsample using uniform filter - works with any array size"""
    filtered = ndimage.uniform_filter1d(data, size=n, axis=1)
    return filtered[:, ::n]


# %%
chirp_stimulus = np.load(os.path.join(data_root, "stimuli", "global_chirp", "chirp_stimulus.npy"))

# %%
cmap = 'coolwarm'

fig, axs = plt.subplots(3, 5, figsize=(7, 3), height_ratios=(2, 1, 1), sharey='row')

for ax in axs.flat:
    ax.axis('off')

#fig.suptitle('All ON SACs in 2p fields', fontsize=14)
plt.tight_layout(h_pad=0)

nmax = 14
size_full_um = 343

for i in range(5):
    ax = axs[0, i]
    png_path = os.path.join(
        image_root,
        "em-fields",
        f"sacs{i}.png"
    )
    img = plt.imread(png_path)
    ax.imshow(img, extent=(0, 1, 0, 1))
    ax.axis('off')
    ax.text(0, 1, f"F{i}", fontsize=14, ha='left', va='top')
    ax.plot([0, 50 / 343], [+.1] * 2, c='k', solid_capstyle='butt', lw=2, clip_on=False)
    if i == 0:
        ax.text(0, 0.1, '50 µm', c='k', ha='left', rotation=0, va='bottom', fontsize=14)

    df_field = df_sacs[df_sacs.field == f'GCL{i}']
    print(f"Field {i}: VD-dist_um={(df_field.ventral_dorsal_pos_um.max() - df_field.ventral_dorsal_pos_um.min())}")

    t_chirp = np.arange(df_field.chirp_average_norm.iloc[0].size) * df_field.chirp_average_dt.iloc[0]
    t_bar = np.arange(df_field.bar_time_component.iloc[0].size) * df_field.bar_snippets_dt.iloc[0]

    chirps = np.vstack(df_field.chirp_average_norm)
    im_chirps = np.full((14, chirps.shape[1]), np.nan)
    im_chirps[:chirps.shape[0], :] = chirps
    im_chirps = downsample_uniform_filter(im_chirps, n=4)

    ax = axs[1, i]
    ax.plot(np.linspace(t_chirp[0], t_chirp[-1], len(chirp_stimulus)),
            1.1 + 0.1 * chirp_stimulus / np.max(chirp_stimulus), c='k', clip_on=False, lw=1, solid_capstyle='butt')
    ax.set_xlim(t_chirp[0], t_chirp[-1])

    ax.imshow(im_chirps, vmin=-1, vmax=+1, aspect='auto', extent=(t_chirp[0], t_chirp[-1], 0, 1), cmap=cmap,
              interpolation='none')
    ax.set_xlim(t_chirp[0], t_chirp[-1])
    for t in [2, 5, 8, 10, 20.5, 30]:
        ax.plot([t, t], [1 - (chirps.shape[0] / (nmax + 0.5)), 1], c='k', lw=0.8, ls='--')

    if i == 0:
        ax.plot([0, 2], [-.1, -.1], c='k', solid_capstyle='butt', lw=2, clip_on=False)
        ax.plot([-0.05 * t_chirp[-1]] * 2, [0, 5 / nmax], c='k', solid_capstyle='butt', lw=2, clip_on=False)

    ax = axs[2, i]
    bars = np.vstack(df_field.bar_time_component)
    im_bars = np.full((nmax, bars.shape[1]), np.nan)
    im_bars[:bars.shape[0], :] = bars

    ax.plot([t_bar[0], t_bar[-1]], 1.1 + 0.1 * np.array([0, 1]), c=(1, 1, 1, 0), clip_on=False, lw=1,
            solid_capstyle='butt')
    ax.imshow(im_bars, vmin=-1, vmax=+1, aspect='auto', extent=(t_bar[0], t_bar[-1], 0, 1), cmap=cmap,
              interpolation='none')
    ax.set_xlim(t_bar[0], t_bar[-1])
    for t in [1.152, 2.432]:
        ax.plot([t, t], [1 - (chirps.shape[0] / (nmax + 0.5)), 1], c='k', lw=0.8, ls='--')

    if i == 0:
        ax.plot([0, 2], [-.1, -.1], c='k', solid_capstyle='butt', lw=2, clip_on=False)
        ax.text(0, -0.2, '2 s', c='k', ha='left', rotation=0, va='top', fontsize=14)
        ax.plot([-0.05 * t_bar[-1]] * 2, [0, 5 / nmax], c='k', solid_capstyle='butt', lw=2, clip_on=False)
        ax.text(-0.05 * t_bar[-1], 0, f'5 ROIs', c='k', ha='right', rotation=90, va='bottom', fontsize=14)

plt.savefig(f'{fig_dir}/ON_SACs.svg', bbox_inches='tight', dpi=600, transparent=True)

# %%
