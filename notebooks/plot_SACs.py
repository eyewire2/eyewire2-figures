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
import skeliner as sk

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
skel_dir = config.get("skel_dir")

# %%
df_sacs = df[df['Cell Type'] == 'ON SAC'].copy()
df_sacs['swc_path'] = df_sacs['Latest SegID'].apply(lambda x: os.path.join(skel_dir, f"{int(x)}.swc"))

# %%
try:
    sys.path.append("../dev")
    from skel_sync import sync_skeletons
    sync_skeletons(df_sacs['swc_path'], skel_dir)
except ImportError:
    pass  # dev-only helper, not present outside this machine; swc-examples.zip should already cover this

# %%
df_sacs['has_swc'] = df_sacs['swc_path'].apply(os.path.isfile)

missing_seg_ids = np.sort(df_sacs.loc[~df_sacs['has_swc'], 'Latest SegID'].astype(int).unique())
print(f"Missing {len(missing_seg_ids)}/{df_sacs['Latest SegID'].nunique()} SWC files for ON SACs, add these to {skel_dir}:")
print(missing_seg_ids)

# %%
# All confirmed as "both_strong"
print(df.loc[df['Cell Type'] == 'ON SAC', 'Latest NucID'].values)

# %%
for field, df_field in df_sacs.groupby('field'):
    print(field)
    print(np.sort(df_field['Latest SegID'].values.astype(int)))

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
chirp_stimulus = np.load(os.path.join(data_root, "data-2p", "stimuli", "global_chirp", "chirp_stimulus.npy"))

# %%
from matplotlib.collections import LineCollection

tab20 = plt.get_cmap('tab20')


def plot_skel_thin(ax, skel, soma_color, dendrite_color, lw=0.4, alpha=0.9, soma_size=20):
    """Plot a skeleton's edges as thin lines (no radius) plus a soma marker."""
    xy = skel.nodes[:, :2]
    segments = xy[skel.edges]
    ax.add_collection(LineCollection(segments, colors=dendrite_color, linewidths=lw, alpha=alpha, zorder=2,
                                      rasterized=True))
    ax.scatter(*skel.soma.center[:2], s=soma_size, color=soma_color, edgecolors='none', zorder=5, rasterized=True)


# %%
cmap = 'coolwarm'

fig, axs = plt.subplots(3, 5, figsize=(7, 3), height_ratios=(2, 1, 1))

# Row 0 (skeletons) needs its own y-range per field (real, differing spatial coords).
# Rows 1-2 (chirp/bar heatmaps) still share a y-range across fields, as before.
for col in range(1, 5):
    axs[1, col].sharey(axs[1, 0])
    axs[2, col].sharey(axs[2, 0])

for ax in axs.flat:
    ax.axis('off')

#fig.suptitle('All ON SACs in 2p fields', fontsize=14)
plt.tight_layout(h_pad=0)

nmax = 14
size_full_um = 343

for i in range(5):
    ax = axs[0, i]

    df_field = df_sacs[df_sacs.field == f'GCL{i}']
    skels = [sk.io.load_swc(p) for p in df_field.loc[df_field['has_swc'], 'swc_path']]

    # Keep cells at their real spatial position within the field (no recentering per cell),
    # just center the view on the field's cells so it lines up with the old EM-field crop.
    centers = np.array([skel.soma.center[:2] for skel in skels]) if skels else np.zeros((0, 2))
    field_center = centers.mean(axis=0) if len(centers) else np.zeros(2)
    xlim = (field_center[0] - size_full_um / 2, field_center[0] + size_full_um / 2)
    ylim = (field_center[1] - size_full_um / 2, field_center[1] + size_full_um / 2)

    for j, skel in enumerate(skels):
        soma_color = tab20(2 * (j % 10))
        dendrite_color = tab20(2 * (j % 10) + 1)
        plot_skel_thin(ax, skel, soma_color=soma_color, dendrite_color=dendrite_color)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0, 1, f"F{i}", fontsize=14, ha='left', va='top', transform=ax.transAxes)
    ax.plot([0, 50 / size_full_um], [+.1] * 2, c='k', solid_capstyle='butt', lw=2, clip_on=False,
            transform=ax.transAxes)
    if i == 0:
        ax.text(0, 0.1, '50 µm', c='k', ha='left', rotation=0, va='bottom', fontsize=14, transform=ax.transAxes)
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
