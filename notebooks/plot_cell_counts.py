# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
# %%time
# # %load_ext autoreload
# # %autoreload 2

# %%
# %%time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
# %%time
import sys

sys.path.append("../utils")

import data_io
from ew1 import get_df_ew1
from labeling import clean_labels

# %% [markdown]
# # Data

# %% [markdown]
# ## EW2

# %%
# %%time
config = data_io.get_data_config()
version = config.get("version", "unknown_version")
file_path = data_io.get_file_path(config)

df2 = pd.read_parquet(file_path)
# df2 = data_io.restore_numpy_arrays(df)

print(df2.shape)

# %% [markdown]
# ## EW1

# %%
# %%time
df1_labels = get_df_ew1(add_categories=False, exclude_types=())

file_path_mapper = os.path.join(config["spreadsheet_dir"], 'Eyewire II Proofread Cells Main List - Cell types and properties 2026-01-23.csv')
df1_labels = clean_labels(
    df1_labels, file_path_mapper=file_path_mapper, celltype_col='ew1_label', remove_suffixes=True, shorten_sbc=True).copy()
df1_labels['ew'] = 1
df1_labels.head()

# %% [markdown]
# # Figures

# %%
# %%time
from style import set_rc_params

set_rc_params()

# %%
# %%time
print(version)

fig_dir = f'../figures/counts/v{version}'
os.makedirs(fig_dir, exist_ok=True)

# %% [markdown]
# ## Size comparison

# %%
# %%time
sizes_all = np.concatenate([
    df2[df2['cellclass_final'].isin(['RGC', 'AC'])].hull_diameter.values,
])


# %%
# %%time
def make_vline(x, label, ax, line_kws=None, text_kws=None):
    line_kws = dict() if line_kws is None else line_kws
    text_kws = dict() if text_kws is None else text_kws
    ax.axvline(x, label=label, **line_kws)
    ax.text(x, 1.05, label, transform=ax.get_xaxis_transform(),
            ha='left', va='bottom', rotation=60, **text_kws)


# %%
# %%time
i_largest_rgc = '720575940551276071'
df2.loc[i_largest_rgc].name

# %%
# %%time
df2.loc[i_largest_rgc].celltype_final

# %%
# %%time
i_largest_ac = df2[(df2['cellclass_final'] == 'AC') & (~df2.post_has_axon) & (df2.status.isin(['Complete', 'Complete (cutoff)']))].hull_diameter.idxmax()
df2.loc[i_largest_ac].name

# %%
# %%time
df2.loc[i_largest_ac].celltype_final

# %%
# %%time
line_kws = dict(c='k')
text_kws = dict(fontsize=8)

ew1_max = 332.035 # (ConvexHull & "ew=1").fetch('hull_diameter').max()

avg_sac = df2[df2.celltype_final.isin(['ON SAC', 'OFF SAC'])].hull_diameter.median()
max_rgc = df2.loc[i_largest_rgc].hull_diameter
max_ac = df2.loc[i_largest_ac].hull_diameter

max_ew2 = 1108.0151710123673


def plot_size_comparison(ax, sizes):
    ax.hist(sizes, bins=30, color='gray')
    ax.set_xlabel('Hull diameter [µm]')
    
    make_vline(x=ew1_max, label='largest RGC in EW1', ax=ax, line_kws=line_kws, text_kws=text_kws)
    make_vline(x=avg_sac, label='average SAC', ax=ax, line_kws=line_kws, text_kws=text_kws)
    make_vline(x=max_rgc, label='largest RGC in EW2', ax=ax, line_kws=line_kws, text_kws=text_kws)
    make_vline(x=max_ac, label='largest AC in EW2', ax=ax, line_kws=line_kws, text_kws=text_kws)
    
    make_vline(x=max_ew2, label='max. EW2', ax=ax, line_kws=line_kws, text_kws=text_kws)


# %%
# %%time
fig, axs = plt.subplot_mosaic(
    """
    A
    """,
    figsize=(4, 2.7)
)

ax = axs['A']
plot_size_comparison(ax, sizes_all)
ax.set_yscale('log')
ax.set_ylabel('Cell count\n(ACs and RGCs only)')

plt.tight_layout()
fig.savefig(f'{fig_dir}/cell_sizes.svg', bbox_inches='tight')
fig.savefig(f'{fig_dir}/cell_sizes.png', dpi=600, bbox_inches='tight')

# %% [markdown]
# # Counts

# %%
# %%time
df1_labels[~df1_labels.ew1_label.isin(['OFF-SAC', 'ON-SAC'])].shape[0]

# %%
# %%time
df2.status.unique()

# %%
# %%time
count_dict = {
    # Class   
    "RGCs": {
        'EW2': {
            #'expected': 3_000, #???
            'identified': df2[(df2['cellclass_final'] == 'RGC')].shape[0],
            'complete': df2[(df2['cellclass_final'] == 'RGC') & (df2.status == 'Complete')].shape[0],
            'cutoff': df2[(df2['cellclass_final'] == 'RGC') & (df2.status == 'Complete (cut off)')].shape[0],
        },
        
        'EW1': {
            'identified': 381,#df1_labels[~df1_labels.ew1_label.isin(['OFF-SAC', 'ON-SAC'])].shape[0],
            'complete': 109,
            'cutoff': 272,
        },
    },

    "ACs": {
        'EW2': {
            #'expected': 10_000, #???
            'identified': df2[(df2['cellclass_final'] == 'AC')].shape[0],
            'complete': df2[(df2['cellclass_final'] == 'AC') & (df2.status == 'Complete')].shape[0],
            'cutoff': df2[(df2['cellclass_final'] == 'AC') & (df2.status == 'Complete (cut off)')].shape[0],
        },
        
        'e06': {  # https://www.nature.com/articles/nature12346
            'identified': 407, #???
        },
    },

    "BCs": {
        'EW2': {
            #'expected': 50_000, #???
            'identified': df2[(df2['cellclass_final'] == 'BC') & (df2['status'] == 'ok')].shape[0],
        },
        
        'e06': {  # https://www.nature.com/articles/nature12346
            'identified': 496, # ???
            'complete': 496,
        },
    },

        
    "SACs": {
        'EW2': {
            'identified': df2[df2.celltype_final.isin(['OFF SAC', 'ON SAC'])].shape[0],
            'complete': df2[df2.celltype_final.isin(['OFF SAC', 'ON SAC']) & (df2.status == 'Complete')].shape[0],
            'cutoff': df2[df2.celltype_final.isin(['OFF SAC', 'ON SAC']) & (df2.status == 'Complete (cut off)')].shape[0],
        },
        
        'EW1': {
            'identified': df1_labels[df1_labels.ew1_label.isin(['OFF-SAC', 'ON-SAC'])].shape[0],
        },
    },


    'legend': {},
    
    "FmOn": {
        'EW2': {
            'identified': df2[df2.celltype_final.isin(['F-mini-ON'])].shape[0],
            'complete': df2[df2.celltype_final.isin(['F-mini-ON']) & (df2.status == 'Complete')].shape[0],
            'cutoff': df2[df2.celltype_final.isin(['F-mini-ON']) & (df2.status == 'Complete (cut off)')].shape[0],
        },
        
        'EW1': {
            'identified': 25,
            'complete': 17,
            'cutoff': 8,
        },
    },
    
    "ON alpha": {
        'EW2': {
            'identified': df2[df2.celltype_final.isin(['ON alpha'])].shape[0],
            'complete': df2[df2.celltype_final.isin(['ON alpha']) & (df2.status == 'Complete')].shape[0],
            'cutoff': df2[df2.celltype_final.isin(['ON alpha']) & (df2.status == 'Complete (cut off)')].shape[0],
        },
        
        'EW1': {
            'identified': 4,
            'complete': 0,
            'cutoff': 4,
        },
    },

    "M1": {
        'EW2': {
            'identified': df2[df2.celltype_final.isin(['M1'])].shape[0],
            'complete': df2[df2.celltype_final.isin(['M1']) & (df2.status == 'Complete')].shape[0],
            'cutoff': df2[df2.celltype_final.isin(['M1']) & (df2.status == 'Complete (cut off)')].shape[0],
        },
        
        'EW1': {
            'identified': 2,
            'complete': 0,
            'cutoff': 2,
        },
    },

    "M2": {
        'EW2': {
            'identified': df2[df2.celltype_final.isin(['M2'])].shape[0],
            'complete': df2[df2.celltype_final.isin(['M2']) & (df2.status == 'Complete')].shape[0],
            'cutoff': df2[df2.celltype_final.isin(['M2']) & (df2.status == 'Complete (cut off)')].shape[0],
        },
        
        'EW1': {  # https://museum.eyewire.org/?neurons=20228
            'identified': 1,
            'complete': 0,
            'cutoff': 1,
        },
    },
    
    "t7": {
        'EW2': {
            'identified': df2[(df2['cellclass_final'] == 'BC') & df2['valid_celltype_final'] & (df2['celltype'] == 't7')].shape[0],
        },
        
        'e06': {
            'identified': 29, #???
            'complete': 29,
        },
    },
    
}

############

def get_stack(d):
    identified = d.get('identified', 0)
    complete   = d.get('complete', 0)
    cutoff     = d.get('cutoff', 0)
    expected   = d.get('expected', identified)  # if no expected, no "missing" bar
    other      = max(identified - complete - cutoff, 0)
    missing    = max(expected - identified, 0)
    return {"Complete": complete, "Removed": cutoff, "Not estimated": other, "Expected": missing}


def add_scale_bar(ax, x_pos, max_val, n_ticks=3):
    """Draw a vertical scale bar with a single rotated label."""
    import math
    magnitude = 10 ** math.floor(math.log10(max_val)) if max_val > 0 else 1
    scale = round(max_val / magnitude * 0.5) * magnitude / 2
    if scale == 0:
        scale = magnitude / 2

    tick_values = [i * scale for i in range(n_ticks + 1) if i * scale <= max_val * 1.05]
    bar_height = tick_values[-1]

    # Just a vertical line, no tick marks
    ax.plot([x_pos, x_pos], [0, bar_height], color="black", solid_capstyle='butt', lw=0.8, clip_on=False)

    # Single rotated label at the midpoint showing the bar's total height
    ax.text(x_pos - 0.01, 0, f"{int(bar_height)}",
            ha="right", va="bottom", fontsize=8, rotation=90)


def plot_count_dict(count_dict, ax):
    categories = ["Complete", "Removed", "Not estimated"]#, "Expected"]
    colors     = {"Complete": "#008B8B", "Removed": "#DD8452", "Not estimated": "gray", "Expected": "#C44E52"}
    bar_width  = 0.35
    group_gap  = 0.6
    pair_gap   = 0.05

    cell_classes = list(count_dict.keys())

    positions_ew2, positions_cmp, group_centers, cmp_labels = [], [], [], []
    x = 0
    for cls in cell_classes:
        datasets = count_dict[cls]
        cmp_name = next(k for k in datasets if k != 'EW2')
        cmp_labels.append(cmp_name)

        pos_ew2 = x
        pos_cmp = x + bar_width + pair_gap
        positions_ew2.append(pos_ew2)
        positions_cmp.append(pos_cmp)
        group_centers.append((pos_ew2 + pos_cmp) / 2)
        x += 2 * bar_width + pair_gap + group_gap

    legend_added = set()
    all_totals_ew2 = []
    all_totals_cmp = []

    for i, cls in enumerate(cell_classes):
        ew2_stack = get_stack(count_dict[cls]['EW2'])
        cmp_name  = cmp_labels[i]
        cmp_stack = get_stack(count_dict[cls][cmp_name])

        # Identified total = everything except "Expected (missing)"
        total_ew2 = sum(ew2_stack[c] for c in categories)
        total_cmp = sum(cmp_stack[c] for c in categories)
        all_totals_ew2.append(total_ew2)
        all_totals_cmp.append(total_cmp)

        bottom_ew2 = bottom_cmp = 0
        for cat in categories:
            v_ew2 = ew2_stack[cat]
            v_cmp = cmp_stack[cat]
            color = colors[cat]
            label = cat if cat not in legend_added else None
            if label:
                legend_added.add(cat)

            ax.bar(positions_ew2[i], v_ew2, bar_width, bottom=bottom_ew2,
                   color=color, label=label, edgecolor="none")
            ax.bar(positions_cmp[i], v_cmp, bar_width, bottom=bottom_cmp,
                   color=color, edgecolor="none")

            bottom_ew2 += v_ew2
            bottom_cmp += v_cmp

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove yticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add scale bar (left of first bar)
    y_max = ax.get_ylim()[1]
    scale_x = positions_ew2[0] - bar_width * 0.7
    add_scale_bar(ax, scale_x, y_max * 0.95)

    # x-axis labels (cell class)
    ax.set_title(cell_classes[0], fontsize=8, pad=1)

    # Sub-labels (EW2 / dataset name) and n= annotations below bars
    y_lim = ax.get_ylim()
    y_range = y_lim[1] - y_lim[0]
    label_offset  = y_range * 0.06
    n_offset      = y_range * 0.23

    for i, (pos_ew2, pos_cmp, cmp_name) in enumerate(zip(positions_ew2, positions_cmp, cmp_labels)):
        ax.text(pos_ew2, -label_offset, "EW2",
                ha="center", va="top", fontsize=7, color="#333333")
        ax.text(pos_cmp, -label_offset, cmp_name,
                ha="center", va="top", fontsize=7, color="#333333")

        # n= below sub-label
        ax.text(pos_ew2, -n_offset, f"{all_totals_ew2[i]}",
                ha="center", va="top", fontsize=7, color="#555555", style="italic")
        ax.text(pos_cmp, -n_offset, f"{all_totals_cmp[i]}",
                ha="center", va="top", fontsize=7, color="#555555", style="italic")
        
    ax.set_ylim(-n_offset * 2.5, y_lim[1] * 1.05)

###################

fig, axs = plt.subplots(2, (len(count_dict)+1)//2, figsize=(4, 2.4))

axs = axs.flatten()

for i, (k, v) in enumerate(count_dict.items()):
    ax = axs[i]
    if k == 'legend':
        ax.axis('off')
        axs[i-1].legend(title='Status', bbox_to_anchor=(0.9, 1.1), loc="upper left", fontsize=6)
    else:
        ax.tick_params(length=0)
        plot_count_dict({k: v}, ax)

fig.savefig(f'{fig_dir}/cell_counts.svg', bbox_inches='tight')
fig.savefig(f'{fig_dir}/cell_counts.png', dpi=600, bbox_inches='tight')

plt.show()

# %%
from watermark import watermark
print(watermark())
