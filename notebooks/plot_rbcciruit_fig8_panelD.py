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

# %%
# %%time
import os

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# %%
# %%time
import sys

sys.path.append("../utils")

import data_io

# %% [markdown]
# # setup

# %%
# %%time
cm = 1/2.54  # centimeters in inches
plt.style.use('paper.mplstyle')

# %%
# %%time
order = ["A17", "A2", "nNOS-1", "Other"]

cmap = {"RBC" :  "#0C7C59",
        "A2":    "#134D7C",
        "A17":   "#32A9DE",
        "nNOS-1":    "#B1DEF2",
        "Separate dendrites and axons":    "#B1DEF2",
        "Other": "#1D74B9" }


# %% [markdown]
# # load data

# %%
# %%time
config = data_io.get_data_config()
version = config.get("version_rb", None)
file_path_celltypes = data_io.get_file_path(config,
                                  version = "version_rb",
                                  file_prefix = "file_prefix_rbc_celltypes")

file_path_synapses = data_io.get_file_path(config,
                                  version = "version_rb",
                                  file_prefix = "file_prefix_rbc_synapses")

# %%
# %%time
assert os.path.isfile(file_path_celltypes), file_path_celltypes
assert os.path.isfile(file_path_synapses), file_path_synapses

# %%
# %%time
celltype_df = pd.read_parquet(file_path_celltypes)
synapses_df = pd.read_parquet(file_path_synapses)

# %%
# %%time
synapses_df

# %% [markdown]
# ## rename celltypes

# %%
# %%time
celltype_df = celltype_df.replace("A17 large", 'A17')       # remove distinction for simplicity
celltype_df = celltype_df.replace("A17 small", 'A17')

celltype_df = celltype_df.replace("H42", 'Other')           # preliminary prediction, remove for manuscript
celltype_df = celltype_df.replace("H52", 'Other')

celltype_df = celltype_df.replace("A1", 'n-NOS1')           # overwrite preliminary name given by classifier



# %%
# %%time
synapses_df[['source_type', 'target_type']] = synapses_df[['source_type', 'target_type']].replace("A17 large", 'A17')
synapses_df[['source_type', 'target_type']] = synapses_df[['source_type', 'target_type']].replace("A17 small", 'A17')

synapses_df[['source_type', 'target_type']] = synapses_df[['source_type', 'target_type']].replace("H42", 'Other')
synapses_df[['source_type', 'target_type']] = synapses_df[['source_type', 'target_type']].replace("H52", 'Other')

synapses_df[['source_type', 'target_type']] = synapses_df[['source_type', 'target_type']].replace("A1", 'nNOS-1')

# %%
# %%time
len(synapses_df), len(celltype_df)

# %% [markdown]
# # overview

# %%
# %%time
celltype_df.value_counts("Cell Type")  # all connections that occur only once will be filtered out

# %% [markdown]
# ## output connections 

# %%
# %%time
### get all synapses from RBC ###
cell_type = "RBC"
synapses_out = synapses_df[synapses_df['source_type'] == cell_type]
# ct_ids = np.unique(synapses_out['source'].values)
# len(ct_ids), len(synapses_out)

# %%
# %%time
### remoove those which appear only once from type to type  ### 
counts = synapses_out.groupby(["source_type", "target_type"]).transform("size")
synapses_out = synapses_out[counts > 1]

# %%
# %%time
### count synapses ##
synapse_counts = synapses_out.value_counts(subset = ['target_type'])
print('total output synapses per postsynaptic cell type')
print('------------------')
synapse_counts

# %%
# %%time
## count postsynaptic cells ###
celltype_counts = synapses_out.value_counts(subset = ['target_type'])
print('total output synapses per postsynaptic cell type')
print('------------------')
celltype_counts

# %% [markdown]
# ## input connections

# %%
# %%time
### get all synapses from RBC ###

cell_type = "RBC"
synapses_in = synapses_df[synapses_df['target_type'] == cell_type]
# # synapses_in = synapses_df[synapses_df['target'].isin(segments)]
# ct_ids = np.unique(synapses_out['source'].values)
# ct_ids

# len(ct_ids), len(synapses_in)

# %%
# %%time
### remoove those which appear only once from type to type  ### 
counts = synapses_in.groupby(["source_type", "target_type"]).transform("size")
synapses_in = synapses_in[counts > 1]

# %%
# %%time
### count synapses ##

celltype_counts = synapses_in.value_counts(subset = ['source_type'])
print('total input synapses per presynaptic cell type')
print('------------------')

celltype_counts

# %%
# %%time
## count pre-synaptic cells ###
celltype_counts = synapses_in.value_counts(subset = ['source_type'])
print('total input synapses per presynaptic cell type')
print('------------------')
celltype_counts

# %% [markdown]
# # figures

# %%
# %%time
fig_dir = f'../figures/connectivity/v{version}'
os.makedirs(fig_dir, exist_ok=True)

# %%
# %%time
# celltype_counts = synapses_out.value_counts(subset = ['target_type'])
# print('total output synapses per postsynaptic cell type')
# celltype_counts

# %%
# %%time
# get all outgoing synapses per RBC per postsynaptic cell
celltype_counts = synapses_out.value_counts(subset = ['target_type', 'source'])   # count synapses per cell to target type
celltype_counts = pd.DataFrame(celltype_counts)
celltype_counts = celltype_counts.reset_index()
celltype_counts = celltype_counts.sort_values('target_type')

celltype_counts_out = celltype_counts
celltype_counts_out

# %% [markdown]
# ## panel D : outputs barplot

# %%
# %%time
degree_type = 'count'
order = ["A17", "A2", "nNOS-1"]
box_width = 0.5   

fig  = plt.figure(figsize = (4*cm,2*cm))
ax = fig.add_subplot(111)

sns.boxplot(data = celltype_counts_out, x = 'target_type', y = degree_type,
            hue = 'target_type', order = order, palette = cmap,
            linewidth=0,
            width=box_width,
            whis=1,
            showfliers=False,
            medianprops=dict(linewidth=0),   # hide seaborn's median; redrawn below
            boxprops=dict(edgecolor='none'),
            whiskerprops=dict(linewidth=.5, color='lightgrey', linestyle='-'),
            capprops=dict(linewidth=.5, color='lightgrey'),
            flierprops = dict(marker='o', markerfacecolor='None', markersize=.5,  markeredgecolor='lightgrey'),
            ax=ax)


# ── Set box alpha ──────────────────────────────────────────────────────────
for patch in ax.patches:
    if isinstance(patch, mpatches.FancyBboxPatch):
        continue
    patch.set_alpha(0.5)

# ── Redraw median lines from tick index, not patch geometry ────────────────
# When hue == x (one hue level per cell type, no dodging), each cell type's
# box centre sits exactly at its integer tick index.
half_w  = box_width / 2
med_ext = 0.08   # how much the median line extends beyond each box edge

for i, label in enumerate(order):
    col_data = celltype_counts_out.loc[celltype_counts_out['target_type'] == label, degree_type].dropna()
    if col_data.empty:
        continue
    median_val = col_data.median()
    color = cmap.get(label, 'grey')

    ax.plot([i - half_w - med_ext, i + half_w + med_ext],
            [median_val, median_val],
            color=color, lw=1.5, solid_capstyle='butt',
            zorder=5, transform=ax.transData)


sns.stripplot(data = celltype_counts_out, x = 'target_type', y = degree_type,
              palette = cmap,
              hue = 'target_type',
              size = 2,
              order = order)


plt.xticks(rotation = 55)
plt.ylim(-10,60)

ax.set_ylabel("#")
ax.set_xlabel("")

sns.despine(trim = True)


# ax.set_title(f"{cell_type} outputs", loc = 'left')
ax.set_title(f"outputs", loc = 'left')
fig.savefig(f'{fig_dir}/fig8_D_outputs_boxplot.svg')
fig.savefig(f'{fig_dir}/fig8_D_outputs_boxplot.png')

# %% [markdown]
# ## panel D : inputs barplot

# %%
# %%time
celltype_counts = synapses_in.value_counts(subset = ['source_type', 'target'])
celltype_counts = pd.DataFrame(celltype_counts)
celltype_counts = celltype_counts.reset_index()
celltype_counts = celltype_counts.sort_values('source_type')
# cell_types = list(celltype_counts['source_type'])
celltype_counts_in = celltype_counts

# %%
# %%time
celltype_counts_in

# %%
# %%time
degree_type = 'count'

order = ["A17", "nNOS-1", "Other"]

fig  = plt.figure(figsize = (4*cm,2*cm))
ax = fig.add_subplot(111)

sns.boxplot(data = celltype_counts_in, x = 'source_type', y = degree_type,
            hue = 'source_type', order = order, palette = cmap,
            linewidth=0,
            width=box_width,
            whis=1,
            showfliers=False,
            medianprops=dict(linewidth=0),   # hide seaborn's median; redrawn below
            boxprops=dict(edgecolor='none'),
            whiskerprops=dict(linewidth=.5, color='lightgrey', linestyle='-'),
            capprops=dict(linewidth=.5, color='lightgrey'),
            flierprops = dict(marker='o', markerfacecolor='None', markersize=.5,  markeredgecolor='lightgrey'),
            ax=ax)


# ── Set box alpha ──────────────────────────────────────────────────────────
for patch in ax.patches:
    if isinstance(patch, mpatches.FancyBboxPatch):
        continue
    patch.set_alpha(0.5)

# ── Redraw median lines from tick index, not patch geometry ────────────────
# When hue == x (one hue level per cell type, no dodging), each cell type's
# box centre sits exactly at its integer tick index.
half_w  = box_width / 2
med_ext = 0.08   # how much the median line extends beyond each box edge

for i, label in enumerate(order):
    col_data = celltype_counts_in.loc[celltype_counts_in['source_type'] == label, degree_type].dropna()
    if col_data.empty:
        continue
    median_val = col_data.median()
    color = cmap.get(label, 'grey')

    ax.plot([i - half_w - med_ext, i + half_w + med_ext],
            [median_val, median_val],
            color=color, lw=1.5, solid_capstyle='butt',
            zorder=5, transform=ax.transData)


sns.stripplot(data = celltype_counts_in, x = 'source_type', y = degree_type,
              palette = cmap,
              hue = 'source_type',
              size = 2,
              order = order)

plt.xticks(rotation = 55)


sns.despine(trim = True)
ax.set_ylabel("#")
ax.set_xlabel("")
ax.set_xticklabels(["A17", "nNOS-1", "MF AC"])


ax.set_title(f"inputs", loc = 'left')
fig.savefig(f'{fig_dir}/fig8_D_inputs_boxplot.svg')
fig.savefig(f'{fig_dir}/fig8_D_inputs_boxplot.png')

# %%
from watermark import watermark
print(watermark())
