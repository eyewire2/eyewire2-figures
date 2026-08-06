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
import seaborn as sns

# %%
# %%time
import sys

sys.path.append("../utils")

import data_io

# %%
# %%time
import colors
from embedding import plot_embedding, save_and_plot_feats
from mosaics import polygon_centroid, plot_multiple_mosaics
from dendrogram import ClusterDendrogram

from plot_cells import plot_cell_morphologies

# %% [markdown]
# # Data

# %%
# %%time
config = data_io.get_data_config()
version = config.get("version", None)
skel_dir = config.get("skel_dir", None)
file_path = data_io.get_file_path(config)

assert os.path.isdir(skel_dir), skel_dir
assert os.path.isfile(file_path), file_path

# %%
# %%time
df = pd.read_parquet(file_path)
print(df.shape)

# %%
# %%time
assert not df.columns.duplicated().any()

# %% [markdown]
# # Labels and subset

# %%
# %%time
df_subset = pd.read_csv('RGC types for Figure.csv')

# %%
# %%time
celltype2short = {row['celltype']: (row['Show'] if not pd.isna(row['Show']) else row["Don't show"]) for i, row in df_subset.iterrows()}
celltype2show = {row['celltype']: not pd.isna(row['Show']) for i, row in df_subset.iterrows()}

for k, v in celltype2short.items():
    print(f"show={celltype2show[k]}: {k} ({v})")

# %%
# %%time
df['celltype_short'] = df['celltype'].apply(lambda x: celltype2short.get(x, None))
df['celltype_final_short'] = df['celltype_final'].apply(lambda x: celltype2short.get(x, None))
df['valid_celltype_final_short'] = df['celltype_final'].apply(lambda x: celltype2show.get(x, False))

# %% [markdown]
# ### Collect labels

# %%
# %%time
unique_labels = np.sort(np.unique(df.loc[df["valid_celltype_final_short"], "celltype_final"]))
unique_labels_short = np.sort(np.unique(df.loc[df["valid_celltype_final_short"], "celltype_final_short"]))

# %%
# %%time
print(len(unique_labels), len(unique_labels_short))

# %%
# %%time
unique_labels_short

# %% [markdown]
# ## Keep only RGC labels
#
# we need to filter by type here, otherwise we will lose a tiny fraction of RGCs that have a valid type label but are not above the threshold for the class label

# %%
# %%time
df = df[(df.cellclass_final=='RGC') | df.celltype_final.isin(unique_labels)].copy()
df = data_io.restore_numpy_arrays(df)

# %%
# %%time
df.shape

# %% [markdown]
# ## TSNE

# %%
# %%time
emb_cols_norm = [c for c in df.columns if c.startswith('RGCemb_norm')]
emb_cols_norm

# %%
# %%time
emb_feats_norm = df[emb_cols_norm].values
emb_feats_norm.shape

# %%
# %%time
df['include_in_embedding'] = (df.cellclass_final=='RGC') & df["valid_cellclass_final"]

# %%
# %%time
import scanpy as sc

adata = sc.AnnData(df.loc[df['include_in_embedding'], emb_cols_norm].values)
sc.tl.tsne(adata, perplexity=8, use_rep='X', random_state=0)  # 20?

# %%
# %%time
df['tsne_d0'] = np.nan
df['tsne_d1'] = np.nan

df.loc[df["include_in_embedding"], 'tsne_d0'] = adata.obsm["X_tsne"][:, 0]
df.loc[df["include_in_embedding"], 'tsne_d1'] = adata.obsm["X_tsne"][:, 1]

# %% [markdown]
# ## Get one per type

# %%
# %%time
size_order = df[df.valid_celltype_final_short].groupby('celltype_final_short').aggregate(
    {'hull_diameter': 'mean'}).sort_values('hull_diameter').index.values

all_types_rows = []

for celltype_final_short in size_order:
    rows = df[(df.celltype_final_short == celltype_final_short)
        & df.valid_celltype_final_short]

    if sum(rows.celltype_final_decision == 'both_strong') > 0:
        rows = rows[rows.celltype_final_decision == 'both_strong']
    
    if len(rows) == 0:
        rows = None
        print(celltype_final_short)

    if rows is None:
        row = None
    else:
        x = rows['tsne_d0']
        y = rows['tsne_d1']
        
        emb_ct = np.stack([x, y]).T
        
        ct_mean = np.mean(emb_ct, axis=0)
        ct_dists = np.sum((emb_ct - ct_mean) ** 2, axis=1) ** 0.5
        row = rows.iloc[np.argmin(ct_dists)]
        
    all_types_rows.append(row)
    
all_types_rows = pd.DataFrame(all_types_rows)
all_types_rows.head(2)

# %%
# %%time
for i, name in all_types_rows['celltype_final'].items():
    print(i, name)

# %% [markdown]
# ## Prepare mosaic cell types

# %%
# %%time
celltypes_to_mosaic = ["sOn-a", "tOff-a", "FmOn", "FmOff"]

# %%
# %%time
# Some cells still have their axon unlabeled, let's remove them here for the mosaics
df_mosaics = df[df["valid_celltype_final_short"]].copy()
df_mosaics["hull_shape_idx"] = ((df.hull_perimeter + 10) / (df.hull_diameter + 10)) / np.pi - 1

fig, axs = plt.subplots(1, len(celltypes_to_mosaic), figsize=(len(celltypes_to_mosaic)*3, 2))

check_cells = []

for i, ct in enumerate(celltypes_to_mosaic):
    df_ct = df_mosaics[(df_mosaics.celltype_final_short == ct)]

    df_ct_flag = df_ct[df_ct["hull_shape_idx"] > 0.2]
    
    print(ct, list(df_ct_flag.index))
    check_cells += list(df_ct_flag.index)

    ax = axs[i]
    ax.set_title(f"{ct}\nn_removed={len(df_ct_flag)}")
    ax.hist(df_ct.hull_shape_idx)

# Confirm the selection manually before filtering
rm_cells = check_cells
df_mosaics = df_mosaics.loc[~df_mosaics.index.isin(rm_cells)]

# %% [markdown]
# # Figures

# %%
# %%time
from style import set_rc_params

set_rc_params()

# %%
# %%time
fig_dir = f'../figures/RGC/v{version}'
os.makedirs(fig_dir, exist_ok=True)

# %% [markdown]
# ## Feature plots

# %% [markdown]
# ### Dendrograms

# %%
# %%time
cd = ClusterDendrogram(
    data=emb_feats_norm[df.valid_celltype_final_short & df.include_in_embedding],
    cluster_labels=df.loc[df.valid_celltype_final_short & df.include_in_embedding, "celltype_final_short"],
    feature_names=emb_cols_norm,
)
avg_features = cd.compute_cluster_averages(log_transform=False)
filtered_features, selected_features = cd.filter_features(verbose=False)
dist_matrix = cd.compute_distance_matrix(metric='euclidean')
Z = cd.hierarchical_cluster(method='average')

# %%
# %%time
from scipy.cluster.hierarchy import dendrogram, linkage

dendro = dendrogram(
    cd.Z,
    labels=cd.clusters,
    leaf_font_size=8,
    color_threshold=0,
    above_threshold_color='gray',
    no_plot=True,
)

# %%
# %%time
leaf_order = cd.clusters[dendro['leaves']]
leaf_order

# %%
# %%time
len(unique_labels)

# %%
# %%time
len(leaf_order)

# %%
# %%time
interpolated_colors = colors.interpolate_colors(n=len(unique_labels))
ordered_colors = colors.reorder_colors_for_distinction(interpolated_colors, n_neighbors=4)

label2color = {leaf_name: ordered_colors[i] for i, leaf_name in enumerate(leaf_order)}

# %% [markdown]
# ### Plot stats

# %%
# %%time
fig = cd.plot_heatmap_with_dendrogram(top_n_features=50, figsize=(8, 6))

ax = fig.ax_heatmap
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.tick_params("x", rotation=90)
xlabels = ax.get_xmajorticklabels()
xlabelnames = []
for label in xlabels:
    label.set_color(label2color[label.get_text()])

fig.savefig(f'{fig_dir}/rgc-dendrogram_features.svg')
fig.savefig(f'{fig_dir}/rgc-dendrogram_features.png', dpi=600)
plt.show()

# %% [markdown]
# ### Plot dendrogram

# %%
# %%time
fig, ax = plt.subplots(figsize=(7, 1.5))
sns.despine(left=True, bottom=True, right=True, top=True)

# Plot dendrogram
dendro = dendrogram(
    cd.Z,
    labels=cd.clusters,
    ax=ax,
    leaf_font_size=8,
    color_threshold=0,
    above_threshold_color='gray',
    no_plot=False,
)

ax.tick_params("x", rotation=90)
xlabels = ax.get_xmajorticklabels()
xlabelnames = []
for label in xlabels:
    label.set_color(label2color[label.get_text()])

assert len(np.unique(xlabelnames)) == len(xlabelnames)

ax.set_yticks([])
plt.tight_layout()

fig.savefig(f'{fig_dir}/rgc-dendrogram.svg')
fig.savefig(f'{fig_dir}/rgc-dendrogram.png', dpi=600)
plt.show()

# %% [markdown]
# ## TSNE

# %% [markdown]
# ### Plot

# %%
# %%time
plot_embedding(
    df=df[df['include_in_embedding']],
    label2color=label2color,
    fig_path_prefix=f"{fig_dir}/rgc-tsne",
    celltype_col="celltype_final_short",
    plot_order=leaf_order,
    is_labelled_col="valid_celltype_final_short",
    dot_size=12,
    edgecolor="k",
    rasterized=False,
    legend_markerscale=None,
)

# %% [markdown]
# ### Feature maps

# %%
# %%time
df.rename({"perc_z_095_1": "perc_z_095", "perc_z_005_1": "perc_z_005"}, axis=1, inplace=True)

# %%
# %%time
df["log_soma_rad_um"] = np.log(df["soma_rad_um"])

# %%
# %%time
example_feature_names = [
    'perc_z_005',
    'perc_z_095',

    'log_hull_diameter',
    'log_branch_points',
    
    'log_soma_rad_um',
    'log_radius_mean',
]

example_features = df.loc[:, example_feature_names].values

# %%
# %%time
fig, axs = save_and_plot_feats(
    all_emb=df.loc[df['include_in_embedding'], ['tsne_d0', 'tsne_d1']].values,
    all_feats=example_features[df['include_in_embedding']],
    all_feat_names=example_feature_names, 
    ncols=2,
    fig_dir=fig_dir,
    file_prefix="rgc-tnse-features",
    clip=False,
    stride=1, 
    s=1,
    figsize_per_cell=(1.5, 1.5)
)

# %% [markdown]
# ## Morph examples

# %%
# %%time
nrows = 3

rad = (all_types_rows.hull_diameter.max() / 2) * 1.1

splits = np.array_split(all_types_rows, nrows)

for i in range(nrows):
    rows = splits[i]

    print(i)
    print(list(rows.index.astype(int)))
    
    fig, axs = plt.subplots(
        2, len(rows), figsize=(7 * rows.shape[0] / splits[0].shape[0], 1.7),
        sharex='all', sharey='row', squeeze=False, height_ratios=(1, 1.2),
        subplot_kw=dict(xlabel=None, ylabel=None, xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0, hspace=0)
    
    plot_cell_morphologies(
        rows=rows, rad=rad,
        skel_dir=skel_dir,
        size=300, fig=fig, axs=axs,
        color=colors.cellclass2color['RGC'],
        show_on_tsne=False, sb_fontsize=8);

    for ax in axs.flat:
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.axis('off')
    
    for j, ax in enumerate(axs[0, :]):
        ct_short = rows.iloc[j].celltype_final_short
        df_ct = df[(df.celltype_final_short==ct_short) & df.valid_celltype_final]
        n_cells = len(df_ct)
        n_cells_clf = sum(df_ct.celltype_final_decision == 'classifier')
        n_cells_label = n_cells - n_cells_clf
        if n_cells_clf > 0:
            ax.set_title(f"{ct_short}\n(n={n_cells_label} [+{n_cells_clf}])", fontsize=7)
        else:
            ax.set_title(f"{ct_short}\n(n={n_cells_label})", fontsize=7)

    fig.savefig(f'{fig_dir}/celltype_final-example_set{i}.svg', dpi=600, bbox_inches='tight')
    fig.savefig(f'{fig_dir}/celltype_final-example_set{i}.png', dpi=600, bbox_inches='tight')
    
    plt.show()

# %% [markdown]
# ## Mosaics

# %%
# %%time
from scipy.spatial import ConvexHull

pts = np.vstack(df.hull_points)[:, :2]
hull = ConvexHull(pts)

# Get hull boundary points in order, closed
hull_pts = pts[np.append(hull.vertices, hull.vertices[0])]
plt.plot(*hull_pts.T)
plt.show()

# %%
# %%time
extent = [20, 1120, 50, 1150]
max_val = 9

fig, naxs = plt.subplot_mosaic(
    """
    A.Bb
    A.Cc
    A.Dd
    """,
    figsize=(7, 4.1),
    sharex=True,
    sharey=True,
    width_ratios=(3, 0.6, 1, 1),
)

coverage_axs = np.array([naxs['A'], naxs['B'], naxs['C'], naxs['D']])
center_axs = np.array([naxs['A'], naxs['b'], naxs['c'], naxs['d']])
sb_axs = np.array([naxs['A'], naxs['B']])

sns.despine(top=True, right=True, bottom=True, left=True)

ims, c_maxs = plot_multiple_mosaics(
    df=df_mosaics,
    celltype_col='celltype_short',
    candidate_col='celltype_final_short',
    candidate_color='C0',
    celltypes=celltypes_to_mosaic,
    plot_candidate_coverage=True,
    fig=fig,
    coverage_axs=coverage_axs,
    resolutions=[3000, 1000, 1000, 1000],
    center_axs=center_axs,
    title_axs=coverage_axs,
    center_x_col="soma_x_um",
    center_y_col="soma_y_um",
    marker_sizes=[10, 3, 3, 3],
    max_val=max_val,
    extent=extent,
    plot_outlines=False,
    outline_hull=hull_pts,
    cb=False,
    sb_axs=sb_axs,
    sb_size=500,
)

for j, ax in enumerate(coverage_axs):
    ct_short = celltypes_to_mosaic[j]
    ax.set_title(f"{ct_short}", fontsize=8, pad=0)
    
plt.tight_layout(w_pad=-2)

# Custom color bar
for i, celltype_final in enumerate(celltypes_to_mosaic):
    ax = coverage_axs[i]
    im = ims[i]
    c_max = c_maxs[i]
    
    if i > 0:
        if c_max == c_maxs[i-1]:
            continue
    
    pos = ax.get_position()
    
    cax = fig.add_axes([
        pos.x1 + 0.01, # left
        pos.y0 + 0.25 * pos.height,   # bottom (adjust spacing)
        0.01,    
        pos.height * 0.5,
    ])
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Coverage', fontsize=8, labelpad=2)
    cbar.set_ticks(np.arange(0, c_max + 1, int(np.ceil(c_max/5))))
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(True)

fig.savefig(f'{fig_dir}/rgc-mosaic-{celltypes_to_mosaic}_{extent}.svg', dpi=600, bbox_inches='tight')
fig.savefig(f'{fig_dir}/rgc-mosaic-{celltypes_to_mosaic}_{extent}.png', dpi=600, bbox_inches='tight')

plt.show()

# %% [markdown]
# # Website

# %%
# %%time
website_dir = f'../website/v{version}'
os.makedirs(website_dir, exist_ok=True)

# %%
# %%time
website_cols = ['celltype_final', 'celltype_final_short', 'soma_x_um', 'soma_y_um', 'soma_z_um', 'soma_annot_x_um', 'soma_annot_y_um', 'soma_annot_z_um']

# %% [markdown]
# ## Single examples

# %%
# %%time
xy0 = np.array((48142, 36349)) * 16 / 1000
xy1 = np.array((42035, 31849)) * 16 / 1000
xy2 = np.array((37535, 37956)) * 16 / 1000
xy3 = np.array((43642, 42456)) * 16 / 1000

# %%
# %%time
# Box center in um
box_center = (xy0 + xy1 + xy2 + xy3) / 4

website_rows = []
for celltype_final_short in size_order:
    rows = df[
        (df.celltype_final_short == celltype_final_short)
        & df.valid_celltype_final_short
    ]
    
    if len(rows) == 0:
        website_rows.append(None)
        continue
    
    # Spatial distance to box center (medium priority)
    soma_xy = np.stack([rows.soma_annot_x_um, rows.soma_annot_y_um]).T
    spatial_dists = np.linalg.norm(soma_xy - box_center, axis=1)
    spatial_dists = np.clip(spatial_dists, 50, None)
    
    # tSNE distance to centroid (lowest priority, tiebreaker)
    emb_ct = np.stack([rows['tsne_d0'], rows['tsne_d1']]).T
    ct_mean = np.mean(emb_ct, axis=0)
    tsne_dists = np.linalg.norm(emb_ct - ct_mean, axis=1)
    
    # Strong flag (highest priority — invert so 0 = strong wins in argmin/lexsort)
    not_strong = (rows.celltype_final_decision != 'both_strong').values.astype(int)
    
    # lexsort sorts by LAST key as primary → put highest priority last
    order = np.lexsort((tsne_dists, spatial_dists, not_strong))
    row = rows.iloc[order[0]]
    
    website_rows.append(row)

df_website_single = pd.DataFrame(website_rows).loc[:, website_cols].copy()
df_website_single.sort_values('celltype_final', inplace=True)
df_website_single.head()

# %%
# %%time
sns.scatterplot(df_website_single, x='soma_x_um', y='soma_y_um');

# %%
# %%time
df_website_single.to_csv(os.path.join(website_dir, f"RGC_website_single_examples.csv"))

# %% [markdown]
# ## Mosaics

# %%
# %%time
df_website_mosaic = df_mosaics.loc[df_mosaics.celltype_final_short.isin(celltypes_to_mosaic), website_cols].copy()
df_website_mosaic.sort_values('celltype_final', inplace=True)
df_website_mosaic.head()

# %%
# %%time
sns.countplot(data=df_website_mosaic, x='celltype_final');

# %%
# %%time
df_website_mosaic.to_csv(os.path.join(website_dir, f"RGC_website_mosaic_examples.csv"))

# %%
from watermark import watermark
print(watermark())

# %%
