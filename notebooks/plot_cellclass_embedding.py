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

# %%
# %%time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
df = df.drop(["polar_dens_1", "polar_dens_2"], axis=1)  # Not needed
df = data_io.restore_numpy_arrays(df)

print(df.shape)

# %% [markdown]
# ## Labels

# %%
# %%time
cellkinds = [
    'RGC',
    'AC',
    'dAC',
    'ON SAC',
    'OFF SAC',
    'BC',
]


# %%
# %%time
def get_cellkind(row):
    cellclass = row['cellclass_final']
    celltype = row['celltype_final']
    soma_z = row['soma_z_um']

    if pd.isna(cellclass) or cellclass in ['BC', 'RGC']:
        return cellclass
    elif cellclass in ['AC']:
        if not pd.isna(celltype) and celltype in ['ON SAC', 'OFF SAC']:
            return celltype
        elif soma_z < 0:
            return 'dAC'
        else:
            return 'AC'


# %%
# %%time
df['cellkind'] = df.apply(lambda x: get_cellkind(x), axis=1)

# %% [markdown]
# ## TSNE

# %%
# %%time
emb_cols_norm = [c for c in df.columns if c.startswith('ALLemb_norm')]
emb_cols_norm

# %%
# %%time
len(emb_cols_norm)

# %%
# %%time
df['include_in_embedding'] = (df["fda_weight"] > 0) & df["valid_cellclass_final"]

print(df['include_in_embedding'].sum())

# %%
# %%time
import scanpy as sc

adata = sc.AnnData(df.loc[df['include_in_embedding'], emb_cols_norm].values)
sc.tl.tsne(adata, perplexity=30, use_rep='X', random_state=0)

# %%
# %%time
df['tsne_d0'] = np.nan
df['tsne_d1'] = np.nan

df.loc[df["include_in_embedding"], 'tsne_d0'] = adata.obsm["X_tsne"][:, 0]
df.loc[df["include_in_embedding"], 'tsne_d1'] = adata.obsm["X_tsne"][:, 1]

# %% [markdown]
# # Figures

# %%
# %%time
from style import set_rc_params

set_rc_params()

# %%
# %%time
fig_dir = f'../figures/All/v{version}'
os.makedirs(fig_dir, exist_ok=True)

# %% [markdown]
# ## Z-Dens

# %%
# %%time
df.rename({"perc_z_095_1": "perc_z_095", "perc_z_005_1": "perc_z_005"}, axis=1, inplace=True)

# %%
# %%time
cellclasses = ['RGC', 'AC', 'BC']

fig, axs = plt.subplots(2, 4, figsize=(3, 0.8), sharex='col', height_ratios=(10, 1),
                        width_ratios=(1, 1, 1, 0.05))

for i, cellclass in enumerate(cellclasses):
    include = (df['cellclass_final'] == cellclass) & df['valid_cellclass_final']
    
    zi0 = 10
    zi1 = 100
    z0 = -20 + zi0*0.5
    z1 = -20 + zi1*0.5
    
    z_profiles_norm = np.vstack(df.loc[include, 'z_dens_1'])
    z_profiles_norm = z_profiles_norm[:, zi0:zi1]
    z_profiles_norm = (z_profiles_norm.T / (np.max(z_profiles_norm, axis=1) + 1e-20).T).T

    order = np.argsort(df.loc[include, "perc_z_005"])

    ax = axs[0, i]
    ax.set_title(cellclass + 's', fontsize=8)
    ax.set(xticks=[], yticks=[])
    im = ax.imshow(
        z_profiles_norm[order].T, aspect='auto', interpolation='none', origin='lower',
        extent=(0, z_profiles_norm.shape[0], z0, z1), rasterized=True, cmap='Grays')
    
    ax.axhline(0, ls='-', c=colors.cellclass2color['ON SAC'], lw=1)
    ax.axhline(12, ls='-', c=colors.cellclass2color['OFF SAC'], lw=1)

    ax = axs[1, i]
    ax.axis('off')

    if cellclass == 'BC':
        size = 20_000
    else:
        size = 2000
        
    ax.plot([0, size], [0.9, 0.9], c='k', solid_capstyle='butt')
    ax.text(0, 0.5, f'{size}', c='k', ha='left', va='top', fontsize=8)
    ax.set_ylim(0, 1)

cbar = plt.colorbar(im, ax=axs[0, -2], cax=axs[0, -1])
cbar.set_label('Norm. density', fontsize=6)
cbar.outline.set_visible(True)
axs[1, -1].axis('off')

fig.savefig(f'{fig_dir}/cellclass-z-profiles.svg', bbox_inches='tight')
fig.savefig(f'{fig_dir}/cellclass-z-profiles.png', dpi=600, bbox_inches='tight')

# %% [markdown]
# ## TSNE

# %%
# %%time
plot_embedding(
    df=df[df['include_in_embedding']],
    label2color=colors.cellclass2color,
    fig_path_prefix=f"{fig_dir}/class-tsne",
    celltype_col='cellkind',
    plot_order=cellkinds,
    is_labelled_col="valid_cellclass_final",
    dot_size=1,
    edgecolor="none",
    rasterized=True,
    legend_markerscale=4,
    figsize=(4, 3.3)
)

# %%
# %%time
type_order = np.unique(df.loc[df['include_in_embedding'] & df['valid_celltype_final'], 'celltype_final'])

plot_embedding(
    df=df[df['include_in_embedding']],
    label2color={k: f'C{i}' for i, k in enumerate(type_order)},
    fig_path_prefix=f"{fig_dir}/class-tsne-celltype-for-ref",
    celltype_col='celltype',
    plot_order=type_order,
    is_labelled_col="valid_celltype_final",
    dot_size=5,
    edgecolor="none",
    rasterized=True,
    legend_markerscale=2,
    figsize=(14, 14)
)

# %%
# %%time
example_feature_names = [
    'perc_z_005',
    'perc_z_095',
    'log_radius_median',
    'log_tips',
    'log_hull_diameter',
]

example_features = df.loc[:, example_feature_names].values

# %%
# %%time
fig, axs = save_and_plot_feats(
    all_emb=df.loc[:, ['tsne_d0', 'tsne_d1']].values,
    all_feats=example_features,
    all_feat_names=example_feature_names, 
    ncols=5,
    clip=False,
    stride=1, 
    figsize_per_cell=(1.3, 1)
)

plt.tight_layout(h_pad=0.3)

fig.savefig(f'{fig_dir}/class-tnse-features.svg', bbox_inches='tight')
fig.savefig(f'{fig_dir}/class-tnse-features.png', dpi=600, bbox_inches='tight')


# %% [markdown]
# ## Examples

# %%
# %%time
def plot_examples(name, rows, color, df, ):
    
    rad = np.maximum(20, (rows.hull_diameter.max() / 2) * 1.3)

    size = 1.8 * rad

    if size >= 1000:
        size = int(size // 1000) * 1000
    elif size >= 600:
        size = int(size // 200) * 200
    elif size >= 200:
        size = int(size // 100) * 100
    else:
        size = int(size // 50) * 50

    if size <= 20:
        size = 20
    
    fig, axs = plt.subplots(
        2, len(rows), figsize=(0.7*(len(rows)), 1.5), squeeze=False,
        height_ratios=(2.5, 3),
        subplot_kw=dict(xlabel=None, ylabel=None, xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0, hspace=0)
    
    for ax in axs.flat:
        ax.axis('off')
        
    print(rows.index.astype(int))

    fig, fig2 = plot_cell_morphologies(
        rows=rows,
        rad=rad,
        skel_dir=skel_dir,
        size=size,
        fig=fig,
        axs=axs,
        color=color,
        sb_fontsize=8,
        show_on_tsne=True,
        df=df,
        is_labelled=df['include_in_embedding'],
        labels=df['cellkind'],
        label_order=cellkinds,
        celltype2color=colors.cellclass2color,
        
    )

    axs[0, 1].set_title(celltype, fontsize=8)

    fig.savefig(f'{fig_dir}/celltype-example_{name}.svg', dpi=600, bbox_inches='tight')
    fig.savefig(f'{fig_dir}/celltype-example_{name}.png', dpi=600, bbox_inches='tight')
    
    fig2.savefig(f'{fig_dir}/celltype-example_{name}_on_tsne.png', dpi=200, bbox_inches='tight')
    
    plt.show()


# %%
# %%time
prefer_label = df['include_in_embedding'] & df['valid_celltype_final'] & (df.celltype_final_decision == 'both_strong')

# %%
# %%time
celltype = 'ON SAC'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[0, 1, 8]]
plot_examples(celltype, rows, color=colors.cellclass2color[celltype], df=df)

# %%
# %%time
celltype = 'OFF SAC'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[5, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color[celltype], df=df)

# %%
# %%time
prefer_bcs = (df['status'] == 'ok') & (df['status_alt'] == 'yes') & (df['perc_z_098_1'] > 30)

# %%
# %%time
celltype = 'XBC'
rows = df.loc[(df.celltype_final == celltype) & prefer_label & prefer_bcs].iloc[[0, 1, 4]]
plot_examples(celltype, rows, color=colors.cellclass2color['BC'], df=df)

# %%
# %%time
celltype = 't7'
rows = df.loc[(df.celltype_final == celltype) & prefer_label & prefer_bcs].iloc[[0, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['BC'], df=df)

# %%
# %%time
celltype = 'RBC'
rows = df.loc[(df.celltype_final == celltype) & prefer_label & prefer_bcs].iloc[[0, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['BC'], df=df)

# %%
# %%time
celltype = 't2'
rows = df.loc[(df.celltype_final == celltype) & prefer_label & prefer_bcs].iloc[[0, 4, 3]]
plot_examples(celltype, rows, color=colors.cellclass2color['BC'], df=df)

# %%
# %%time
celltype = 'A2'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[0, 3, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['AC'], df=df)

# %%
# %%time
celltype = 'H22'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[0, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['AC'], df=df)

# %%
# %%time
celltype = 'H23'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[0, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['AC'], df=df)

# %%
# %%time
celltype = 'A17 large'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[0, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['AC'], df=df)

# %%
# %%time
celltype = 'A17 small'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[0, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['AC'], df=df)

# %%
# %%time
celltype = 'F-mini-ON'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[0, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['RGC'], df=df)

# %%
# %%time
celltype = 'F-mini-OFF'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[0, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['RGC'], df=df)

# %%
# %%time
celltype = 'OFF transient alpha'
rows = df.loc[(df.celltype_final == celltype) & prefer_label].iloc[[0, 1, 2]]
plot_examples(celltype, rows, color=colors.cellclass2color['RGC'], df=df)

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
max_val = 7

celltypes = ['F-mini-ON', 'A2', 'H22', 'H23', 'ON SAC', 'OFF SAC', 'A17 large', 'A17 small', 'XBC', 't7']
marker_sizes = [1.2] * len(celltypes)
ncols = len(celltypes)

for name, extent in {'wide': [50, 1150, 50, 1150]}.items(): # 'zoom2': [200, 300, 600, 700], 'zoom': [200, 400, 600, 800]
    
    fig, axs = plt.subplots(ncols=len(celltypes), nrows=1, figsize=(len(celltypes)*1.5, 2),
                            sharex='row', sharey='row', squeeze=False)    
    if 'wide' in name.lower():
        sns.despine(top=True, right=True, left=True, bottom=True)
    else:
        sns.despine(top=False, right=False, left=False, bottom=False)
    
    ims, c_maxs = plot_multiple_mosaics(
        df=df[df.valid_celltype_final],
        celltype_col='celltype',
        candidate_col='celltype_final',
        candidate_color='C0',
        celltypes=celltypes,
        fig=fig,
        coverage_axs=None,
        resolutions=[1000] * ncols,
        center_axs=axs[0, :],            
        title_axs=axs[0, :],
        sb_axs=[axs[0, 0]],
        center_x_col="hull_center_x",
        center_y_col="hull_center_y",
        marker_sizes=marker_sizes,
        max_val=max_val,
        extent=extent,
        plot_outlines='zoom' in name.lower(),
        outline_hull=hull_pts if 'wide' in name.lower() else None,
        cb=False,
        sb_size=500 if 'wide' in name.lower() else 100,
        outline_kws = dict(c='gray', ls='-', clip_on=False),
    )

    for j, ax in enumerate(axs.flat):
        celltype = celltypes[j]
        df_ct = df[(df.celltype_final==celltype) & df.valid_celltype_final]
        n_cells = len(df_ct)
        n_cells_clf = sum(df_ct.celltype_final_decision == 'classifier')
        n_cells_label = n_cells - n_cells_clf
        if n_cells_clf > 0:
            ax.set_title(f"{celltype}\n(n={n_cells_label} [+{n_cells_clf}])", fontsize=7)
        else:
            ax.set_title(f"{celltype}\n(n={n_cells_label})", fontsize=7)
    
    fig.savefig(f'{fig_dir}/scatter-class_{extent}.svg', dpi=600, bbox_inches='tight')
    fig.savefig(f'{fig_dir}/scatter-class_{extent}.png', dpi=600, bbox_inches='tight')
    plt.show()

# %% [markdown]
# # Website
#
# Only ACs here, rest is done in different notebooks

# %%
# %%time
website_dir = f'../website/v{version}'
os.makedirs(website_dir, exist_ok=True)

# %%
# %%time
website_cols = ['celltype_final', 'soma_x_um', 'soma_y_um', 'soma_z_um', 'soma_annot_x_um', 'soma_annot_y_um', 'soma_annot_z_um']

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
website_selection = ['A2', 'H22', 'H23', 'ON SAC', 'OFF SAC', 'A17 small', 'A17 large']

# Box center in um
box_center = (xy0 + xy1 + xy2 + xy3) / 4

website_rows = []
for celltype_final in website_selection:
    rows = df[
        (df.celltype_final == celltype_final)
        & df.valid_celltype_final
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
df_website_single.to_csv(os.path.join(website_dir, f"AC_website_single_examples.csv"))

# %%
# %%time
df_website_single.index.astype(int)

# %% [markdown]
# ## Mosaics

# %%
# %%time
website_mosaics = ['ON SAC', 'OFF SAC']

df_website_mosaic = df.loc[df.valid_celltype_final & df.celltype_final.isin(website_mosaics), website_cols].copy()
df_website_mosaic.sort_values('celltype_final', inplace=True)
df_website_mosaic.head()

# %%
# %%time
sns.countplot(data=df_website_mosaic, x='celltype_final');

# %%
# %%time
df_website_mosaic.to_csv(os.path.join(website_dir, f"AC_website_mosaic_examples.csv"))

# %%
from watermark import watermark
print(watermark())
