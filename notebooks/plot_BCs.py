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

# %%
# %%time
assert os.path.isdir(skel_dir), skel_dir
assert os.path.isfile(file_path), file_path

# %%
# %%time
df = pd.read_parquet(file_path)
df = df[(df.cellclass_final == 'BC')].copy()
df = df.drop(["polar_dens_1"], axis=1)  # Not needed
df = data_io.restore_numpy_arrays(df)

print(df.shape)

# %%
# %%time
assert not df.columns.duplicated().any()

# %% [markdown]
# ## Labels

# %%
# %%time
plot_order = ['t1', 'GluMI', 't2', 't3a', 't3b', 't4', 't5o', 't5t', 't5i', 'XBC', 't7', 't6', 't8', 't9', 'RBC']

# %%
# %%time
sns.countplot(df, y='celltype_final', order=plot_order);

# %% [markdown]
# ## TSNE

# %%
# %%time
emb_cols_norm = [c for c in df.columns if c.startswith('BCemb_norm')]
emb_cols_norm

# %%
# %%time
emb_feats_norm = df[emb_cols_norm].values
emb_feats_norm.shape

# %%
# %%time
import scanpy as sc

adata = sc.AnnData(df[emb_cols_norm].values)
sc.tl.tsne(adata, perplexity=30, use_rep='X', random_state=0)

# %%
# %%time
df['tsne_d0'] = adata.obsm["X_tsne"][:, 0]
df['tsne_d1'] = adata.obsm["X_tsne"][:, 1]

# %% [markdown]
# ## Select one per type

# %%
# %%time
df.loc[df.post_has_soma == 1, 'soma_rad_um'].hist();

# %%
# %%time
all_types_rows = []

for ct in plot_order:
    rows = df[(df['celltype_final'] == ct) & df.valid_celltype_final & df.post_has_soma & ((df.soma_rad_um > 4) & (df.soma_rad_um < 6))]
    
    if len(rows) == 0:
        rows = None
        print(celltype_final)

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

# %% [markdown]
# # Figures

# %%
# %%time
from style import set_rc_params

set_rc_params()

# %%
# %%time
fig_dir = f'../figures/BC/v{version}'
os.makedirs(fig_dir, exist_ok=True)

# %% [markdown]
# ## Colors

# %%
# %%time
interpolated_colors = colors.interpolate_colors(n=len(plot_order))
ordered_colors = colors.reorder_colors_for_distinction(interpolated_colors, n_neighbors=4)

label2color = {leaf_name: ordered_colors[i] for i, leaf_name in enumerate(plot_order)}

# %%
# %%time
label2color

# %% [markdown]
# ## Dendrogram

# %%
# %%time
cd = ClusterDendrogram(
    data=emb_feats_norm[df.valid_celltype_final],
    cluster_labels=df.loc[df.valid_celltype_final, "celltype_final"],
    feature_names=emb_cols_norm,
)
avg_features = cd.compute_cluster_averages(log_transform=False)
filtered_features, selected_features = cd.filter_features(verbose=False)
dist_matrix = cd.compute_distance_matrix(metric='euclidean')
Z = cd.hierarchical_cluster(method='average')

# %%
# %%time
from scipy.cluster.hierarchy import dendrogram, linkage

fig, ax = plt.subplots(figsize=(4, 1.2))
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

fig.savefig(f'{fig_dir}/bc-dendrogram.svg')
fig.savefig(f'{fig_dir}/bc-dendrogram.png', dpi=600)

# %% [markdown]
# ## Nearest neighbors

# %%
# %%time
import matplotlib.ticker as ticker
from scipy.spatial import cKDTree
from scipy import stats


def add_scale_bar(ax, x_pos, x_pos_text, max_val, n_ticks=3):
    """Draw a vertical scale bar with a single label."""
    import math
    magnitude = 10 ** math.floor(math.log10(max_val)) if max_val > 0 else 1
    scale = round(max_val / magnitude * 0.5) * magnitude / 2
    if scale == 0:
        scale = magnitude / 2
    tick_values = [i * scale for i in range(n_ticks + 1) if i * scale <= max_val * 1.05]
    bar_height = tick_values[-1]
    ax.plot([x_pos, x_pos], [0, bar_height], color="black", solid_capstyle='butt', lw=0.8, clip_on=False)
    ax.text(x_pos_text, 0, f"{int(bar_height)}", ha="right", va="bottom", fontsize=8, rotation=90)


def plot_nnd_histograms(df, celltypes,
                        center_x_col="hull_center_x",
                        center_y_col="hull_center_y",
                        diameter_col="hull_diameter",
                        celltype_col="celltype_final"):

    df = df[df.valid_celltype_final].copy()
    nrows = min(2, len(celltypes) + 1)
    ncols = int(np.ceil((len(celltypes) + 1) / nrows))

    fig, axs = plt.subplots(nrows, ncols, figsize=(7.5, nrows * 1), squeeze=False)
    sns.despine(fig=fig)
    nnd_medians = {}
    dia_medians = {}

    for idx, ct in enumerate(celltypes):
        ax = axs[idx // ncols, idx % ncols]
        sub = df[df[celltype_col] == ct].dropna(subset=[center_x_col, center_y_col])

        # ── NND ──────────────────────────────────────────────────────────
        dists, _ = cKDTree(
            sub[[center_x_col, center_y_col]].values).query(
            sub[[center_x_col, center_y_col]].values, k=2)
        nnd = dists[:, 1]
        nnd_median = np.median(nnd)

        ood = nnd > 3 * nnd_median
        if np.any(ood):
            print(f"Removed {np.sum(ood)} outlier for {ct=}")
            nnd = nnd[~ood]

        # vertical histogram: values on x-axis
        sns.histplot(
            x=nnd, bins=51, ax=ax,
            color="gray", alpha=0.80,
            edgecolor="none", linewidth=0)
        ax.set(xlabel=None, ylabel=None)

        nnd_medians[ct] = np.median(nnd)
        ax.axvline(nnd_median, color="C0",
                   linewidth=1.1, linestyle="--", label="med NND")

        ax.set_xlim(0, np.nanmax(nnd))

        # ── Dendritic field size ─────────────────────────────────────────
        dia_median = np.median(sub[diameter_col].dropna())
        ax.axvline(dia_median, color="C1",
                   linewidth=1.1, linestyle=":", label="med size")

        dia_medians[ct] = dia_median

        ax.set_title(f"{ct}", color=label2color[ct], pad=1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(2))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(2))

        if idx == (ncols - 1):
            ax.legend(frameon=False, loc="upper left", handlelength=1.2, bbox_to_anchor=(1.1, 1))

        xlo, xhi = ax.get_xlim()
        xrange = xhi - xlo
        x_pos = xlo - 0.10 * xrange
        x_pos_text = xlo - 0.12 * xrange
            
        add_scale_bar(ax, x_pos=x_pos, x_pos_text=x_pos_text, max_val=ax.get_ylim()[1], n_ticks=3)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

    for ax in axs[-1, :]:
        ax.set_xlabel("NND [µm]")

    for idx in range(len(celltypes), nrows * ncols):
        axs[idx // ncols, idx % ncols].set_visible(False)

    ax = axs[-1, -1]
    ax.set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    df_scatter = pd.DataFrame({"NND [µm]": nnd_medians, "Size [µm]": dia_medians}).reset_index()
    vmax = np.maximum(df_scatter['Size [µm]'].max(), df_scatter['NND [µm]'].max())

    slope, intercept, r, p, _ = stats.linregress(df_scatter['NND [µm]'], df_scatter['Size [µm]'])
    x_line = np.linspace(df_scatter['NND [µm]'].min(), df_scatter['NND [µm]'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, c='k', ls='--', zorder=-20)

    sns.scatterplot(df_scatter, x="NND [µm]", y="Size [µm]", clip_on=False,
                    hue='index', palette=label2color, legend=False, s=10, ec='none')
    #ax.set(label=None)
    ax.set_title('med', pad=1)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)

    fig.tight_layout(pad=0.4, h_pad=0.9, w_pad=0.4)

    ax.text(-0.4, 1.1, f"r={r:.2f}\np={p:.2g}", transform=ax.transAxes, va='top')

    return fig


fig = plot_nnd_histograms(df[df.valid_celltype_final], celltypes=plot_order)

fig.savefig(f'{fig_dir}/BC_NND.svg', dpi=600, bbox_inches='tight')
fig.savefig(f'{fig_dir}/BC_NND.png', dpi=600, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## TSNE

# %%
# %%time
print((df['valid_celltype_final']).sum())
print((~df['valid_celltype_final']).sum())

# %%
# %%time
plot_embedding(
    df=df,
    label2color=label2color,
    fig_path_prefix=f"{fig_dir}/bc-tsne",
    celltype_col='celltype_final',
    plot_order=plot_order,
    is_labelled_col="valid_celltype_final",
    dot_size=1,
    edgecolor="none",
    rasterized=True,
    legend_markerscale=4,
    figsize=(2.5, 2.3)
)

# %% [markdown]
# ## Feature maps

# %%
# %%time
df.rename({"perc_z_050_3": "perc_z_050", "perc_z_025_3": "perc_z_025"}, axis=1, inplace=True)

# %%
# %%time
example_feature_names = [
    'perc_z_025',
    'perc_z_050',

    'log_radius_median',
    'log_radius_q95',
    
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
    ncols=3,
    clip=False,
    stride=1, 
    figsize_per_cell=(1.3, 1)
)

plt.tight_layout(h_pad=0.3)

fig.savefig(f'{fig_dir}/bc-tnse-features.svg', bbox_inches='tight')
fig.savefig(f'{fig_dir}/bc-tnse-features.png', dpi=600, bbox_inches='tight')

# %% [markdown]
# ## Morph examples

# %%
# %%time
nrows = 2
ncols = int(np.ceil(len(all_types_rows) / nrows))

rad = (all_types_rows.hull_diameter.max() / 2) * 1.3

for i in range(nrows):
    print(i)
    rows = np.array_split(all_types_rows, nrows)[i]
    
    width = 6.9 * rows.shape[0] / ncols
    print(width)
    
    fig, axs = plt.subplots(
        2, len(rows), figsize=(width, 2.3), sharex='all', sharey='row', squeeze=False, height_ratios=(1.7, 4),
        subplot_kw=dict(xlabel=None, ylabel=None, xticks=[], yticks=[]), layout='tight')
    for ax in axs.flat:
        ax.axis('off')
    print(rows.index.astype(int))
    fig, _ = plot_cell_morphologies(
        rows=rows, rad=rad,
        skel_dir=skel_dir,
        size=20, fig=fig, axs=axs,
        color_list=[label2color[ct] for ct in rows['celltype_final'].values],
        show_on_tsne=False
    )

    fig.tight_layout(w_pad=0, h_pad=-0.5)

    if i == 0:
        shift_up = 0.1
        for ax in axs[-1, :]:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + shift_up, pos.width, pos.height])

    
    fig.savefig(f'{fig_dir}/celltype_final-example_set{i}({nrows}).svg', dpi=800, bbox_inches='tight')
    fig.savefig(f'{fig_dir}/celltype_final-example_set{i}({nrows}).png', dpi=800, bbox_inches='tight')
    
    plt.show()

# %% [markdown]
# ## Coverage

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

for name, extent in {'wide': [50, 1150, 50, 1150], 'zoom': [200, 400, 600, 800], }.items(): # 'zoom2': [200, 300, 600, 700], 

    for i in range(nrows):
        celltypes_to_plot = np.array_split(plot_order, nrows)[i]

        plot_bodies = 'zoom' in name.lower()
        
        fig, axs = plt.subplots(ncols=ncols, nrows=2 if plot_bodies else 1,
                                figsize=(8.5, 3 if plot_bodies else 2),
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
            plot_candidate_coverage=True,
            celltypes=celltypes_to_plot,
            fig=fig,
            coverage_axs=axs[0, :],
            resolutions=[1000] * ncols,
            center_axs=axs[1, :] if plot_bodies else None,            
            title_axs=axs[0, :],
            sb_axs=[axs[0, 1]],
            center_x_col="hull_center_x",
            center_y_col="hull_center_y",
            marker_sizes=np.ones(ncols) * (0.1 if 'wide' in name.lower() else 3),
            max_val=max_val,
            extent=extent,
            plot_outlines='zoom' in name.lower(),
            outline_hull=hull_pts if 'wide' in name.lower() else None,
            cb=True,
            sb_size=500 if 'wide' in name.lower() else 100,
        )

        for celltype, ax in zip(celltypes_to_plot, axs[0, :]):
            df_ct = df[(df.celltype_final==celltype) & df.valid_celltype_final]
            n_cells = len(df_ct)
            n_cells_clf = sum(df_ct.celltype_final_decision == 'classifier')
            n_cells_label = n_cells - n_cells_clf
            if n_cells_clf > 0:
                ax.set_title(f"{celltype}\n(n={n_cells_label} [+{n_cells_clf}])", fontsize=7)
            else:
                ax.set_title(f"{celltype}\n(n={n_cells_label})", fontsize=7)
        
        fig.savefig(f'{fig_dir}/bc-{celltypes_to_plot}_set{i}({nrows})_{extent}.svg', dpi=600, bbox_inches='tight')
        fig.savefig(f'{fig_dir}/bc-{celltypes_to_plot}_set{i}({nrows})_{extent}.png', dpi=600, bbox_inches='tight')
        plt.show()

# %% [markdown]
# # Website

# %%
# %%time
website_dir = f'../website/v{version}'
os.makedirs(website_dir, exist_ok=True)

# %%
# %%time
website_cols = ['celltype_final', 'soma_x_um', 'soma_y_um', 'soma_z_um']

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
for celltype_final in plot_order:
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
df_website_single.to_csv(os.path.join(website_dir, f"BC_website_single_examples.csv"))

# %% [markdown]
# ## Mosaics

# %%
# %%time
# Polygon vertices in order
polygon = np.array([xy0, xy1, xy2, xy3])

def is_in_box(hull_center_x, hull_center_y):
    """
    Returns True if the point is inside the quadrilateral.
    Uses ray casting.
    """
    x, y = hull_center_x, hull_center_y
    inside = False

    n = len(polygon)
    p1x, p1y = polygon[0]

    for i in range(n + 1):
        p2x, p2y = polygon[i % n]

        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

                if p1x == p2x or x <= xinters:
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside

# Filter dataframe
df_in_box = df[
    df.apply(
        lambda row: is_in_box(row["hull_center_x"], row["hull_center_y"]),
        axis=1
    )
]

df_website_mosaic = df_in_box.loc[df_in_box.valid_celltype_final & df_in_box.celltype_final.isin(plot_order), website_cols].copy()
df_website_mosaic.sort_values('celltype_final', inplace=True)
df_website_mosaic.head()

# %%
# %%time
sns.countplot(data=df_website_mosaic, x='celltype_final');

# %%
# %%time
df_website_mosaic.to_csv(os.path.join(website_dir, f"BC_website_mosaic_examples.csv"))

# %%
# %%time
from watermark import watermark
print(watermark())
