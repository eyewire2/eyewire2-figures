import numpy as np
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt, patheffects

FEAT2NAME = {
    'log_xys_dens_mean': 'log(mean-density)',
    'log_hull_perimeter': 'log(hull-perimeter)',
    'log_hull_diameter': 'log(hull-diameter)',
    'log_radius_mean': 'log(mean-radius)',
    'log_radius_median': 'log(median-radius)',
    'log_radius_q95': 'log(q95-radius)',
    'log_soma_rad_um': 'log(soma-radius)',
    'log_tips': 'log(tips)',
    'log_branch_points': 'log(branch-points)',
    'log_tortuosity_median': 'log(med-tortuosity)',
    'perc_z_005': 'z-density-q5',
    'perc_z_025': 'z-density-q25',
    'perc_z_050': 'z-density-q50',
    'perc_z_095': 'z-density-q95',
    'log_n_ribbons': 'log(# ribbons)',
}
for i in range(20):
    if f'norm-z-PC{i}' not in FEAT2NAME:
        FEAT2NAME[f'norm-z-PC{i}'] = f'z-PC{i}'


def plot_feats(
        all_feats, all_emb, all_feat_names, figsize=(7.2, 1.5), nrows=5, ncols=6,
        stride=1, clip=2, cmap='PRGn', s=0.1):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axs = axs.flatten()
    plot_order = np.random.permutation(np.arange(all_feats.shape[0]))[::stride]
    for i, feat in enumerate(all_feat_names):
        if i >= len(axs):
            break
        ax = axs[i]
        x_feat = all_feats[:, i]
        q5_q95 = (-2, +2) if clip else np.percentile(x_feat, q=[5, 95])
        scatter = ax.scatter(*all_emb[plot_order, :].T, s=s, lw=0, c=x_feat[plot_order],
                             vmin=q5_q95[0], vmax=q5_q95[1], cmap=cmap, rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(FEAT2NAME.get(feat, feat), fontsize=8)
        ax.axis('off')
        cbar = plt.colorbar(mappable=scatter, shrink=0.4, ticks=q5_q95)
        cbar.ax.set_yticklabels([f'<{q5_q95[0]:.2g}', f'>{q5_q95[1]:.2f}'])
        cbar.outline.set_visible(False)
    for ax in axs.flat[len(all_feat_names):]:
        ax.axis('off')
    plt.tight_layout()
    return fig, axs


def save_and_plot_feats(
        all_feats, all_emb, all_feat_names, fig_dir=None, file_prefix=None,
        ncols=2, clip=False, stride=1, cmap='PRGn',
        figsize_per_cell=(1.5, 1.5), s=0.1
):
    nrows = int(np.ceil(len(all_feat_names) / ncols))
    figsize = (figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows)
    fig, axs = plot_feats(
        all_feats, all_emb, all_feat_names,
        nrows=nrows, ncols=ncols, clip=clip,
        stride=stride, cmap=cmap, figsize=figsize, s=s)
    if fig_dir is not None and file_prefix is not None:
        fig.savefig(f'{fig_dir}/{file_prefix}.svg', bbox_inches='tight')
        fig.savefig(f'{fig_dir}/{file_prefix}.png', dpi=600, bbox_inches='tight')
    return fig, axs


def plot_embedding(
        df,
        label2color,
        fig_path_prefix,
        celltype_col,
        plot_order,
        is_labelled_col="is_labelled",
        tsne_x_col="tsne_d0",
        tsne_y_col="tsne_d1",
        annotate=True,
        figsize=(4, 4),
        dot_size=12,
        dot_alpha=0.8,
        edgecolor="k",
        rasterized=False,
        legend_markerscale=None,
        random_seed=42,
):
    """
    Plot a t-SNE embedding with optional per-cell-type annotation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing t-SNE coordinates, cell type labels, and a
        labelled/unlabelled flag.
    label2color : dict
        Mapping from cell-type label to color used for the scatter palette.
    fig_path_prefix : str
        File path prefix (including directory) for saved figures, e.g.
        ``'figures/rgc-tsne'``. Both ``.svg`` and ``.png`` are saved.
    celltype_col : str
        Name of the column in ``df`` that holds cell-type labels.
    plot_order : array-like
        Ordered sequence of cell-type labels that controls hue ordering in
        the legend and the annotation loop.
    is_labelled_col : str, optional
        Boolean column in ``df`` marking labelled points. Default ``'is_labelled'``.
    tsne_x_col : str, optional
        Column name for the t-SNE x-axis. Default ``'tsne_d0'``.
    tsne_y_col : str, optional
        Column name for the t-SNE y-axis. Default ``'tsne_d1'``.
    annotate : bool, optional
        Whether to draw per-cluster text labels. Default ``True``.
    figsize : tuple, optional
        Figure size as ``(width, height)``. Default ``(4, 4)``.
    dot_size : float, optional
        Marker size (``s``) for scatter points. Default ``12``.
    dot_alpha : float, optional
        Opacity of labelled scatter points. Default ``0.8``.
    edgecolor : str or None, optional
        Edge colour for labelled scatter points. Default ``'k'`` (black).
        Pass ``'none'`` to suppress edges.
    rasterized : bool, optional
        Rasterize scatter layers (useful for large datasets). Default ``False``.
    legend_markerscale : float or None, optional
        Scale factor for legend markers. ``None`` uses matplotlib's default.
    random_seed : int, optional
        NumPy random seed for reproducible point ordering. Default ``42``.
    """
    np.random.seed(random_seed)

    x = df[tsne_x_col].values
    y = df[tsne_y_col].values
    labels = df[celltype_col]
    is_labelled = df[is_labelled_col].values
    label_order = np.asarray(plot_order)

    assert len(set(label_order) - set(labels)) == 0, (
        "plot_order contains labels not present in the data"
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_aspect('equal', adjustable='datalim')

    # ── unlabelled points (background, light grey) ──────────────────────────
    if np.any(~is_labelled):
        sns.scatterplot(
            x=x[~is_labelled],
            y=y[~is_labelled],
            ax=ax,
            alpha=1,
            color="lightgray",
            edgecolor="none",
            legend=False,
            marker="o",
            s=dot_size,
            rasterized=rasterized,
        )

    # ── labelled points (coloured by cell type) ──────────────────────────────
    rand_order = np.random.permutation(np.where(is_labelled)[0])
    sns.scatterplot(
        x=x[rand_order],
        y=y[rand_order],
        hue=labels[rand_order],
        hue_order=label_order,
        ax=ax,
        marker="o",
        s=dot_size,
        alpha=dot_alpha,
        palette=label2color,
        edgecolor=edgecolor,
        rasterized=rasterized,
    )

    # ── per-cluster text annotations ─────────────────────────────────────────
    if annotate:
        texts = []
        for label in label_order:
            is_ct = (labels == label).values
            if not np.any(is_ct):
                continue
            emb_ct = np.stack([x[is_ct], y[is_ct]]).T
            ct_mean = np.mean(emb_ct, axis=0)
            ct_dists = np.sum((emb_ct - ct_mean) ** 2, axis=1) ** 0.5
            pos_text = emb_ct[np.argmin(ct_dists)]
            text = ax.text(
                *pos_text,
                label,
                ha="center",
                va="center",
                c="k",
                fontsize=6,
                path_effects=[
                    patheffects.withStroke(linewidth=2, foreground="w", alpha=0.9)
                ],
            )
            texts.append(text)
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k"))

    # ── legend ────────────────────────────────────────────────────────────────
    legend_kwargs = dict(bbox_to_anchor=(1, 1), loc="upper left", frameon=False, ncols=1)
    if legend_markerscale is not None:
        legend_kwargs["markerscale"] = legend_markerscale
    ax.legend(**legend_kwargs)

    # ── save ──────────────────────────────────────────────────────────────────
    plt.savefig(f"{fig_path_prefix}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(f"{fig_path_prefix}.png", dpi=600, bbox_inches="tight")
    plt.show()