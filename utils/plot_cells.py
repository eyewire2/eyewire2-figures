from typing import Optional

import dataclasses
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle

import seaborn as sns

import skeliner as sk
from skeliner.plot.vis2d import _project

import colors

_PLANE_AXES = {
    "xy": (0, 1),
    "yx": (1, 0),
    "xz": (0, 2),
    "zx": (2, 0),
    "yz": (1, 2),
    "zy": (2, 1),
}


def _radii_to_sizes(rr: np.ndarray, ax: Axes) -> tuple[np.ndarray, float]:
    """
    Convert radii (data units) -> scatter sizes (points**2) so that the same
    physical radius is rendered identically in every subplot.

    matplotlib's scatter ``s`` is the squared *diameter* in points for the
    default circular marker (the unit-diameter marker path is scaled by
    ``sqrt(s)``), not the marker's area -- so ``s`` must be ``diameter_pt**2``,
    not ``pi * r_pt**2`` (the latter under-sizes markers by a factor of
    ``sqrt(pi)/2`` in radius, i.e. ``pi/4`` in area).
    """
    fig = ax.figure
    dpi = fig.dpi

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox = ax.get_window_extent()

    ppd_x = bbox.width / abs(x1 - x0)
    ppd_y = bbox.height / abs(y1 - y0)
    ppd = min(ppd_x, ppd_y)

    r_px = rr * ppd
    r_pt = r_px * 72.0 / dpi
    return (2 * r_pt) ** 2, ppd


def skeliner_projection(
    skel: sk.Skeleton,
    *,
    plane: str = "xy",
    radius_metric: str | None = None,
    scale: float = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    rasterized: bool = True,
    ax: Axes | None = None,
    edge_lw: float = 0.2,
    circle_alpha: float = 1.0,
    unit: str | None = None,
    node_color: str | tuple = "red",
    soma_color: str | tuple = "black",
    node_lw: float = 0.5,
    soma_style: str = "dashed",  # "dashed" | "filled"
    soma_marker_size: float = 15,
    soma_linewidth: float = 0.8,
    show_soma: bool = True,
) -> tuple[Figure, Axes]:
    """Orthographic 2-D projection of a skeleton (node circles, edges, soma marker).

    Parameters
    ----------
    skel : skeliner.Skeleton
        The centre-line skeleton to visualise.
    plane : {"xy", "xz", "yz", "yx", "zx", "zy"}
        Projection plane.
    scale : float, default 1
        Multiplicative scale applied to node coordinates and radii.
    xlim, ylim : (min, max) or None
        Spatial extent to crop to and set as axis limits.
    rasterized : bool
        Rasterize skeleton glyphs.
    ax : matplotlib.axes.Axes | None
        Existing Axes to draw into. When None, a new figure is created.
    edge_lw : float
        Line width of skeleton edges.
    node_lw : float
        Line width of skeleton node circles.
    circle_alpha : float
        Transparency of skeleton node circles.
    unit : str | None
        Axis-label unit.
    node_color, soma_color : str or tuple
        Color of the skeleton node circles and of the soma marker, respectively.
    soma_style : str
        How to plot the soma, currently supported styles are:
        - "dashed" : dashed circle outline (default)
        - "filled" : filled circle with soma colour
    soma_marker_size : float
        Size (``s``) of the small centre-dot marker drawn for ``soma_style="dashed"``.
    soma_linewidth : float
        Line width of the soma circle outline.
    show_soma : bool
        Whether to draw the soma marker/circle at all. Set to False when the
        soma centre falls outside the plotted window.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {tuple(_PLANE_AXES)}")

    ix, iy = _PLANE_AXES[plane]

    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]

    if unit is None:  # try to grab from metadata
        unit = skel.meta.get("unit", None)

    xy_skel = _project(skel.nodes, ix, iy) * scale
    rr = skel.radii[radius_metric] * scale

    def _crop_window(xy: np.ndarray) -> np.ndarray:
        keep = np.ones(len(xy), dtype=bool)
        if xlim is not None:
            keep &= (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        if ylim is not None:
            keep &= (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        return keep

    keep_skel = _crop_window(xy_skel)
    xy_skel = xy_skel[keep_skel]
    rr = rr[keep_skel]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # skeleton node circles
    if xy_skel.size:
        # limits need to be defined before converting radii -> scatter sizes
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ax.set_xlim((xy_skel[:, 0].min(), xy_skel[:, 0].max()))
            ax.set_ylim((xy_skel[:, 1].min(), xy_skel[:, 1].max()))

        ax.set_aspect(1)
        sizes, _ppd = _radii_to_sizes(rr, ax)

        # node 0 is the soma centre, drawn separately below
        ax.scatter(
            xy_skel[1:, 0],
            xy_skel[1:, 1],
            s=sizes[1:],
            facecolors=node_color,
            edgecolors=node_color,
            linewidths=node_lw,
            alpha=circle_alpha,
            zorder=1,
            rasterized=rasterized,
        )

    # soma marker
    if show_soma:
        c_xy = _project(skel.nodes[[0]] * scale, ix, iy).ravel()

        if soma_style == 'filled':
            soma_fc = soma_color
            soma_ec = 'k'
            soma_ls = '-'
        else:
            ax.scatter(*c_xy, color="black", s=soma_marker_size, zorder=3)
            soma_fc = 'none'
            soma_ec = 'k'
            soma_ls = '--'

        soma_circle = Circle(
            c_xy,
            skel.soma.equiv_radius * scale,
            facecolor=soma_fc,
            edgecolor=soma_ec,
            linewidth=soma_linewidth,
            linestyle=soma_ls,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(soma_circle)

    # skeleton edges
    if skel.edges.size:
        ekeep = keep_skel[skel.edges[:, 0]] & keep_skel[skel.edges[:, 1]]
        edges_kept = skel.edges[ekeep]
        if edges_kept.size:
            # original -> compressed index map
            idx_map = -np.ones(len(keep_skel), int)
            idx_map[np.flatnonzero(keep_skel)] = np.arange(keep_skel.sum())

            seg_start = xy_skel[idx_map[edges_kept[:, 0]]]
            seg_end = xy_skel[idx_map[edges_kept[:, 1]]]
            segments = np.stack((seg_start, seg_end), axis=1)

            lc = LineCollection(
                segments.tolist(),
                colors="black",
                linewidths=edge_lw,
                alpha=1.0,
                zorder=2,
                rasterized=rasterized,
                capstyle='round',
                joinstyle='round',
            )
            ax.add_collection(lc)

    if plane in ['xy', 'yx']:
        ax.set_aspect('equal', adjustable='box')

    unit_str = "" if unit is None else f"({unit})"
    ax.set_xlabel(f"{plane[0]} {unit_str}")
    ax.set_ylabel(f"{plane[1]} {unit_str}")

    # guarantee limits if user requested specific window
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig, ax


def _prune_to_window(
        skel: sk.Skeleton,
        rad: float,
        zlim: tuple[float, float],
) -> sk.Skeleton:
    """Drop nodes whose (x, y, z) falls outside the plotted x/y/z window.

    ``skeliner_projection`` only crops nodes on the two axes of the plane
    being drawn, so e.g. a neurite cropped out of the xy view (a long RGC
    axon leaving the window in y) would still show up in the xz view, and
    one cropped out of the xz view in z would still show up in the xy view.
    Pruning on x/y/z up front keeps both views restricted to the same set
    of nodes.
    """
    keep = (
        (skel.nodes[:, 0] >= -rad) & (skel.nodes[:, 0] <= rad) &
        (skel.nodes[:, 1] >= -rad) & (skel.nodes[:, 1] <= rad) &
        (skel.nodes[:, 2] >= zlim[0]) & (skel.nodes[:, 2] <= zlim[1])
    )
    keep[0] = True  # soma node

    idx_map = -np.ones(len(keep), dtype=int)
    idx_map[np.flatnonzero(keep)] = np.arange(keep.sum())

    ekeep = keep[skel.edges[:, 0]] & keep[skel.edges[:, 1]]
    edges = idx_map[skel.edges[ekeep]]

    return dataclasses.replace(
        skel,
        nodes=skel.nodes[keep],
        radii={k: v[keep] for k, v in skel.radii.items()},
        edges=edges,
        ntype=skel.ntype[keep] if skel.ntype is not None else None,
    )


def get_skel_center(
        center: str | tuple[float, float] | None,
        skel: sk.Skeleton,
        density: np.ndarray | None = None,
        nodes: np.ndarray | None = None,
) -> tuple[float, float]:
    if density is None or nodes is None:
        from pywarper.warpers import segment_lengths

        density, nodes = segment_lengths(skel)

    if isinstance(center, str):
        if center == "soma":
            center = [skel.soma.center[0], skel.soma.center[1]]
        elif center == "tree":
            # get weighted center of mass of xy positions
            mask = skel.ntype == 3
            total_mass = density[mask].sum()
            if total_mass == 0:
                center = (0.0, 0.0)
            else:
                x_cm = (nodes[mask, 0] * density[mask]).sum() / total_mass
                y_cm = (nodes[mask, 1] * density[mask]).sum() / total_mass
                center = (x_cm, y_cm)
        else:
            raise ValueError("center string must be one of {'soma','tree'}")
    elif center is None:
        center = (0.0, 0.0)
    else:
        center = (float(center[0]), float(center[1]))

    return center


def plot_cell_morphologies(
        rows: pd.DataFrame,
        rad: float,
        skel_dir: str = '../data/',
        color: str | tuple = 'k',
        color_list: Optional[list[str | tuple]] = None,
        size: Optional[float] = None,
        show_on_tsne: bool = True,
        fig: Optional[plt.Figure] = None,
        axs: Optional[np.ndarray] = None,
        sb_fontsize: float = 10,
        zlim: tuple[float, float] = (-30, +50),
        edge_lw: float = 0.2,
        node_lw: float = 0.5,
        soma_marker_size: float = 15,
        soma_linewidth: float = 0.8,
        # --- tSNE plot parameters (only used if show_on_tsne=True) ---
        df: Optional[pd.DataFrame] = None,
        is_labelled: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        label_order: Optional[list] = None,
        celltype2color: Optional[dict] = None,
        # --- color lookup for reference lines ---
        on_sac_color: str | tuple = colors.cellclass2color['ON SAC'],
        off_sac_color: str | tuple = colors.cellclass2color['OFF SAC'],
) -> tuple[plt.Figure, Optional[plt.Figure]]:
    """
    Plot skeletal projections for a set of rows, with an optional tSNE overlay.

    Each row in `rows` corresponds to one column of subplots. Two projection
    planes are shown per skeleton: xz (top row) and xy (bottom row).
    Optionally, a second figure shows where the rows fall in tSNE space.

    Parameters
    ----------
    rows : pd.DataFrame
        DataFrame whose rows each represent one cell to plot. Must contain
        a ``skel_file`` column with paths to .npz skeleton files, and
        ``tsne_d0`` / ``tsne_d1`` columns if ``show_on_tsne=True``.
    rad : float
        Half-width of the spatial window (in µm) for x and y axes.
    color : str or tuple, optional
        Default color for all skeletons. Overridden per-skeleton by
        ``color_list``. Default is ``'k'`` (black).
    color_list : list of (str or tuple), optional
        Per-skeleton colors, indexed by column position. If provided,
        overrides ``color`` for each skeleton. Length must match ``len(rows)``.
    size : float, optional
        Length of the scale bar in µm. Defaults to ``rad`` if not provided.
    show_on_tsne : bool, optional
        If True, produce a second figure showing the tSNE embedding with the
        selected rows highlighted. Requires ``df``, ``is_labelled``,
        ``labels``, ``label_order``, and ``celltype2color``. Default is True.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into. Must be provided together with ``axs``.
    axs : np.ndarray of matplotlib.axes.Axes, optional
        Array of shape ``(2, len(rows))`` to draw into. Must be provided
        together with ``fig``.
    sb_fontsize : float, optional
        Font size for the scale bar label. Default is 10.
    zlim : tuple of (float, float), optional
        Y-axis limits for the xz projection plane, in µm. Default is
        ``(-30, +50)``.
    edge_lw : float, optional
        Line width of skeleton edges, passed to ``skeliner_projection``.
    node_lw : float, optional
        Line width of skeleton node circles, passed to ``skeliner_projection``.
    soma_marker_size : float, optional
        Size of the soma centre-dot marker, passed to ``skeliner_projection``.
    soma_linewidth : float, optional
        Line width of the soma circle outline, passed to ``skeliner_projection``.
    df : pd.DataFrame, optional
        Full embedding DataFrame containing ``tsne_d0``, ``tsne_d1`` columns
        for background scatter. Required when ``show_on_tsne=True``.
    is_labelled : np.ndarray of bool, optional
        Boolean mask over ``df`` indicating which points have a label and
        should appear in the tSNE scatter. Required when ``show_on_tsne=True``.
    labels : np.ndarray, optional
        Label array (same length as ``df``) used for hue coloring in the
        tSNE scatter. Required when ``show_on_tsne=True``.
    label_order : list, optional
        Ordered list of label values for consistent legend ordering.
        Required when ``show_on_tsne=True``.
    celltype2color : dict, optional
        Mapping from label/cell-type name to color, used as the tSNE scatter
        palette. Required when ``show_on_tsne=True``.
    on_sac_color : str or tuple, optional
        Color for the ON SAC reference line (drawn at z=0).
    off_sac_color : str or tuple, optional
        Color for the OFF SAC reference line (drawn at z=12).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The main figure containing skeleton projection subplots.
    fig2 : matplotlib.figure.Figure or None
        The tSNE figure if ``show_on_tsne=True``, otherwise ``None``.

    Raises
    ------
    AssertionError
        If ``axs`` is provided without ``fig``.
    ValueError
        If ``show_on_tsne=True`` but any of ``df``, ``is_labelled``,
        ``labels``, ``label_order``, or ``celltype2color`` are not provided.

    Examples
    --------
    >>> fig, fig2 = plot_cell_morphologies(
    ...     rows=selected_df,
    ...     rad=50,
    ...     color_list=['steelblue', 'tomato', 'forestgreen'],
    ...     show_on_tsne=False,
    ... )
    >>> fig.savefig('projections.pdf')
    """
    if show_on_tsne:
        missing = [
            name for name, val in [
                ('df', df), ('is_labelled', is_labelled), ('labels', labels),
                ('label_order', label_order), ('celltype2color', celltype2color),
            ] if val is None
        ]
        if missing:
            raise ValueError(
                f"show_on_tsne=True requires the following arguments: {missing}"
            )

    ncols = len(rows)
    if size is None:
        size = rad

    if axs is None:
        fig, axs = plt.subplots(
            2, ncols,
            figsize=(ncols, 2.2),
            sharex='all',
            sharey='row',
            squeeze=False,
            height_ratios=(1.7, 4),
            subplot_kw=dict(xlabel=None, ylabel=None, xticks=[], yticks=[]),
            layout='tight',
        )
    else:
        assert fig is not None

    sns.despine(top=True, bottom=True, left=True, right=True)

    skels = [sk.io.load_swc(os.path.join(skel_dir, f"{cell}.swc"))
             for cell in rows.index]

    for i, skel in enumerate(skels):
        center = get_skel_center('tree', skel)
        skel.nodes[:, 0] -= center[0]
        skel.nodes[:, 1] -= center[1]
        soma_in_bounds = (
            -rad <= skel.nodes[0, 0] <= rad and
            -rad <= skel.nodes[0, 1] <= rad and
            zlim[0] <= skel.nodes[0, 2] <= zlim[1]
        )
        skel = _prune_to_window(skel, rad, zlim)

        for j, plane in enumerate(['xz', 'xy']):
            ax = axs[j, i]
            current_color = color_list[i] if color_list is not None else color

            skeliner_projection(
                skel, ax=ax, plane=plane,
                circle_alpha=1,
                node_color=current_color,
                soma_color=current_color,
                xlim=(-rad, +rad),
                ylim=(-rad, +rad) if plane == 'xy' else zlim,
                edge_lw=edge_lw,
                node_lw=node_lw,
                soma_marker_size=soma_marker_size,
                soma_linewidth=soma_linewidth,
                show_soma=soma_in_bounds,
            )
            ax.set_xlim(-rad, +rad)

            if plane == 'xy':
                ax.set_ylim(-rad, +rad)
                ax.set_box_aspect(1)
                if i == 0:
                    ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

            elif plane[1] == 'z':
                ylim = zlim
                ax.set_ylim(ylim)
                ax.set_aspect('auto')
                soma_z = skel.soma.center[2]
                y0 = ylim[1] - 10 if soma_z < 0 else ylim[0] + 10
                y0_text = y0 + 3 if soma_z < 0 else y0 - 3
                va_text = 'bottom' if soma_z < 0 else 'top'
                if i == 0:
                    ax.plot(
                        [-size / 2, size / 2], [y0, y0],
                        c='k', solid_capstyle='butt', lw=0.8, clip_on=False,
                    )
                    ax.text(
                        0, y0_text, f'{size} µm',
                        c='k', ha='center', va=va_text, fontsize=sb_fontsize,
                    )
                ax.axhline(0, ls='-', c=on_sac_color, lw=0.8, zorder=-5)
                ax.axhline(12, ls='-', c=off_sac_color, lw=0.8, zorder=-5)
                if i == 0:
                    ax.spines['left'].set_visible(True)

    if show_on_tsne:
        fig2, ax2 = plt.subplots(1, 1, figsize=(3, 3))
        x = df.tsne_d0.values
        y = df.tsne_d1.values
        rand_order = np.random.permutation(np.where(is_labelled)[0])[::3]
        sns.scatterplot(
            ax=ax2,
            x=x[rand_order],
            y=y[rand_order],
            hue=labels[rand_order],
            hue_order=label_order,
            style=None,
            style_order=np.random.permutation(label_order),
            marker='o',
            alpha=0.3,
            palette=celltype2color,
            edgecolor='none',
        )
        for i, row in rows.iterrows():
            print(f"{row.tsne_d0:.1f} {row.tsne_d1:.1f}")
            ax2.plot(row.tsne_d0, row.tsne_d1, 'rX', ms=10, mfc='none')
        ax2.legend(bbox_to_anchor=(0.95, 1), loc='upper left', frameon=False)
    else:
        fig2 = None

    return fig, fig2
