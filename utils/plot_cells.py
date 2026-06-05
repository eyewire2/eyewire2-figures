from typing import Optional, Sequence

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle

import seaborn as sns
from scipy.stats import binned_statistic_2d

import skeliner as sk
from skeliner.plot.vis2d import _resolve_swc_palette_from_skel_cmap, _project, _as_cmap, _radii_to_sizes, \
    _soma_ellipse2d, _trapezoid_3d

import colors

_PLANE_AXES = {
    "xy": (0, 1),
    "yx": (1, 0),
    "xz": (0, 2),
    "zx": (2, 0),
    "yz": (1, 2),
    "zy": (2, 1),
}

_PLANE_NORMAL = {
    "xy": np.array([0, 0, 1.0]),
    "yx": np.array([0, 0, 1.0]),
    "xz": np.array([0, 1.0, 0]),
    "zx": np.array([0, 1.0, 0]),
    "yz": np.array([1.0, 0, 0]),
    "zy": np.array([1.0, 0, 0]),
}


def skeliner_projection(
    skel: sk.Skeleton,
    mesh = None,
    *,
    plane: str = "xy",
    radius_metric: str | None = None,
    bins: int | tuple[int, int] = 800,
    scale: float = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    draw_skel: bool = True,
    draw_mesh: bool = True,
    draw_edges: bool = True,
    draw_cylinders: bool = False,
    rasterized: bool = True,
    ax: Axes | None = None,
    mesh_cmap = "Blues",  # mesh color map
    skel_cmap = "Pastel2",  # skeleton color map
    vmax_fraction: float = 0.10,
    edge_lw: float = 0.5,
    circle_alpha: float = 0.25,
    cylinder_alpha: float = 0.5,
    highlight_nodes: int | Sequence[int] | None = None,
    highlight_face_alpha: float = 0.5,
    unit: str | None = None,
    # soma --------------------------------------------------------------- #
    draw_soma_mask: bool = True,
    soma_style: str = "dashed",  # "dashed" | "filled"
    # colors
    color_by: str = "fixed",  # "ntype" or "fixed"
) -> tuple[Figure, Axes]:
    """Orthographic 2‑D overview of a skeleton with an **optional** mesh‑density
    background.

    Parameters
    ----------
    skel : skeliner.Skeleton
        The centre‑line skeleton to visualise.
    mesh : trimesh.Trimesh | None, default *None*
        Surface mesh used to draw a vertex‑density heat‑map and (optionally) the
        soma surface.  Pass *None* to **omit** the background histogram and any
        mesh‑based overlays.
    plane : {"xy", "xz", "yz", "yx", "zx", "zy"}
        Projection plane.
    bins : int | (int,int), default *800*
        Resolution of the background histogram.  Ignored when *mesh* is *None*.
    scale : float | (float, float), default *1*
        Multiplicative scale(s).  Either a scalar applied to both skeleton and
        mesh or a pair ``(s_skel, s_mesh)``.
    xlim, ylim : (min, max) or *None*
        Spatial extent **before** plotting.  If not given, limits are inferred
        from the histogram (when *mesh* is available) or the skeleton.
    draw_skel, draw_mesh, draw_edges, draw_cylinders: bool
        Toggles for skeleton glyphs.
    soma_style : str
        How to plot the soma, currently supported styles are:
        - "dashed" : dashed ellipse outline (default)
        - "filled" : filled circle with soma colour
    rasterized : bool | int
        Rasterize skeleton. If int and > 1, not only skeleton will be rasterized.
    ax : matplotlib.axes.Axes | None
        Existing *Axes* to draw into.  When *None*, a new figure is created.
    mesh_cmap, vmax_fraction : appearance of the histogram – see original docs.
    circle_alpha, cylinder_alpha : transparencies of skeleton glyphs.
    highlight_nodes : node IDs to highlight.
    unit : str | None
        Axis‑label unit.
    draw_soma_mask : bool, default *True*
        Draw the soma shell when both *mesh* **and** soma vertices are
        available.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # ───────────────────────────────── validation & setup ──────────────────
    if plane not in _PLANE_AXES:
        raise ValueError(f"plane must be one of {tuple(_PLANE_AXES)}")

    ix, iy = _PLANE_AXES[plane]

    # normalise *scale* → [skel_scale, mesh_scale]
    if not isinstance(scale, Sequence):
        scale = [scale, scale]
    if len(scale) != 2:
        raise ValueError("scale must be a scalar or a pair of two scalars")
    scl_skel, scl_mesh = map(float, scale)

    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]

    if unit is None:  # try to grab from metadata
        unit = skel.meta.get("unit", None)

    highlight_set = (
        set(map(int, np.atleast_1d(highlight_nodes)))
        if highlight_nodes is not None
        else set()
    )

    # ─────────────────────────────colormap ─────────────
    swc_colors = _resolve_swc_palette_from_skel_cmap(skel_cmap)

    # ───────────────────────────── project (and optionally crop) ───────────
    xy_skel = _project(skel.nodes, ix, iy) * scl_skel
    rr = skel.radii[radius_metric] * scl_skel

    if mesh is not None and draw_mesh:
        xy_mesh = _project(mesh.vertices, ix, iy) * scl_mesh
    else:  # empty placeholder for unified code‑path
        xy_mesh = np.empty((0, 2), dtype=float)

    # helper – applies *xlim/ylim* cropping on a 2‑column array
    def _crop_window(xy: np.ndarray) -> np.ndarray:
        keep = np.ones(len(xy), dtype=bool)
        if xlim is not None:
            keep &= (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        if ylim is not None:
            keep &= (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        return keep

    # crop *before* heavy lifting
    keep_skel = _crop_window(xy_skel)
    keep_mask = keep_skel  # ← keep the original name for edges
    idx_keep = np.flatnonzero(keep_mask)  # 1-D array of kept node IDs
    xy_skel = xy_skel[keep_mask]  # already done
    rr = rr[keep_mask]  # already done

    # colour array for the *kept* nodes
    if color_by == "ntype" and skel.ntype is not None:
        col_nodes = swc_colors[skel.ntype[idx_keep]]
    else:
        col_nodes = "red"

    if mesh is not None and xy_mesh.size and draw_mesh:
        keep_mesh = _crop_window(xy_mesh)
        xy_mesh = xy_mesh[keep_mesh]

    # ─────────────────────────────── histogram (mesh may be None) ──────────
    if mesh is not None and xy_mesh.size and draw_mesh:
        # ensure bins argument correct
        if isinstance(bins, int):
            bins_arg: int | tuple[int, int] = bins
        elif (
                isinstance(bins, tuple)
                and len(bins) == 2
                and all(isinstance(b, int) for b in bins)
        ):
            bins_arg = (int(bins[0]), int(bins[1]))
        else:
            raise ValueError("bins must be an int or a tuple of two ints")

        hist, xedges, yedges, _ = binned_statistic_2d(
            xy_mesh[:, 0],
            xy_mesh[:, 1],
            None,
            statistic="count",
            bins=bins_arg,
        )
        hist = hist.T  # imshow expects (rows = y)
    else:
        hist = None

    # ───────────────────────────── figure / axes boilerplate ───────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # background image – only when we do have a histogram
    if hist is not None and draw_mesh:
        ax.imshow(
            hist,
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            origin="lower",
            cmap=_as_cmap(mesh_cmap),
            vmax=hist.max() * vmax_fraction,
            alpha=1.0,
            rasterized=rasterized > 1,
        )

    # ──────────────────────── draw skeleton circles (always) ───────────────
    if draw_skel and xy_skel.size:
        # limits need to be defined before converting radii → scatter sizes
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        # elif hist is None:  # fallback to skeleton extents
        else:
            ax.set_xlim((xy_skel[:, 0].min(), xy_skel[:, 0].max()))
            ax.set_ylim((xy_skel[:, 1].min(), xy_skel[:, 1].max()))

        ax.set_aspect(1)
        sizes, _ppd = _radii_to_sizes(rr, ax)

        ax.scatter(
            xy_skel[:, 0][1:],
            xy_skel[:, 1][1:],
            s=sizes[1:],
            facecolors="none",
            edgecolors=col_nodes[1:]
            if isinstance(col_nodes, np.ndarray)
            else col_nodes,
            linewidths=1.0,
            alpha=circle_alpha,
            zorder=2,
            rasterized=rasterized > 0,
        )

        # highlighted nodes – filled circles
        if highlight_set:
            orig_ids = np.flatnonzero(keep_skel)
            hilite_mask = np.isin(orig_ids, list(highlight_set))
            if hilite_mask.any():
                ax.scatter(
                    xy_skel[hilite_mask, 0],
                    xy_skel[hilite_mask, 1],
                    s=sizes[hilite_mask],
                    facecolors="green",
                    edgecolors="green",
                    linewidths=0.9,
                    alpha=highlight_face_alpha,
                    zorder=3.5,
                    rasterized=rasterized > 1,
                )

    # ───────────────────────── soma shell & center (if possible) ───────────
    c_xy = _project(skel.nodes[[0]] * scl_skel, ix, iy).ravel()
    col_soma = swc_colors[1] if color_by == "ntype" else "pink"

    if soma_style == 'filled':
        soma_fc = col_soma
        soma_ec = 'k'
        soma_ls = '-'
        soma_mc = 'none'
    else:
        ax.scatter(*c_xy, color="black", s=15, zorder=3)
        soma_fc = 'none'
        soma_ec = 'k'
        soma_ls = '--'
        soma_mc = col_soma

    if (
            mesh is not None
            and skel.soma is not None
            and skel.soma.verts is not None
    ):
        if draw_soma_mask:
            xy_soma = _project(mesh.vertices[np.asarray(skel.soma.verts, int)], ix, iy)
            xy_soma = xy_soma * scl_mesh
            xy_soma = xy_soma[_crop_window(xy_soma)]  # respect crop

            ax.scatter(
                xy_soma[:, 0],
                xy_soma[:, 1],
                s=1,
                c=[soma_mc],
                alpha=0.5,
                linewidths=0,
                label="soma surface",
                rasterized=rasterized > 0,
            )
        # dashed ellipse outline
        ell = _soma_ellipse2d(skel.soma, plane, scale=scl_skel)
        ell.set_edgecolor(soma_ec)
        ell.set_facecolor(soma_fc)
        ell.set_linestyle(soma_ls)
        ell.set_linewidth(0.8)
        ell.set_alpha(0.9)
        ax.add_patch(ell)
    else:
        soma_circle = Circle(
            c_xy,
            skel.soma.equiv_radius * scl_skel,
            facecolor=soma_fc,
            edgecolor=soma_ec,
            linewidth=0.8,
            linestyle=soma_ls,
            alpha=0.9,
        )
        ax.add_patch(soma_circle)

    # ─────────────────────── draw edges & cylinders (unchanged) ────────────
    if draw_skel and skel.edges.size:
        keep = keep_skel  # alias
        if draw_edges:
            ekeep = keep[skel.edges[:, 0]] & keep[skel.edges[:, 1]]
            edges_kept = skel.edges[ekeep]
            if edges_kept.size:
                # original → compressed index map
                idx_map = -np.ones(len(keep), int)
                idx_map[np.flatnonzero(keep)] = np.arange(keep.sum())

                seg_start = xy_skel[idx_map[edges_kept[:, 0]]]
                seg_end = xy_skel[idx_map[edges_kept[:, 1]]]
                segments = np.stack((seg_start, seg_end), axis=1)

                lc = LineCollection(
                    segments.tolist(),
                    colors="black",
                    linewidths=edge_lw,
                    alpha=cylinder_alpha,
                    rasterized=rasterized > 0,
                )
                ax.add_collection(lc)

        if draw_cylinders:
            ekeep = keep_skel[skel.edges[:, 0]] & keep_skel[skel.edges[:, 1]]
            edges_kept = skel.edges[ekeep]
            if edges_kept.size:
                idx_map = -np.ones(len(keep_skel), int)
                idx_map[np.flatnonzero(keep_skel)] = np.arange(keep_skel.sum())

                quads = []
                for n0, n1 in edges_kept:
                    i0, i1 = idx_map[[n0, n1]]
                    quad = _trapezoid_3d(
                        skel.nodes[n0] * scl_skel,
                        skel.nodes[n1] * scl_skel,
                        rr[i0],
                        rr[i1],
                        plane,
                    )
                    if quad is not None:
                        quads.append(quad)

                if quads:
                    # make sure axes limits are already set before adding
                    if xlim is not None:
                        ax.set_xlim(xlim)
                    if ylim is not None:
                        ax.set_ylim(ylim)

                    pc = PolyCollection(
                        quads,
                        facecolors="red",
                        edgecolors="red",
                        alpha=cylinder_alpha,
                        zorder=10,
                        rasterized=rasterized > 0,
                    )
                    ax.add_collection(pc)

    # ────────────────────────────── final cosmetics ────────────────────────
    if plane in ['xy', 'yx']:
        ax.set_aspect('equal', adjustable='box')

    if unit is None:
        unit_str = "" if scl_skel == 1.0 else f"(×{scl_skel:g})"
    else:
        unit_str = f"({unit})"

    ax.set_xlabel(f"{plane[0]} {unit_str}")
    ax.set_ylabel(f"{plane[1]} {unit_str}")

    # guarantee limits if user requested specific window
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig, ax


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

        for j, plane in enumerate(['xz', 'xy']):
            ax = axs[j, i]
            current_color = color_list[i] if color_list is not None else color

            skeliner_projection(
                skel, ax=ax, plane=plane,
                circle_alpha=1,
                skel_cmap=[current_color, (0, 0, 0, 0)],
                color_by='ntype',
                draw_soma_mask=False,
                xlim=(-rad, +rad),
                ylim=(-rad, +rad) if plane == 'xy' else zlim,
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
