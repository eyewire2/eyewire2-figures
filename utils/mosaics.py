from typing import Iterable

import pandas as pd
import numpy as np
import warnings

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import cell_mosaics


def plot_coverage(
        coverage_count: np.ndarray,
        cell_outlines: Iterable[np.ndarray] | None = None,
        extent: tuple[float, float, float, float] | None = None,
        ax: Axes | None = None,
        colormap: 'str' = 'viridis',
        figsize: 'tuple[int, int]' = (10, 8),
        alpha: 'float' = 1.0,
        interpolation: 'str' = 'nearest',
        edgecolor='k',
        edge_kws=None,
        plot_cbar=True,
        min_val=None,
        max_val=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # Set up discrete colormap for integer data
    min_val = int(np.min(coverage_count)) if min_val is None else min_val
    max_val = int(np.max(coverage_count)) if max_val is None else max_val
    n_levels = max_val - min_val + 1

    if max_val < int(np.max(coverage_count)):
        warnings.warn(f'max_val too small. Should be >={int(np.max(coverage_count))}')

    if min_val > int(np.min(coverage_count)):
        warnings.warn('max_val too small')

    # Get colors from the specified colormap
    base_cmap = cm.get_cmap(colormap)
    colors = base_cmap(np.linspace(0, 1, n_levels))
    discrete_cmap = ListedColormap(colors)

    im = ax.imshow(
        coverage_count,
        extent=extent,
        origin='lower',
        cmap=discrete_cmap,
        alpha=alpha,
        interpolation=interpolation,
        vmin=min_val - 0.5,
        vmax=max_val + 0.5
    )

    if plot_cbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage', rotation=270, labelpad=20)

        # Set discrete ticks on colorbar
        cbar.set_ticks(np.arange(min_val, max_val + 1))

    if cell_outlines is not None:
        for poly in cell_outlines:
            if extent is not None:
                xmin, xmax, ymin, ymax = extent
                poly_arr = np.asarray(poly)
                # Skip polygons entirely outside the extent
                if (poly_arr[:, 0].max() < xmin or poly_arr[:, 0].min() > xmax or
                        poly_arr[:, 1].max() < ymin or poly_arr[:, 1].min() > ymax):
                    continue
            cell_mosaics.plotting.plot_polygon(ax, poly, edgecolor=edgecolor, facealpha=0, edge_kws=edge_kws)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    return fig, ax, im


def plot_multiple_mosaics(
        df,
        celltype_col: str,
        celltypes: list[str],
        fig,
        coverage_axs=None,
        center_axs=None,
        title_axs=None,
        center_x_col="soma_x_um",
        center_y_col="soma_y_um",
        marker_sizes=None,
        max_val: float = 1,
        extent: tuple[int, int, int, int] = (250, 650, 500, 900),
        outline_hull=None,
        resolutions=None,
        plot_outlines: bool = True,
        tight: bool = False,
        cb: bool = False,
        sb_axs=None,
        sb_size: int = 200,
        sb_color: str = 'k',
        candidate_col: str | None = None,
        candidate_color: str = 'gray',
        candidate_marker_sizes=None,
        plot_candidate_coverage: bool = False,
        outline_kws: dict | None = None,
):
    """
    Plot coverage mosaics for multiple cell types.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `celltype_col`, `hull_points`, `soma_x_um`, `soma_y_um`.
    celltype_col : str
        Column name used to filter cell types.
    celltypes : list[str]
        Ordered list of cell types to plot.
    fig : matplotlib.figure.Figure
    coverage_axs : array-like of Axes or None
    center_axs : array-like of Axes or None
    title_axs : array-like of Axes or None
    marker_sizes : list[int] or None
        Per-cell-type marker sizes for the soma scatter plot; falls back to 2.
    max_val : float
        Minimum floor for the colour scale maximum.
    extent : (x_min, x_max, y_min, y_max)
    outline_hull : array-like or None
        Optional outline drawn on every panel.
    resolutions : list[int] or None
        Per-cell-type mapper resolution; falls back to 500.
    plot_outlines : bool
    tight : bool
    cb : bool
        Whether to draw per-panel colour bars.
    sb_axs : list of Axes or None
        Axes on which to draw the scale bar; defaults to first coverage ax.
    sb_size : int
        Scale bar length in µm.
    sb_color : str
    candidate_col : str or None
        Name of a boolean column in ``df``. Rows where the column is ``True``
        are treated as candidates and plotted in ``candidate_color``; rows
        where it is ``False`` / ``NaN`` are treated as normal cells.
        Because the two sets are mutually exclusive by construction, a cell is
        never shown as both a candidate and a normal cell.
    candidate_color : str
        Colour used for candidate soma markers. Default ``'gray'``.
    candidate_marker_sizes : list[int] or None
        Per-cell-type marker sizes for candidate cells; falls back to the
        corresponding value in ``marker_sizes``, then to 2.
    plot_candidate_coverage : bool
        If ``True``, candidate cells contribute their hull polygons to the
        coverage heatmap together with the normal cells (single merged
        coverage_count). Candidate cell outlines are drawn in
        ``candidate_color`` to distinguish them from normal cells, whose
        outlines keep their default colour. Has no effect if
        ``candidate_col`` is ``None``. Default ``False``.
    """
    if outline_kws is None:
        outline_kws = dict(c='gray', ls='--', clip_on=False)
    
    ims: list = []
    c_maxs: list[int] = []
    
    if coverage_axs is not None:
        coverage_axs = np.asarray(coverage_axs)
    if center_axs is not None:
        center_axs = np.asarray(center_axs)
    if title_axs is None:
        title_axs = coverage_axs
    else:
        title_axs = np.asarray(title_axs)

    # ------------------------------------------------------------------
    # Pre-compute mappers, re-using results for duplicate cell types
    # ------------------------------------------------------------------
    mapper_cache: dict[tuple, object] = {}  # (celltype, resolution, include_cand) -> mapper

    for i, celltype in enumerate(celltypes):
        # Split into normal cells and candidates
        is_ct = df[celltype_col] == celltype
        df_ct = df[is_ct]
        if candidate_col is not None:
            is_candidate = (df[candidate_col] == celltype) & ~is_ct
            df_cand = df[is_candidate]

            print(
                f'{celltype=}: Found {sum(is_ct)} cells and {sum(is_candidate)} candidates.'
            )
        else:
            df_cand = pd.DataFrame(columns=df.columns)

        hull_points = df_ct['hull_points']
        cand_hull_points = df_cand['hull_points'] if len(df_cand) > 0 else None
        merge_candidates = (
            plot_candidate_coverage
            and candidate_col is not None
            and cand_hull_points is not None
        )
        resolution = resolutions[i] if resolutions is not None else 500

        # ---- coverage (only when axes are provided) ------------------
        if coverage_axs is not None:
            ax = coverage_axs.flat[i]
            cache_key = (celltype, resolution, merge_candidates)

            if cache_key not in mapper_cache:
                mapper = cell_mosaics.CoverageDensityMapper(
                    field_bounds=extent,
                    resolution=resolution,
                )
                mapper.add_multiple_polygons(hull_points)
                if merge_candidates:
                    mapper.add_multiple_polygons(cand_hull_points)
                mapper_cache[cache_key] = mapper
            else:
                mapper = mapper_cache[cache_key]

            max_val_i = max(max_val, int(mapper.coverage_count.max()))

            # If we need to colour candidate outlines differently, suppress
            # the mapper's built-in outline drawing and overlay manually.
            draw_outlines_manually = (
                plot_outlines and merge_candidates and len(df_cand) > 0
            )
            cell_outlines_arg = (
                None if draw_outlines_manually
                else (mapper.cell_outlines if plot_outlines else None)
            )

            _, _, im = plot_coverage(
                ax=ax,
                coverage_count=mapper.coverage_count,
                cell_outlines=cell_outlines_arg,
                extent=extent,
                colormap='gist_heat_r',
                min_val=0,
                max_val=max_val_i,
                plot_cbar=False,
                edge_kws=dict(lw=0.2),
            )

            # Manual outline overlay: normal cells in default colour,
            # candidate cells in candidate_color.
            if draw_outlines_manually:
                for poly in hull_points:
                    poly_arr = np.asarray(poly)
                    closed_poly = np.vstack([poly_arr, poly_arr[0]])
                    ax.plot(*closed_poly.T, c='k', lw=0.2)
                for poly in cand_hull_points:
                    poly_arr = np.asarray(poly)
                    closed_poly = np.vstack([poly_arr, poly_arr[0]])
                    ax.plot(*closed_poly.T, c=candidate_color, lw=0.2)

            ims.append(im)
            c_maxs.append(max_val_i)
            ax.set(xticks=[], yticks=[], ylabel=None, xlabel=None)

        # ---- soma scatter (only when axes are provided) --------------
        if center_axs is not None:
            c_ax = center_axs.flat[i]

            # Filter cells within extent
            x_min, x_max, y_min, y_max = extent
            mask = ((df_ct[center_x_col] >= x_min) & (df_ct[center_x_col] <= x_max) &
                    (df_ct[center_y_col] >= y_min) & (df_ct[center_y_col] <= y_max))
            df_ct_filtered = df_ct[mask]

            c_ax.scatter(
                df_ct_filtered[center_x_col], df_ct_filtered[center_y_col],
                marker='o',
                c='k',
                linewidths=0,
                s=marker_sizes[i] if marker_sizes is not None else 2,
            )

            # ---- candidate soma scatter ------------------------------
            if df_cand is not None and len(df_cand) > 0:
                cand_mask = (
                    (df_cand[center_x_col] >= x_min) & (df_cand[center_x_col] <= x_max) &
                    (df_cand[center_y_col] >= y_min) & (df_cand[center_y_col] <= y_max)
                )
                cand_filtered = df_cand[cand_mask]
                if len(cand_filtered) > 0:
                    cand_size = (
                        candidate_marker_sizes[i] if candidate_marker_sizes is not None
                        else (marker_sizes[i] if marker_sizes is not None else 2)
                    )
                    c_ax.scatter(
                        cand_filtered[center_x_col], cand_filtered[center_y_col],
                        marker='o',
                        linewidths=0,
                        c=candidate_color,
                        s=cand_size,
                    )

            c_ax.set(xticks=[], yticks=[], ylabel=None, xlabel=None)
            c_ax.set_aspect('equal', 'box')

        if title_axs is not None:
            if candidate_col is not None and len(df_cand) > 0:
                title = f"{celltype}\n(n={len(df_ct)} [+{len(df_cand)}])"
            else:
                title = f"{celltype}\n(n={len(df_ct)})"
            title_axs.flat[i].set_title(title, fontsize=8, pad=2)

    # ------------------------------------------------------------------
    # Outline
    # ------------------------------------------------------------------
    if outline_hull is not None:
        closed = np.vstack([outline_hull, outline_hull[0]])

        if coverage_axs is not None:
            for i, c_ax in enumerate(coverage_axs.flat):
                if i < len(celltypes):
                    c_ax.plot(*closed.T, **outline_kws)
        if center_axs is not None:
            for i, c_ax in enumerate(center_axs.flat):
                if i < len(celltypes):
                    c_ax.plot(*closed.T, **outline_kws)

    # ------------------------------------------------------------------
    # Scale bar
    # ------------------------------------------------------------------
    if sb_axs is None:
        sb_axs = []
        if coverage_axs is not None:
            sb_axs = [coverage_axs.flat[0]]
        elif center_axs is not None:
            sb_axs = [center_axs.flat[0]]

    x0 = extent[0] + 0.6 * sb_size
    y0 = extent[2] - 0.1 * (extent[3] - extent[2])
    y0_text = y0 - 0.05 * (extent[3] - extent[2])

    for ax in sb_axs:
        ax.plot(
            [x0 - sb_size / 2, x0 + sb_size / 2], [y0, y0],
            c=sb_color, solid_capstyle='butt', lw=0.8, clip_on=False,
        )
        ax.text(
            x0 - sb_size / 2, y0_text,
            f'{sb_size} µm', c=sb_color, ha='left', va='top',
        )

    if tight:
        plt.tight_layout(w_pad=0.3, h_pad=0.5)

    # ------------------------------------------------------------------
    # Colour bars — skip panels whose c_max matches a previously seen value
    # ------------------------------------------------------------------
    if cb:
        seen_cmaxes: set[int] = set()
        for i, celltype in enumerate(celltypes):
            c_max = c_maxs[i]
            if c_max in seen_cmaxes:
                continue
            seen_cmaxes.add(c_max)

            ax = coverage_axs.flat[i]
            pos = ax.get_position()

            cax = fig.add_axes((
                pos.x0 + 0.2*pos.width,
                pos.y0 - 0.03,
                0.6*pos.width,
                0.018,
            ))

            cbar = fig.colorbar(ims[i], cax=cax, orientation='horizontal')
            cbar.set_label('Coverage', rotation=0, fontsize=6, labelpad=-1)
            tick_step = int(np.ceil(c_max / 5))
            cbar.set_ticks(np.arange(0, c_max + 1, tick_step))
            cbar.ax.tick_params(labelsize=6, length=2, pad=1)
            cbar.outline.set_visible(True)

    return ims, c_maxs


def polygon_centroid(hull_points):
    """
    Compute the centroid of a convex hull polygon.
    hull_points: numpy array of shape (N, 2), vertices in order (CW or CCW)
    """
    x = hull_points[:, 0]
    y = hull_points[:, 1]

    # Shoelace terms
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y

    area = 0.5 * np.abs(np.sum(cross))

    cx = np.sum((x + np.roll(x, -1)) * cross) / (6 * area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6 * area)

    return np.array([cx, cy])
