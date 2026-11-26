"""Mouse-cam receptive fields (RFs) on EM skeletons.

Small copies of eyewire2-functional-analysis code (``plot_rf``, ``registration``, ``rf_alignment`` and
``receptive_fields.movie_to_retina``), so that this repo does not depend on it. Load the inputs with
``data_2p_loader.load_df_mc_rfs``, ``load_registration`` and ``load_rf_alignment``.

All positions are in um in the retinal frame (x = nasal, y = dorsal), relative to the soma, which sits at the ROI
position. The mouse-cam movie is centered on the recording field. With an ``alignment``, the RF positions are corrected
by the per-field stimulus shift measured with the moving bar, and skeletons get the per-field EM -> 2p stretch (see
``scripts/analysis/rf_offsets/REPORT.md`` in eyewire2-functional-analysis).

sRF and tRF are flipped so that the fitted DoG center is positive, i.e. the tRF carries the center polarity: a tRF that
is positive just before the response means an ON center.
"""
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.collections import LineCollection
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from skeleton import transform_skel_xy

CHANNELS = ('green', 'uv')
CH_COLORS = {'green': '#2ca02c', 'uv': '#8e44ad'}
CH_LABELS = {'green': 'green', 'uv': 'UV'}
SRF_CMAP = 'RdBu_r'


# Alignment ------------------------------------------------------------------------------------------------------------

def _field_alignment(alignment, field):
    if alignment is None:
        return None
    if field not in alignment['fields']:
        warnings.warn(f'No RF alignment for field {field!r}; no correction applied.')
        return None
    return alignment['fields'][field]


def stim_shift(alignment, field) -> np.ndarray:
    """Stimulus shift ``(dx, dy)`` in um of ``field``, subtracted from RF positions; zeros without alignment."""
    f = _field_alignment(alignment, field)
    return np.zeros(2) if f is None else np.asarray(f['stim_shift_um'], float)


def em_stretch(alignment, field) -> np.ndarray:
    """2x2 EM -> 2p stretch of ``field`` (applied after the registration rotation); identity without alignment."""
    f = _field_alignment(alignment, field)
    return np.eye(2) if f is None else np.asarray(f['em_affine'], float)


def _similarity_matrix(angle_deg, flip_x=False, flip_y=False):
    """Rotation by ``angle_deg``, then mirror flip (column-vector convention)."""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.diag([-1. if flip_x else 1., -1. if flip_y else 1.]) @ np.array([[c, -s], [s, c]])


def field_rotation(reg, field, direction='em_to_2p') -> np.ndarray:
    """Rotation + flip (2x2) of the registration for ``field``: the global fit, then the per-field refinement."""
    d = reg['directions'][direction]
    M = _similarity_matrix(d['global']['angle_deg'], reg['meta']['flip_x'], reg['meta']['flip_y'])
    field_fit = d['fields'].get(field)
    if field_fit is None:
        warnings.warn(f'No per-field registration for {field!r}; using the global fit.')
        return M
    return _similarity_matrix(field_fit['angle_deg']) @ M


def place_skel(skel, reg, field, alignment=None):
    """EM skeleton rotated into the 2p frame by the registration of ``field``, stretched by the EM -> 2p stretch of
    the field (if ``alignment`` is given), with its soma at the origin. Returns a copy."""
    skel = transform_skel_xy(skel, em_stretch(alignment, field) @ field_rotation(reg, field))
    c = np.array(skel.soma.center[:2], dtype=float)
    skel.nodes[:, :2] -= c
    skel.soma.center[:2] -= c
    return skel


# RF geometry ----------------------------------------------------------------------------------------------------------

def movie_to_retina(img: np.ndarray) -> np.ndarray:
    """Movie-space image ``(rows, cols)`` in the retinal frame, to be shown with ``origin='lower'``: array row ->
    temporal, array column -> ventral, i.e. the transpose with both axes reversed."""
    return img.T[::-1, ::-1]


def srf_retina(row, ch, alignment=None):
    """sRF of channel ``ch`` in the retinal frame and its extent ``(left, right, bottom, top)`` in um relative to the
    soma, for ``imshow(img, extent=extent, origin='lower')``."""
    img = movie_to_retina(row[f'mc_rf_{ch}_srf'])
    px = row.mc_rf_pixel_size_um
    dx, dy = stim_shift(alignment, row.field)
    cx = row.field_temporal_nasal_pos_um - row.temporal_nasal_pos_um - dx  # stimulus center relative to the soma
    cy = row.field_ventral_dorsal_pos_um - row.ventral_dorsal_pos_um - dy
    ny, nx = img.shape
    return img, (cx - nx * px / 2, cx + nx * px / 2, cy - ny * px / 2, cy + ny * px / 2)


def _movie_px_to_retina_um(row, x, y, shape, alignment=None):
    """Movie pixel coordinates (``x`` = column, ``y`` = row) to the retinal frame, consistent with ``srf_retina``."""
    ny, nx = shape
    img_r = (nx - 1) - np.asarray(x)
    img_c = (ny - 1) - np.asarray(y)
    _, (left, _, bottom, _) = srf_retina(row, CHANNELS[0], alignment)
    px = row.mc_rf_pixel_size_um
    return left + (img_c + 0.5) * px, bottom + (img_r + 0.5) * px


def dog_ellipse(row, ch, which='center', n_sigma=2., n=200, alignment=None):
    """``n_sigma`` ellipse ``(xs, ys)`` of the DoG center or surround of channel ``ch``; None if not fitted."""
    p = f'mc_rf_{ch}_dog_'
    sx, sy = (row[p + 'sx'], row[p + 'sy']) if which == 'center' else (row[p + 'sx_s'], row[p + 'sy_s'])
    x0, y0, theta = row[p + 'x0'], row[p + 'y0'], row[p + 'theta']
    if not np.all(np.isfinite([x0, y0, sx, sy, theta])):
        return None
    t = np.linspace(0, 2 * np.pi, n)
    u, v = n_sigma * sx * np.cos(t), n_sigma * sy * np.sin(t)
    c, s = np.cos(theta), np.sin(theta)
    x, y = x0 + c * u + s * v, y0 - s * u + c * v
    return _movie_px_to_retina_um(row, x, y, row[f'mc_rf_{ch}_srf'].shape, alignment)


def surround_volume(row, ch) -> float:
    """Volume of the DoG surround relative to the center, ``w k^2``."""
    return row[f'mc_rf_{ch}_dog_surround_rel_amp'] * row[f'mc_rf_{ch}_dog_surround_ratio'] ** 2


def rf_radius(row, chs=CHANNELS, margin=20., alignment=None) -> float:
    """Half-width of a soma-centered square that contains the 2-sigma DoG center ellipses of ``chs``, plus ``margin``."""
    rads = [np.abs(np.concatenate(xy)).max() for ch in chs if (xy := dog_ellipse(row, ch, alignment=alignment))
            is not None]
    return float(max(rads)) + margin if rads else 0.


# Resolution floor -----------------------------------------------------------------------------------------------------

def bspline_basis(n, df, degree=3):
    """B-spline basis of shape ``(n, df)`` with uniformly spaced knots on 0, ..., n - 1."""
    inner = np.linspace(0, n - 1, df - degree + 1)
    knots = np.r_[[inner[0]] * degree, inner, [inner[-1]] * degree]
    return BSpline.design_matrix(np.arange(n, dtype=float), knots, degree).toarray()


def _unit_gauss(xx, yy, x0, y0, sx, sy, theta):
    a = np.cos(theta) ** 2 / (2 * sx ** 2) + np.sin(theta) ** 2 / (2 * sy ** 2)
    b = -np.sin(2 * theta) / (4 * sx ** 2) + np.sin(2 * theta) / (4 * sy ** 2)
    c = np.sin(theta) ** 2 / (2 * sx ** 2) + np.cos(theta) ** 2 / (2 * sy ** 2)
    return np.exp(-(a * (xx - x0) ** 2 + 2 * b * (xx - x0) * (yy - y0) + c * (yy - y0) ** 2))


def fit_dog_sigmas(srfs, center_prior_sd=4., center_prior_weight=0.05, coupling_weight=0.05,
                   surround_ratio=(2., 10.), sigma_init=(1.5, 4.)):
    """Joint DoG fit to the sRFs ``(n_channels, ny, nx)``, with soft priors on the centers (window center and coupling
    of the channels); returns the center sigmas ``(sx, sy)`` per channel, in pixels."""
    n_ch, ny, nx = srfs.shape
    yy, xx = np.mgrid[:ny, :nx].astype(float)
    xc, yc = (nx - 1) / 2, (ny - 1) / 2
    z = srfs / np.sqrt(np.sum(srfs ** 2))
    s_max = float(max(nx, ny))
    # per channel: x0, y0, theta, sx, sy, amp, k (surround / center sigma), w (relative surround amplitude)
    lb = np.tile([-0.5, -0.5, -np.inf, 0.3, 0.3, -np.inf, surround_ratio[0], 0.], n_ch)
    ub = np.tile([nx - 0.5, ny - 0.5, np.inf, s_max, s_max, np.inf, surround_ratio[1], 1.], n_ch)

    def model(p):
        g = _unit_gauss(xx, yy, p[0], p[1], p[3], p[4], p[2])
        g = g - p[7] * _unit_gauss(xx, yy, p[0], p[1], p[6] * p[3], p[6] * p[4], p[2])
        return p[5] * g

    def residuals(theta):
        ps = theta.reshape(n_ch, 8)
        res = [(model(p) - z[c]).ravel() for c, p in enumerate(ps)]
        res.append(np.sqrt(center_prior_weight) * (ps[:, :2] - [xc, yc]).ravel() / center_prior_sd)
        for i, j in itertools.combinations(range(n_ch), 2):
            res.append(np.sqrt(coupling_weight) * (ps[i, :2] - ps[j, :2])
                       / np.sqrt(ps[i, 3] * ps[i, 4] + ps[j, 3] * ps[j, 4]))
        return np.concatenate(res)

    # start from the window center and from the largest |value|, each with every sigma in sigma_init
    ch0 = int(np.argmax(np.abs(z).max(axis=(1, 2))))
    iy, ix = np.unravel_index(np.argmax(np.abs(z[ch0])), (ny, nx))
    best = None
    for (x0, y0), s0 in itertools.product({(xc, yc), (float(ix), float(iy))}, sigma_init):
        g0 = _unit_gauss(xx, yy, x0, y0, s0, s0, 0.)
        p0 = np.concatenate([[x0, y0, 0., s0, s0, (np.sign(np.sum(g0 * z[c])) or 1.) * max(np.abs(z[c]).max(), 1e-3),
                              3., 0.2] for c in range(n_ch)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sol = least_squares(residuals, np.clip(p0, lb + 1e-6, ub - 1e-6), bounds=(lb, ub), x_scale='jac',
                                    max_nfev=2000)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(sol.cost) and (best is None or sol.cost < best.cost):
            best = sol
    return best.x.reshape(n_ch, 8)[:, 3:5]


def resolution_floor_area(n_px, px_um, df_s=14, n_sigma=2.):
    """Area (um^2) of the ``n_sigma`` DoG center ellipse of the smallest sRF the RF model can express: the central
    one of ``df_s`` x ``df_s`` tensor-product B-spline basis functions on the ``n_px`` x ``n_px`` stimulus grid."""
    basis = bspline_basis(n_px, df_s)
    srf_min = np.outer(basis[:, df_s // 2], basis[:, df_s // 2])
    sx, sy = fit_dog_sigmas(np.stack([srf_min, srf_min]))[0]
    return float(np.pi * n_sigma ** 2 * sx * sy * px_um ** 2)


# Plotting -------------------------------------------------------------------------------------------------------------

def plot_skel_lines(ax, skel, color='k', lw=0.5, alpha=0.9, rasterized=True):
    """Skeleton edges as a flat XY line collection plus a soma disk. ``rasterized``: draw the edges as an image in
    vector output (an SVG gets one path element per edge otherwise, i.e. tens of MB for a few cells)."""
    xy = skel.nodes[:, :2]
    ax.add_collection(LineCollection(np.stack([xy[skel.edges[:, 0]], xy[skel.edges[:, 1]]], axis=1),
                                     colors=color, linewidths=lw, alpha=alpha, rasterized=rasterized))
    soma_r = getattr(skel.soma, 'equiv_radius', 0) or 0
    ax.add_patch(patches.Circle(skel.soma.center[:2], max(soma_r, 1e-6), facecolor=color, edgecolor='none',
                                alpha=alpha * 0.6))


def draw_dog(ax, row, ch, color=None, lw=1., min_surround_vol=0.1, alignment=None, zorder=4, halo=1., n_sigma=2.):
    """``n_sigma`` ellipses of the DoG center (solid) and, if its relative volume is at least ``min_surround_vol``,
    the surround (dotted) of channel ``ch``, with a white halo ``halo`` points wider than the line."""
    color = color or CH_COLORS[ch]
    for which, ls in (('center', '-'), ('surround', ':')):
        if which == 'surround' and not surround_volume(row, ch) >= min_surround_vol:
            continue
        xy = dog_ellipse(row, ch, which, n_sigma=n_sigma, alignment=alignment)
        if xy is None:
            continue
        ax.plot(*xy, c='w', lw=lw + halo, ls='-', alpha=0.8, zorder=zorder, solid_capstyle='round')
        ax.plot(*xy, c=color, lw=lw, ls=ls, zorder=zorder + 0.1)


def plot_srf_on_morph(ax, row, skel, ch, vabsmax=None, alignment=None, show_fit=True, skel_lw=0.5,
                      min_surround_vol=0.1, interpolation='bicubic'):
    """sRF of channel ``ch`` (red = positive, blue = negative) with the skeleton (soma at the origin, see
    ``place_skel``), the stimulus border and the DoG ellipses on top. Returns the image (e.g. for a colorbar)."""
    img, extent = srf_retina(row, ch, alignment)
    if vabsmax is None:
        vabsmax = np.nanmax(np.abs(img))
    im = ax.imshow(img, cmap=SRF_CMAP, vmin=-vabsmax, vmax=vabsmax, origin='lower', extent=extent,
                   interpolation=interpolation, zorder=0)
    ax.add_patch(plt.Rectangle(extent[::2], extent[1] - extent[0], extent[3] - extent[2], fill=False, ec='0.6',
                               lw=0.5, zorder=1))
    plot_skel_lines(ax, skel, lw=skel_lw)
    if show_fit:
        draw_dog(ax, row, ch, min_surround_vol=min_surround_vol, alignment=alignment)
    ax.set(aspect='equal', xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def plot_orientation(ax, x0, y0, size=40., fontsize=6, lw=0.6, head_length=0.25, head_width=0.12, d_offset=1.15):
    """Two arrows from ``(x0, y0)`` (data units) pointing nasal (right) and dorsal (up), labeled N and D; the D label
    sits ``d_offset`` arrow lengths above the origin."""
    kw = dict(arrowstyle=f'-|>,head_length={head_length},head_width={head_width}', lw=lw, color='k', shrinkA=0,
              shrinkB=0)
    ax.annotate('', xy=(x0 + size, y0), xytext=(x0, y0), arrowprops=kw, annotation_clip=False)
    ax.annotate('', xy=(x0, y0 + size), xytext=(x0, y0), arrowprops=kw, annotation_clip=False)
    ax.text(x0 + size * 1.15, y0, 'N', ha='left', va='center', fontsize=fontsize)
    ax.text(x0, y0 + size * d_offset, 'D', ha='center', va='bottom', fontsize=fontsize)


def plot_trfs(ax, row, chs=CHANNELS, lw=1., labels=True):
    """tRFs of ``chs`` (they carry the polarity of the DoG center), scaled by the sRF peak so that the channels are
    comparable."""
    t = row.mc_rf_trf_t0 + np.arange(row[f'mc_rf_{chs[0]}_trf'].size) * row.mc_rf_trf_dt
    ax.axhline(0, c='0.8', lw=0.5, zorder=0)
    ax.axvline(0, c='0.8', lw=0.5, ls='--', zorder=0)
    for ch in chs:
        ax.plot(t, row[f'mc_rf_{ch}_trf'] * np.nanmax(np.abs(row[f'mc_rf_{ch}_srf'])), c=CH_COLORS[ch], lw=lw,
                label=CH_LABELS[ch])
    ax.set(xlim=(t[0], t[-1]), yticks=[])
    if labels:
        ax.set_xlabel('time to response (s)')
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
