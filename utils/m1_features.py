"""Per-cell morphology features computed directly from a skeleton (SWC).

Minimal copies of the eyewire2-datajoint feature definitions, for cells that are not (yet) in the
figure dataframe:

- convex hull of the dendrites: ``datajoint/tables/convex_hull.py`` + ``utils/skel.py``
  (``compute_convex_hull_features``)
- waviness ``excess_mean``: ``datajoint/tables/dendrite_waviness.py`` (only the parts needed for
  ``excess_mean``, same default parameters)
"""
from collections import deque

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull

from mosaics import polygon_centroid

DENDRITE_NTYPES = (3,)


# ---------------------------------------------------------------- convex hull (convex_hull.py)
def convex_hull_features(points):
    hull = ConvexHull(points)
    area = hull.volume  # in 2D, 'volume' is the area
    return dict(hull_points=points[hull.vertices, :2], hull_area=area,
                hull_diameter=np.sqrt(4 * area / np.pi))


# ---------------------------------------------------------------- waviness (dendrite_waviness.py)
def _long_paths(nodes, edges, ntype, root=0, dendrite_ntypes=DENDRITE_NTYPES):
    """Long paths through the tree (continue into the child with most downstream cable)."""
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    parent = np.full(n, -1)
    order = []
    seen = np.zeros(n, bool)
    seen[root] = True
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                q.append(v)

    cable = np.zeros(n)
    for v in reversed(order):
        p = parent[v]
        if p >= 0:
            cable[p] += cable[v] + np.linalg.norm(nodes[v] - nodes[p])
    children = [[] for _ in range(n)]
    for v in order[1:]:
        children[parent[v]].append(v)
    heavy = np.full(n, -1)
    for u in range(n):
        if children[u]:
            heavy[u] = max(children[u], key=lambda c: cable[c])

    is_dendrite = np.isin(ntype, dendrite_ntypes)
    paths = []
    starts = list(children[root])
    while starts:
        s = starts.pop()
        p = [] if parent[s] == root else [parent[s]]
        v = s
        while v >= 0:
            p.append(v)
            starts.extend(c for c in children[v] if c != heavy[v])
            v = heavy[v]
        p = np.array(p)
        idx = np.flatnonzero(np.diff(np.r_[0, is_dendrite[p].astype(int), 0]))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a >= 3:
                paths.append(p[a:b])
    return paths


def _arclen(P):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def _resample(P, step):
    s = _arclen(P)
    if s[-1] < 2 * step:
        return None
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])])


def _smooth_curve_reparam(P, sigma, step, n_iter=2):
    t = _arclen(P)
    if t[-1] < 4 * step:
        return None
    for _ in range(n_iter):
        u = np.arange(0, t[-1] + 1e-9, step)
        R = np.column_stack([np.interp(u, t, P[:, k]) for k in range(P.shape[1])])
        S = gaussian_filter1d(R, sigma / step, axis=0, mode='nearest')
        t = np.interp(t, u, _arclen(S))
    return _resample(S, step)


def _window_excess(C, window, step, trim):
    a = int(round(trim / step))
    C = C[a:len(C) - a]
    n = int(round(window / step))
    if len(C) <= n:
        return np.array([])
    j = np.arange(0, len(C) - n, max(1, n // 2))
    chord = np.linalg.norm(C[j + n] - C[j], axis=1)
    return window / np.maximum(chord, 1e-6) - 1


def waviness_excess_mean(nodes, edges, ntype, step_um=0.25, smooth_sigma_um=1.0, trim_um=2.0,
                         min_path_len_um=12.0, excess_window_um=10.0):
    """Mean extra path length (arc / chord - 1) of the smoothed dendrite paths in 10 um windows."""
    excess = []
    for path in _long_paths(nodes, edges, ntype):
        P = nodes[path]
        if _arclen(P)[-1] < min_path_len_um:
            continue
        C = _smooth_curve_reparam(P, smooth_sigma_um, step_um)
        if C is not None:
            excess.append(_window_excess(C, excess_window_um, step_um, trim_um))
    excess = np.concatenate(excess) if excess else np.array([])
    return float(np.mean(excess)) if len(excess) else np.nan


# ---------------------------------------------------------------- per cell
def dendrite_length(nodes, edges, ntype):
    """Cable length (um) of edges between two dendrite nodes."""
    is_d = np.isin(ntype, DENDRITE_NTYPES)
    e = edges[is_d[edges[:, 0]] & is_d[edges[:, 1]]]
    return float(np.linalg.norm(nodes[e[:, 0]] - nodes[e[:, 1]], axis=1).sum())


def cell_features(skel):
    """Soma, dendritic hull, cable density and waviness of one skeleton (in um, soma = node 0)."""
    nodes, edges, ntype = skel.nodes, np.asarray(skel.edges), skel.ntype
    hull = convex_hull_features(nodes[np.isin(ntype, DENDRITE_NTYPES), :2])
    length = dendrite_length(nodes, edges, ntype)
    poly = hull['hull_points']
    return dict(
        soma_x_um=nodes[0, 0], soma_y_um=nodes[0, 1], soma_z_um=nodes[0, 2],
        soma_rad_um=float(np.mean(skel.soma.axes)),
        **dict(zip(('hull_center_x', 'hull_center_y'), polygon_centroid(poly))),
        **hull,
        dendrite_length=length,
        dendrite_density=length / hull['hull_area'] * 1000,  # um per 1000 um^2 of hull
        waviness=waviness_excess_mean(nodes, edges, ntype),
    )
