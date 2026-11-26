"""Text measuring and placement for figures laid out in inches: text widths, label fallbacks, collision-free point
labels and legend column counts."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.textpath import TextPath

PT = 1 / 72  # inches per point


def text_width_in(text, fontsize, prop=None):
    """Width (inches) of the longest line of ``text``, measured with ``prop`` (default: the rcParams font)."""
    prop = prop or FontProperties()
    return max(TextPath((0, 0), line or " ", size=fontsize, prop=prop).get_extents().width
               for line in text.split("\n")) * PT


def fit_text(text, fallback, width_in, fontsize):
    """``text`` if it is at most ``width_in`` inches wide, else ``fallback`` (e.g. the same text on two lines)."""
    return text if text_width_in(text, fontsize) <= width_in else fallback


def n_lines(text):
    return text.count("\n") + 1


def label_points(ax, labels, fontsize, size, color="0.3", leader_color="0.55"):
    """Put each label ``(text, x, ys)`` next to its points (above, below, left, right or diagonal), at increasing
    distance, with a thin leader line when it is not right next to them; the first position that overlaps no point
    (scatter marker size ``size`` in pt^2), no other label or text and not the legend, and stays inside the axes (else
    the position with the fewest overlaps)."""
    renderer = ax.figure.canvas.get_renderer()
    ax.apply_aspect()
    pts_px = ax.transData.transform(np.vstack([c.get_offsets() for c in ax.collections
                                               if hasattr(c, "get_sizes") and len(c.get_offsets())]))
    px_per_pt = ax.figure.dpi / 72
    r_pt = np.sqrt(size) / 2                      # marker radius (pt)
    r_px = (r_pt + 0.8) * px_per_pt
    ax_box, leg = ax.get_window_extent(renderer), ax.get_legend()
    taken = [leg.get_window_extent(renderer)] if leg is not None else []
    taken += [t.get_window_extent(renderer).padded(1.0 * px_per_pt) for t in ax.texts]  # e.g. '1:1'
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    for text, x, ys in labels:
        anchors = {1: (x, max(ys)), -1: (x, min(ys)), 0: (x, (max(ys) + min(ys)) / 2)}
        best = None
        for dist in (3, 8, 13):                   # pt from the points
            for dx, dy in directions:
                leader = dist > 3
                t = ax.annotate(
                    text, anchors[dy], xytext=(dx * dist * (1.3 if dy == 0 else 0.8), dy * dist),
                    textcoords="offset points", ha={-1: "right", 0: "center", 1: "left"}[dx],
                    va={-1: "top", 0: "center", 1: "bottom"}[dy], fontsize=fontsize, color=color,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=leader_color, shrinkA=0.5, shrinkB=r_pt + 0.5)
                    if leader else None)
                t.update_positions(renderer)  # place the text at its offset before measuring it
                bb = Text.get_window_extent(t, renderer).padded(1.0 * px_per_pt)  # the text only, without the leader
                n_bad = (sum(bb.x0 - r_px < px < bb.x1 + r_px and bb.y0 - r_px < py < bb.y1 + r_px
                             for px, py in pts_px)
                         + sum(bb.overlaps(o) for o in taken)
                         + 2 * (not (ax_box.contains(bb.x0, bb.y0) and ax_box.contains(bb.x1, bb.y1))))
                if best is None or n_bad < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (n_bad, t, bb)
                else:
                    t.remove()
                if n_bad == 0:
                    break
            if best[0] == 0:
                break
        taken.append(best[2])


def legend_ncol(handles, max_width, **legend_kw):
    """Largest number of columns (fewest rows) for which the legend is at most ``max_width`` inches wide."""
    probe = plt.figure()
    try:
        r = probe.canvas.get_renderer()
        for ncol in range(len(handles), 0, -1):
            leg = probe.legend(handles=handles, ncol=ncol, **legend_kw)
            width = leg.get_window_extent(r).width / probe.dpi
            leg.remove()
            if width <= max_width:
                return ncol
        return 1
    finally:
        plt.close(probe)
