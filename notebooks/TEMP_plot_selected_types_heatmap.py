"""BC type x individual RGC heatmap comparing two RGC types -- the Bernstein poster's
selected-types panel (plot_selected_types_all_cells), with type labels added and the type colors
shared with plot_example_rgc_inputs.py (via u24_common).

Cells are grouped into contiguous per-type blocks (in pair order), ordered by BC-type-composition
similarity within each block; each cell's BC-type counts are normalized to sum 1 (after the
ribbon filter, applied per RGC as in plot_example_rgc_inputs.py). --normalize mean instead divides
raw counts by the mean per-cell total, so cut-off cells (e.g. at the volume border) show up faint
instead of with inflated fractions (files get a _meannorm suffix).

By default makes every pair in u24_common.EXAMPLE_PAIRS x every filter version
(u24_common.FILTER_VERSIONS), saved under --output-dir as <A>_vs_<B>_heatmap_<filter>.{pdf,svg,png}.
--anonymize replaces all cell-type names by "RGC type A/B" and "BC A, B, ..." (u24_common; files
get an _anonymous suffix). Passing --pair and/or a filter threshold makes just that one pair / filter version.

Usage:
    uv run python scripts/U24/plot_selected_types_heatmap.py
    uv run python scripts/U24/plot_selected_types_heatmap.py --pair PixON M2 --min-bcs-per-type 5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Bernstein_Poster"))
from plot_poster_selected_types import plot_selected_types_all_cells

from em_synapses.connectivity.ribbon_input_common import resolve_bc_order_and_palette
from em_synapses.utils.paths import PLOTS_DIR
from u24_common import (CSV_DEFAULT, EXAMPLE_PAIRS, FILTER_VERSIONS, TYPE_COLORS, anon_bc_label,
                        anon_key, anon_rgc_labels, filter_ribbons, filter_tag, pair_stem, type_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=CSV_DEFAULT)
    parser.add_argument("--pair", nargs=2, default=None,
                        help="Two types to compare. Defaults to all pairs in u24_common.EXAMPLE_PAIRS.")
    parser.add_argument("--output-dir", default=PLOTS_DIR / "U24")
    parser.add_argument("--min-ribbons-per-bc", type=int, default=None)
    parser.add_argument("--min-bcs-per-type", type=int, default=None,
                        help="If neither threshold is given, all versions in u24_common.FILTER_VERSIONS "
                             "are made.")
    parser.add_argument("--anonymize", action="store_true",
                        help="Replace cell-type names by RGC type A/B and BC A, B, ...")
    parser.add_argument("--normalize", choices=["cell", "mean"], default="cell",
                        help="cell: each cell's counts sum to 1; mean: counts / mean per-cell total.")
    parser.add_argument("--cmap", default="Greys")
    parser.add_argument("--truncate-types", nargs="+", default=[],
                        help="Types to show only --truncate-n evenly sampled cells of, with a '...' gap.")
    parser.add_argument("--truncate-n", type=int, default=20)
    parser.add_argument("--width-in", type=float, default=10)
    parser.add_argument("--height-in", type=float, default=5.5)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=["cell", "cell_type", "partner_id", "celltype"],
                     dtype={"cell": str, "partner_id": str})
    bc_order, _ = resolve_bc_order_and_palette(df["cell_type"].dropna().unique())
    pairs = [tuple(args.pair)] if args.pair else list(EXAMPLE_PAIRS)
    if args.min_ribbons_per_bc is None and args.min_bcs_per_type is None:
        filters = list(FILTER_VERSIONS.values())
    else:
        filters = [(args.min_ribbons_per_bc or 1, args.min_bcs_per_type or 1)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        missing = [t for t in pair if t not in set(df["celltype"].dropna())]
        if missing:
            raise ValueError(f"Selected type(s) not found: {missing}")
        missing_colors = [t for t in pair if t not in TYPE_COLORS]
        if missing_colors:
            raise ValueError(f"No U24 color for {missing_colors}; add it to u24_common.TYPE_COLORS.")

        df_pair = df[df["celltype"].isin(pair)]
        for min_rib, min_bc in filters:
            name = f"{pair_stem(*pair)}_heatmap_{filter_tag(min_rib, min_bc)}"
            name += "_meannorm" if args.normalize == "mean" else ""
            name += "_anonymous" if args.anonymize else ""
            display_labels = anon_rgc_labels(pair) if args.anonymize else {t: type_label(t) for t in pair}
            print(name)
            fig = plot_selected_types_all_cells(
                filter_ribbons(df_pair, min_rib, min_bc), bc_order, list(pair),
                display_labels, {t: TYPE_COLORS[t] for t in pair},
                label="RGC", cmap=args.cmap, figsize=(args.width_in, args.height_in),
                truncate_types=tuple(args.truncate_types), truncate_n=args.truncate_n, label_blocks=True,
                normalize=args.normalize)
            if args.anonymize:
                ax_heatmap = fig.axes[1]  # [strip, heatmap, colorbar]
                ax_heatmap.set_yticklabels([anon_bc_label(tl.get_text()) for tl in ax_heatmap.get_yticklabels()])
            for ext in ("pdf", "svg", "png"):
                fig.savefig(output_dir / f"{name}.{ext}", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
    if args.anonymize:
        (output_dir / "anonymous_label_key.txt").write_text(anon_key())


if __name__ == "__main__":
    main()
