"""Resource-paper RGC type labels, applied on top of the dataset's labels.

The parquet (``df_all_neurons_*``) and the 09-16 main sheet carry the RGC type names of their export. The RGC types
were renamed and refined later; the resource-paper subset lives in its own sheet (the "RGC types sheet", config keys
``file_prefix_rgc_types_sheet`` + ``version_rgc_types_sheet`` in ``spreadsheet_dir``). It has one row per cell of the
31 resource-paper RGC types (Index, coordinates, IDs, Cell Class and Cell Type), plus a few override rows: cells of
another class whose label in the newer main sheet is wrong (e.g. H3160, an AC labelled as an RGC type), with their
class and an empty type. It is written by
``eyewire2-morphological-clustering/notebooks/celltype_explorer/rgc_final/sheet_resource_paper.py``.

Rules (the same for the parquet and for the main sheet):

* A cell in the RGC types sheet gets its Cell Type and Cell Class from it. In the parquet these are both the final
  labels (``celltype_final``/``cellclass_final``, decision 'refined', valid) and the human labels
  (``celltype``/``cellclass``, valid). An empty type in the sheet gives a missing (not valid) type.
* Any other cell loses its type label (set to missing) if its class is RGC, or if its label is a resource-paper type
  name or an old name of a renamed type (``RENAMED``). Those labels are stale: the cell is no longer in that type, or
  it is in a type that is not in the resource paper. Its class is kept.
* All other labels (BCs, ACs including ON/OFF SAC, ...) are untouched.

Three columns are added, all 'yes'/'no': ``Label from RGC types sheet``, ``Updated Cell Type``, ``Updated Cell Class``
(the label differs from the one in the parquet / main sheet). In the parquet the original values are kept in
``<column>_parquet`` columns, and ``celltype_final_parquet_renamed`` has the parquet's type under the new names (to
tell a renamed type from a cell that moved to another type, e.g. for ``same_type_both_strong``).

Cells are matched on ``Index`` (the main-sheet row id). In the parquet, cells without an Index are matched on their
segment ID (the parquet index) against Latest SegID / Proofread SegID of the RGC types sheet, if that is unique.
"""
import os

import numpy as np
import pandas as pd

FROM_SHEET, UPDATED_TYPE, UPDATED_CLASS = 'Label from RGC types sheet', 'Updated Cell Type', 'Updated Cell Class'
INDICATOR_COLS = [FROM_SHEET, UPDATED_TYPE, UPDATED_CLASS]

# Type labels of the dataset export (parquet 2026-09-16-15h, sheet 2026-09-16) that were renamed: old -> new. Copied
# from RENAME in eyewire2-morphological-clustering/notebooks/celltype_explorer/rgc_final/sheet_rename_types.py
# (2026-11-25). Note the swap: 'Bursty suppressed by contrast' -> 'OFF sustained alpha' -> 'EW2o'.
RENAMED = {
    'ON delayed': 'EW73',
    'OFF medium sustained': 'EW2i',
    'OFF transient medium RF': 'EW4on',
    'UHD': 'EW5ti',
    'Bursty suppressed by contrast': 'OFF sustained alpha',
    'ON-OFF DS - direction unknown': 'ON-OFF DS - other directions',
    'ON-OFF DS - ventral': 'ON-OFF DS - ventral dendrites',
    'ON vertical OS small RF': 'EW72',
    'ON DS sustained - direction unknown': 'ON DS EW7i',
    'OFF transient small RF': 'EW4i',
    'HD1': 'EW5si',
    'ON DS transient': 'ON DS EW7o',
    'OFF sustained alpha': 'EW2o',
    'ON transient medium RF': 'ON transient alpha',
    'ON transient small RF': 'EW6sn',
    'ON bursty': 'EW3i',
    'ON alpha': 'ON sustained alpha',
    'ON transient EW6t': 'EW6t',
    'OFF sustained EW3o': 'EW3o',
    'PixON': 'M5',
    'OFF sustained EW1no': 'EW1no',
    'ON small OFF large': 'EW1ni',
    'EW81i': 'EW8n-like',
    'Sustained suppressed-by-contrast strong surround EW28': 'EW28',
    'EW28-like': 'EW27-like-A',
    'Sustained suppressed-by-contrast no surround EW27': 'EW27-like-B',
    'EW27-like': 'EW27-like-C',
}
OLD_RGC_TYPE_NAMES = list(RENAMED)

PARQUET_LABEL_COLS = ['Index', 'celltype_final', 'cellclass_final', 'celltype_final_decision',
                      'cellclass_final_decision', 'valid_celltype_final', 'valid_cellclass_final',
                      'celltype', 'cellclass', 'valid_celltype', 'valid_cellclass']
REFINED = 'refined'   # celltype_final_decision / cellclass_final_decision of labels from the RGC types sheet


def rgc_types_sheet_path(config):
    return os.path.abspath(os.path.join(
        config['spreadsheet_dir'], f"{config['file_prefix_rgc_types_sheet']}{config['version_rgc_types_sheet']}.csv"))


def load_rgc_types_sheet(config):
    """The RGC types sheet (all columns as str), checked: unique Index, a type for every RGC, no type for other cells."""
    path = rgc_types_sheet_path(config)
    assert os.path.isfile(path), f"RGC types sheet not found: {path}"
    sheet = pd.read_csv(path, dtype=str)
    assert sheet['Index'].notna().all() and sheet['Index'].is_unique, 'RGC types sheet: Index missing or not unique'
    assert sheet['Cell Class'].notna().all(), 'RGC types sheet: rows without Cell Class'
    is_rgc = sheet['Cell Class'] == 'RGC'
    assert sheet.loc[is_rgc, 'Cell Type'].notna().all(), 'RGC types sheet: RGC rows without Cell Type'
    assert sheet.loc[~is_rgc, 'Cell Type'].isna().all(), 'RGC types sheet: override rows (not RGC) with a Cell Type'
    return sheet


def _differs(a, b):
    """Elementwise a != b for label Series, with missing == missing."""
    return ~((a == b) | (a.isna() & b.isna()))


def _yes_no(mask):
    return np.where(mask, 'yes', 'no')


def _stale(types, classes, in_sheet, sheet_types):
    """Cells not in the sheet whose type label has to go (see module docstring)."""
    names = set(sheet_types.dropna()) | set(OLD_RGC_TYPE_NAMES)
    return ~in_sheet & types.notna() & ((classes == 'RGC') | types.isin(names))


def same_type_both_strong(df):
    """Cells whose parquet label was 'both_strong' (classifier and human label agreed) for the same type they have now,
    i.e. not moved to another type since the export. Use it to prefer example cells."""
    return ((df['celltype_final_decision_parquet'] == 'both_strong')
            & (df['celltype_final_parquet_renamed'] == df['celltype_final']))


def match_parquet(df, sheet):
    """Sheet row label (``sheet.index``) per parquet row, NaN where unmatched: on Index, then on segment ID."""
    by_index = pd.Series(sheet.index, index=sheet['Index'])
    match = df['Index'].map(by_index)

    no_index = df['Index'].isna()
    if no_index.any():
        seg = pd.concat([sheet['Latest SegID'], sheet['Proofread SegID']]).dropna()
        seg = seg.groupby(seg).filter(lambda s: s.index.nunique() == 1)           # segment IDs of exactly one row
        seg = pd.Series(seg.index, index=seg.values)
        seg = seg[~seg.index.duplicated()]
        by_seg = pd.Series(df.index.astype(str), index=df.index).map(seg)
        taken = set(match.dropna())
        by_seg[by_seg.isin(taken)] = np.nan                                         # row already matched on Index
        match = match.where(~no_index, by_seg)

    dup = match.dropna()[match.dropna().duplicated()]
    assert dup.empty, f'RGC types sheet rows matched to more than one parquet cell: {sheet.loc[dup, "Index"].tolist()}'
    return match


def apply_to_parquet(df, sheet, verbose=True):
    """Apply the RGC types sheet to a ``df_all_neurons`` frame (needs the ``PARQUET_LABEL_COLS``). Returns a copy."""
    missing = [c for c in PARQUET_LABEL_COLS if c not in df.columns]
    assert not missing, f'columns needed to apply the RGC types sheet are missing: {missing}'
    df = df.copy()
    for c in PARQUET_LABEL_COLS[1:]:
        df[f'{c}_parquet'] = df[c]
    df['celltype_final_parquet_renamed'] = df['celltype_final'].map(lambda t: RENAMED.get(t, t))

    match = match_parquet(df, sheet)
    in_sheet = match.notna()
    rows = sheet.loc[match[in_sheet]]
    new_type = pd.Series(rows['Cell Type'].values, index=df.index[in_sheet])
    new_class = pd.Series(rows['Cell Class'].values, index=df.index[in_sheet])
    has_type = new_type.notna()
    typed = has_type[has_type].index

    class_changed = in_sheet.copy()
    class_changed[in_sheet] = _differs(df.loc[in_sheet, 'cellclass_final'], new_class)

    df.loc[in_sheet, 'celltype_final'] = new_type
    df.loc[in_sheet, 'cellclass_final'] = new_class
    df.loc[in_sheet, 'celltype_final_decision'] = np.nan
    df.loc[typed, 'celltype_final_decision'] = REFINED
    df.loc[in_sheet, 'valid_celltype_final'] = has_type
    df.loc[in_sheet, 'valid_cellclass_final'] = True
    df.loc[class_changed, 'cellclass_final_decision'] = REFINED
    # the sheet's labels are human labels: also the human-label columns (celltype / cellclass, used e.g. to tell
    # labelled cells from classifier candidates in the mosaic plots)
    df.loc[in_sheet, 'celltype'] = new_type
    df.loc[in_sheet, 'cellclass'] = new_class
    df.loc[in_sheet, 'valid_celltype'] = has_type
    df.loc[in_sheet, 'valid_cellclass'] = True

    stale = _stale(df['celltype_final'], df['cellclass_final'], in_sheet, sheet['Cell Type'])
    df.loc[stale, 'celltype_final'] = np.nan
    df.loc[stale, 'celltype_final_decision'] = np.nan
    df.loc[stale, 'valid_celltype_final'] = False
    stale_human = _stale(df['celltype'], df['cellclass_final'], in_sheet, sheet['Cell Type'])
    df.loc[stale_human, 'celltype'] = np.nan
    df.loc[stale_human, 'valid_celltype'] = False

    df[FROM_SHEET] = _yes_no(in_sheet)
    df[UPDATED_TYPE] = _yes_no(_differs(df['celltype_final'], df['celltype_final_parquet']))
    df[UPDATED_CLASS] = _yes_no(_differs(df['cellclass_final'], df['cellclass_final_parquet']))

    if verbose:
        n_unmatched = len(sheet) - in_sheet.sum()
        print(f"RGC types sheet: {in_sheet.sum()} / {len(sheet)} rows matched to the parquet "
              f"({n_unmatched} cells are not in the parquet); {(df[UPDATED_TYPE] == 'yes').sum()} type and "
              f"{(df[UPDATED_CLASS] == 'yes').sum()} class labels changed, {stale.sum()} stale type labels removed")
    return df


def apply_to_main_sheet(df_main, sheet, verbose=True):
    """Apply the RGC types sheet to the main cell sheet (columns 'Index', 'Cell Type', 'Cell Class'). Returns a copy.

    The original values are kept in 'Cell Type (sheet)' and 'Cell Class (sheet)'.
    """
    df = df_main.copy()
    for c in ['Cell Type', 'Cell Class']:
        if c not in df.columns:      # dropped by dropna(axis=1, how='all') if empty
            df[c] = np.nan
        df[f'{c} (sheet)'] = df[c]
    by_index = sheet.set_index('Index')
    in_sheet = df['Index'].isin(by_index.index)
    df.loc[in_sheet, 'Cell Type'] = df.loc[in_sheet, 'Index'].map(by_index['Cell Type'])
    df.loc[in_sheet, 'Cell Class'] = df.loc[in_sheet, 'Index'].map(by_index['Cell Class'])

    stale = _stale(df['Cell Type'], df['Cell Class'], in_sheet, sheet['Cell Type'])
    df.loc[stale, 'Cell Type'] = np.nan

    df[FROM_SHEET] = _yes_no(in_sheet)
    df[UPDATED_TYPE] = _yes_no(_differs(df['Cell Type'], df['Cell Type (sheet)']))
    df[UPDATED_CLASS] = _yes_no(_differs(df['Cell Class'], df['Cell Class (sheet)']))
    if verbose:
        print(f"RGC types sheet: {in_sheet.sum()} / {len(sheet)} rows in the main sheet; "
              f"{(df[UPDATED_TYPE] == 'yes').sum()} type and {(df[UPDATED_CLASS] == 'yes').sum()} class labels changed, "
              f"{stale.sum()} stale type labels removed")
    return df
