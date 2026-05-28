import numpy as np
import pandas as pd


EW_TO_NAME = {
    "2o": "Bursty suppressed by contrast",
    "1ws": "M1",
    # "1ws": "ON sustained - subtype unknown",
    "2an": "F-mini-OFF",
    "63": "F-mini-ON",
    "5si": "HD1",
    "5so": "HD2",
    "51": "Local edge detector",
    "9w": "M2",
    "91": "M6",
    "5to": "EW5to",
    "2aw": "OFF h/v OS - a/symmetric",
    "2i": "OFF medium sustained",
    "1wt": "OFF sustained alpha",
    "1no": "OFF sustained EW1no",
    "3o": "OFF sustained EW3o",
    "4ow": "OFF transient alpha",
    "4on": "OFF transient medium RF",
    "4i": "OFF transient small RF",
    "8w": "ON alpha",
    "3i": "ON bursty",
    "73": "ON delayed",
    "7iv,r,d": "ON DS sustained - direction unknown",
    "7iv": "ON DS sustained - dorsonasal",
    "7ir": "ON DS sustained - temporal",
    "7id": "ON DS sustained - ventral",
    "7o": "ON DS transient - temporal",
    "1ni": "ON small OFF large",
    "6t": "ON transient EW6t",
    "6sw": "ON transient medium RF",
    "6sn": "ON transient small RF",
    "37": "ON-OFF DS - direction unknown",
    "37v": "ON-OFF DS - dorsal",
    "37c": "ON-OFF DS - nasal",
    "37r": "ON-OFF DS - temporal",
    "37d": "ON-OFF DS - ventral",
    "9n": "PixON",
    "27": "Sustained suppressed-by-contrast no surround EW27",
    "28": "Sustained suppressed-by-contrast strong surround EW28",
    "5ti": "UHD",
    "25": "EW25",
    "915": "EW915",
    "85": "EW85",
    '82n': "EW82",
    '82wi': "EW82",
    '82wo': "EW82",
}



def clean_labels(df, file_path_mapper, celltype_col='celltype',
                 remove_suffixes=True, shorten_sbc=False):

    df_mapper = pd.read_csv(file_path_mapper, dtype=str)

    merged_col = f'{celltype_col}_merged'
    short_col = f'{celltype_col}_short'

    cell_type_names = set(df_mapper['Cell type name'].dropna())
    helmsteader_names = set(df_mapper['Helmstaeder name'].dropna())
    other_names = set(df_mapper['Other names'].dropna())

    assert np.all(
        np.unique(
            df_mapper[df_mapper['Short name for MS'].notnull()]['Short name for MS'],
            return_counts=True
        )[1] == 1
    )

    df[merged_col] = df[celltype_col]

    for i, row in df[df[celltype_col].notnull()].iterrows():
        val = row[celltype_col]

        if val in EW_TO_NAME:
            df.loc[i, merged_col] = EW_TO_NAME[val]
        elif val in cell_type_names:
            continue
        elif val in helmsteader_names:
            mapped = df_mapper.loc[
                df_mapper['Helmstaeder name'] == val, 'Cell type name'
            ].iloc[0]
            df.loc[i, merged_col] = mapped
        elif val in other_names:
            df.loc[i, merged_col] = 'other'


    name2short = {
        row['Cell type name']: row['Short name for MS']
        for _, row in df_mapper[df_mapper['Short name for MS'].notnull()].iterrows()
    }

    df[short_col] = df[celltype_col].apply(lambda x: name2short.get(x, x))

    if remove_suffixes:
        for suffix in [
            ' - nasal', ' - dorsal', ' - temporal', ' - ventral', ' - dorsonasal',
            ' - direction unknown', ' - orientation unknown',
        ]:
            df[merged_col] = df[merged_col].str.replace(suffix, '', regex=False)

        for suffix in [ '-dn', '-n', '-d', '-t', '-v',]:
            df[short_col] = df[short_col].str.replace(suffix, '', regex=False)

    if remove_suffixes:
        for suffix in [

        ]:
            df[merged_col] = df[merged_col].str.replace(suffix, '', regex=False)

    if shorten_sbc:
        df[merged_col] = df[merged_col].str.replace(
            'suppressed by contrast', 'SbC', regex=False
        )
        df[merged_col] = df[merged_col].str.replace(
            'suppressed-by-contrast', 'SbC', regex=False
        )

    def check_label(l):
        if pd.isna(l) or not isinstance(l, str):
            return False
        ll = l.lower()
        if ll in {'unknown', 'other', 'new-other'}:
            return False
        if 'subtype unknown' in ll:
            return False
        return True

    df['is_labelled'] = df[merged_col].apply(check_label)

    return df
