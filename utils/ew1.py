import re

import pandas as pd

ew1_urls = {
    'OFF-SAC': "https://museum.eyewire.org/?neurons=70014,70016,70023,70024,70025,70026,70027,70030,70031,70032,70033,70034,70035,70048,70050,70066,70068,70076,70077,70079,70080,70081,70082,70083,70084,70085,70086,70087,70088,70089,70090,70093,70095,70096,70099,70100,70102,70106,70108,70109,70110,70111,70112,70113,70114,70115,70116,70117,70118,70119,70120,70121,70122,70123,70124,70125,70126,70127,70128,70129,70130,70131,70133,70134,70137,70138,70141,70145,70146,70147,70148,70149,70150,70151,70152,70154,70155,70156,70158,70167&browse=1",
    'ON-SAC': "https://museum.eyewire.org/?neurons=20025,20030,20032,20044,20062,20196,20204,26007,26009,26010,26011,26012,26013,26014,26015,26016,26017,26139,26143,26169,26174,26180,26183,26184,26186,26197,70028,70029,70161,70162,70163,70164,70168,70169,70170,70171,70172,70174,70176,70178,70179,70180,70181,70182,70183,70184,70185,70186,70187,70188,70189,70191,70192,70193,70194,70195,70196,70197,70198,70199,70200,70201,70202,70203,70204,70205,70206,70207,70208,70209,70211,70212,70213,70214,70215,70216,70217,70218,70219,70220,70221,70222,70223,70224,70225,70227,70228,70229,70230,70231,70232,70233,70234,70235,70236,70237,70238,70239,70240,70241,70242,70243,70244&browse=1",
    '1ni': "https://museum.eyewire.org/?neurons=17182,20132,20164,26019,26087&browse=1",
    '1no': "https://museum.eyewire.org/?neurons=10007,17021,17050,17110,17236,20092,20157,26024&browse=1",
    '1ws': "https://museum.eyewire.org/?neurons=20029,20203&browse=1",
    '1wt': "https://museum.eyewire.org/?neurons=10018,17109,26022&browse=1",
    '25': "https://museum.eyewire.org/?neurons=17132,17176,20036,20045,20067,20104,20105,20186,20237,25006,26031,26037,26040,26042,26060,26066,26099,26117,26134,26145,26167,26175&browse=1",
    '27': "https://museum.eyewire.org/?neurons=17212,20117,26051,26065&browse=1",
    '28': "https://museum.eyewire.org/?neurons=20155,20163,20243,20257,26005&browse=1",
    '2an': "https://museum.eyewire.org/?neurons=10010,10017,15018,15066,17027,17062,17105,17130,17177,20024,20060,20066,20101,20147,20168,20264,26026,26041,26049,26082,26129,26147,26172,26190,50001&browse=1",
    '2aw': "https://museum.eyewire.org/?neurons=17024,17028,17060,17061,17075,17107,17200,17205,20047,20103,20201,26003,26018,26038,26055,26095,26110,26150,26163,26189,26193&browse=1",
    '2i': "https://museum.eyewire.org/?neurons=17013,17092,17144,17192,20051,20082,20234,26109,26126,26131,26157,50004&browse=1",
    '2o': "https://museum.eyewire.org/?neurons=10005,10013,17216,26062,26118&browse=1",
    '37c': "https://museum.eyewire.org/?neurons=20002,20179,20210,20245,20254,26036,26047,26056,26101,26137&browse=1",
    '37d': "https://museum.eyewire.org/?neurons=20014,20125,26029,26094,26162,90002&browse=1",
    '37r': "https://museum.eyewire.org/?neurons=17080,20213,20220,25005,26032,26084,26103,26165,90001&browse=1",
    '37v': "https://museum.eyewire.org/?neurons=17161,20016,20096,20137,20233,26115,26138,26158,26178&browse=1",
    '3i': "https://museum.eyewire.org/?neurons=17077,17135,20107,26063,26104,26116&browse=1",
    '3o': "https://museum.eyewire.org/?neurons=17037,17076,20121,26155,26188&browse=1",
    '4i': "https://museum.eyewire.org/?neurons=17022,17057,17247,20170,20174,25004,26006,26008,26050,26096,26102,26164&browse=1",
    '4on': "https://museum.eyewire.org/?neurons=17034,17064,17151,17167,20041,20230,26021,26086,26121,26146,26160&browse=1",
    '4ow': "https://museum.eyewire.org/?neurons=17079,17188,20156,26004&browse=1",
    '51': "https://museum.eyewire.org/?neurons=17011,17012,17035,17095,17098,17138,20037,20120,20153,20182,20212,20258,26025,26039,26054,26085,26113,26136,26154,26177&browse=1",
    '5si': "https://museum.eyewire.org/?neurons=17040,17055,17071,20070,20135,20183,26044,26106,26133,26142&browse=1",
    '5so': "https://museum.eyewire.org/?neurons=17081,17127,17146,17160,17168,20012,20053,20223,26046,26098,26111,26122,26140,26151,26159,26181,26187&browse=1",
    '5ti': "https://museum.eyewire.org/?neurons=17059,17078,17090,17093,17121,17159,17181,17190,20055,20089,20097,20102,20114,20127,20184,20191,20216,20226,20262,26053,26083,26112,26120,26123,26144,26152,26156,26161,26170,50002&browse=1",
    '5to': "https://museum.eyewire.org/?neurons=20128,20165,20240&browse=1",
    '63': "https://museum.eyewire.org/?neurons=17084,17097,17114,17140,20005,20011,20019,20071,20129,20140,20178,20181,20208,26023,26027,26028,26057,26068,26089,26125,26141,26148,26191,30002,30003&browse=1",
    '6sn': "https://museum.eyewire.org/?neurons=17082,20073,20198,26035,26043,26171&browse=1",
    '6sw': "https://museum.eyewire.org/?neurons=17083,20068,20217,20222,26020&browse=1",
    '6t': "https://museum.eyewire.org/?neurons=20113,20232,20255&browse=1",
    '72': "https://museum.eyewire.org/?neurons=17069,20074,20166,20221,26124&browse=1",
    '73': "https://museum.eyewire.org/?neurons=20043,20100,20150,20187,26059,26073,26132&browse=1",
    '7id': "https://museum.eyewire.org/?neurons=26070,26075,26078&browse=1",
    '7ir': "https://museum.eyewire.org/?neurons=20021,20075,26002,26128&browse=1",
    '7iv': "https://museum.eyewire.org/?neurons=17152,26077&browse=1",
    '7o': "https://museum.eyewire.org/?neurons=17053,20180,20239,26034,26048,26100,26130&browse=1",
    '81i': "https://museum.eyewire.org/?neurons=20158,26090,26097&browse=1",
    '81o': "https://museum.eyewire.org/?neurons=20069&browse=1",
    '82n': "https://museum.eyewire.org/?neurons=20161,20167,20197,26052,26058,26072,26076&browse=1",
    '82wi': "https://museum.eyewire.org/?neurons=20251,26067,30001&browse=1",
    '82wo': "https://museum.eyewire.org/?neurons=20118,26091&browse=1",
    '85': "https://museum.eyewire.org/?neurons=17038,20046,20063,20072,20200,26045,26061,26092&browse=1",
    '8n': "https://museum.eyewire.org/?neurons=20126&browse=1",
    '8w': "https://museum.eyewire.org/?neurons=17111,26001,26071,26079&browse=1",
    '91': "https://museum.eyewire.org/?neurons=20020,20042,20081,20218,25003,26080,26088&browse=1",
    '915': "https://museum.eyewire.org/?neurons=17009,20080,26033,26119&browse=1",
    '9n': "https://museum.eyewire.org/?neurons=20006,20056,20076,20112,26074,26127,26135,26149,26168&browse=1",
    '9w': "https://museum.eyewire.org/?neurons=20228&browse=1",
}

# "https://museum.eyewire.org/?neurons=" + ','.join([str(i) for i in df_labels.index]) + "&browse=1"

EXCLUDE_IDS = [
    "50001", # Fragment
    "30002", # Fragment
    "17090", # Fragment
    "17021", # Fragment
    "17152", # Fragment
    "17012", # Severely clipped
    "26177", # Severely clipped
    "26154", # Severely clipped
    "26126", # Severely clipped
]


def extract_neuron_ids(url):
    """
    Extract neuron IDs from an Eyewire museum URL.

    Args:
        url (str): The Eyewire museum URL containing neuron IDs

    Returns:
        list: A list of integer neuron IDs
    """
    # Find the substring between "neurons=" and "&browse"
    start_marker = "neurons="
    end_marker = "&browse"

    start_idx = url.find(start_marker)
    if start_idx == -1:
        return []

    start_idx += len(start_marker)
    end_idx = url.find(end_marker, start_idx)

    if end_idx == -1:
        # If "&browse" is not found, take the rest of the string
        neuron_str = url[start_idx:]
    else:
        neuron_str = url[start_idx:end_idx]

    # Split by comma and convert to integers
    neuron_ids = [str(id_str) for id_str in neuron_str.split(',')]

    neuron_ids = list(set(neuron_ids) - set(EXCLUDE_IDS))  # Remove excluded IDs

    return neuron_ids


def get_label_palette():
    labels = ['1ni', '1no', '1ws', '1wt', '25', '27', '28', '2an', '2aw', '2i',
              '2o', '37', '37c', '37d', '37r', '37v', '3i', '3o', '4i', '4on', '4ow',
              '51', '5si', '5so', '5ti', '5to', '63', '6sn', '6sw', '6t', '72',
              '73', '7i', '7id', '7ir', '7iv', '7o', '81', '81i', '81o',
              '82', '82n', '82wi', '82wo', '85', '8n', '8w', '91', '915', '9n', '9w']

    import matplotlib.cm as cm

    # Group labels by their leading digit
    from collections import defaultdict
    label_groups = defaultdict(list)
    for label in labels:
        leading_digit = label[0]
        label_groups[leading_digit].append(label)

    # Create color map per group
    palette = {}
    colormaps = [cm.Wistia, cm.Oranges, cm.Reds, cm.Purples, cm.Blues, cm.BuGn, cm.Greens, cm.Greys, cm.cool]
    group_ids = sorted(label_groups.keys())
    assert len(group_ids) <= len(colormaps), "Not enough base colormaps for leading digit groups."

    for i, digit in enumerate(group_ids):
        group = label_groups[digit]
        cmap = colormaps[i]
        colors = [cmap((j + 5) / (len(group) + 5)) for j in range(1, len(group) + 1)]
        for label, color in zip(group, colors):
            palette[label] = color

    return palette


def get_df_ew1(add_categories=False, exclude_types=('OFF-SAC', 'ON-SAC')):
    ew1_rgc_labels = {cell_id: key
                      for key, url in ew1_urls.items()
                      for cell_id in extract_neuron_ids(url)
                      if key not in exclude_types}

    df1 = pd.DataFrame({"cell_id": ew1_rgc_labels.keys(), "ew1_label": ew1_rgc_labels.values()})
    df1.set_index('cell_id', inplace=True)

    if add_categories:
        df1 = extract_categories(df1)

    return df1

def extract_categories(df1):
    df1['nw'] = df1['ew1_label'].apply(extract_nw)
    df1['a'] = df1['ew1_label'].apply(extract_a)
    df1['st'] = df1['ew1_label'].apply(extract_st)
    df1['io'] = df1['ew1_label'].apply(extract_io)
    df1['nw_st_io_a'] = df1['nw'] + df1['st'] + df1['io'] + df1['a']
    return df1

def extract_nw(label):
    if 'n' in label:
        return 'n'
    elif 'w' in label:
        return 'w'
    else:
        return '-'


def extract_a(label):
    if 'a' in label:
        return 'a'
    else:
        return '-'


def extract_st(label):
    if 's' in label:
        return 's'
    elif 't' in label:
        return 't'
    else:
        return '-'


def extract_io(label):
    if 'i' in label:
        return 'i'
    elif 'o' in label:
        return 'o'
    else:
        return '-'

def simplify_label(label, merge_37=True, merge_82=True, merge_7i=True, merge_71=True, merge_81=True):
    if label.startswith('37') and merge_37:
        return '37'
    elif label.startswith('7i') and merge_7i:
        return '7i'
    elif label.startswith('82') and merge_82:
        return '82'
    elif label.startswith('81') and merge_81:
        return '81'
    elif label.startswith('71') and merge_71:
        return '71'
    else:
        return label


def clean_label(label):
    if label is None:
        return None

    # Convert to string if not already
    label = str(label)

    if label.lower() == 'none':
        return None
    if label.lower() == 'nan':
        return None
    if label.lower() == 'n/a':
        return None
    if label.lower() == 'na':
        return None

    # Rule 1: If starts with 37, simplify to 37
    if label.startswith('37'):
        return '37'

    if label.startswith('7i'):
        return '7i'

    if label.lower().startswith('pixon'):
        return '4on'

    # Rule 2: If there are two options separated by 'or' or 'and', take the first
    repeat = True
    while repeat:
        if ' or ' in label:
            label = label.split(' or ')[0]
        elif ' and ' in label:
            label = label.split(' and ')[0]
        elif ', ' in label:
            # Handle comma-separated options
            label = label.split(', ')[0]
        else:
            repeat = False

    # Rule 3: Remove question marks and other punctuation
    label = re.sub(r'[?!]', '', label)

    # Remove any trailing/leading whitespace
    label = label.strip()

    # Remove any additional annotations (like "DB::wk bistrat", "(symmetric)")
    if ' ' in label and not any(char.isalpha() for char in label.split()[0]):
        # If first part is not alphabetic, keep only the first part
        label = label.split()[0]

    return label