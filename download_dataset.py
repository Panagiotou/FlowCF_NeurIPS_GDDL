"""Download the raw files for the six datasets of paper Table 2.

    python download_dataset.py                  # all six
    python download_dataset.py adult magic      # a subset

Each dataset is described by data/Info/<name>.json, which also fixes the column
names, types and target. Five datasets download without credentials; Lending Club
comes from Kaggle and needs an API token in ~/.kaggle/kaggle.json (see the README).
Afterwards run process_dataset.py to build the train/test splits.
"""

import json
import os
import sys
import zipfile
from urllib import request

import pandas as pd

DATA_DIR = 'data'

# UCI archives: the zip contains the raw file named in the Info file.
UCI_ZIP = {
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
    'default': 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
    'magic': 'https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip',
    'htru2': 'https://archive.ics.uci.edu/static/public/372/htru2.zip',
}
# Single CSV served over https.
CSV_URL = {
    'bank': 'https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/bank-full.csv',
}
# Kaggle dataset slugs.
KAGGLE = {
    'lending-club': 'wordsforthewise/lending-club',
}

ALL = list(UCI_ZIP) + list(CSV_URL) + list(KAGGLE)


def info_for(name):
    with open(f'{DATA_DIR}/Info/{name}.json') as f:
        return json.load(f)


def download_uci(name):
    """Download and unzip a UCI archive; for magic/htru2 also normalise the raw
    file into the headered CSV that process_dataset.py expects."""
    save_dir = f'{DATA_DIR}/{name}'
    os.makedirs(save_dir, exist_ok=True)
    zip_path = f'{save_dir}/{name}.zip'
    if not os.path.exists(zip_path):
        request.urlretrieve(UCI_ZIP[name], zip_path)
        print(f'{name}: downloaded {UCI_ZIP[name]}')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(save_dir)

    info = info_for(name)
    if 'raw_file' not in info:      # adult / default are read from the raw file directly
        return
    target = info['target_col']
    raw = pd.read_csv(f'{save_dir}/{info["raw_file"]}', header=None, names=info['raw_columns'])
    df = raw[[c for c in info['column_names'] if c != target] + [target]].copy()
    if 'label_map' in info:
        # 1 is the favourable (target) class.
        df[target] = df[target].astype(str).str.strip().map(info['label_map'])
        if df[target].isna().any():
            raise ValueError(f'{name}: raw target values not covered by label_map')
    df.to_csv(info['data_path'], index=False)
    print(f'{name}: wrote {info["data_path"]}  ({len(df)} rows)')


def download_csv(name):
    save_dir = f'{DATA_DIR}/{name}'
    os.makedirs(save_dir, exist_ok=True)
    pd.read_csv(CSV_URL[name]).to_csv(info_for(name)['data_path'])
    print(f'{name}: downloaded {CSV_URL[name]}')


def download_kaggle(name):
    import kaggle
    save_dir = f'{DATA_DIR}/{name}'
    os.makedirs(save_dir, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE[name], path=save_dir, unzip=False)
    with zipfile.ZipFile(f'{save_dir}/{name}.zip') as z:
        z.extractall(save_dir)
    print(f'{name}: downloaded {KAGGLE[name]}')


if __name__ == '__main__':
    for name in (sys.argv[1:] or ALL):
        if os.path.exists(info_for(name)['data_path']):
            print(f'{name}: already downloaded.')
        elif name in UCI_ZIP:
            download_uci(name)
        elif name in CSV_URL:
            download_csv(name)
        else:
            download_kaggle(name)
