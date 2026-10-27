"""Turn the raw files fetched by download_dataset.py into train/test splits.

For each dataset it reads data/Info/<name>.json (columns, types, target, split
sizes), writes data/<name>/{train.csv,test.csv} and the matching .npy matrices, and
saves the resolved metadata as data/<name>/info.json, which every method reads.

    python process_dataset.py                    # all six datasets
    python process_dataset.py --dataname magic
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

INFO_PATH = 'data/Info'

# The six datasets of paper Table 2.
DATASETS = ['adult', 'bank', 'default', 'lending-club', 'magic', 'htru2']

parser = argparse.ArgumentParser(description='process dataset')

# General configs
parser.add_argument('--dataname', type=str, default=None, help='Name of dataset.')
args = parser.parse_args()

def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.iloc[train_idx]
        test_df = data_df.iloc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1
        
    return train_df, test_df, seed


def process_data(name):
    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        info = json.load(f)

    data_path = info['data_path']
    if info['file_type'] == 'csv':
        data_df = pd.read_csv(data_path, header = info['header'], low_memory=False)
    elif info['file_type'] == 'xls':
        data_df = pd.read_excel(data_path, sheet_name='Data', header=1)
        data_df = data_df.drop('ID', axis=1)

    unnamed_cols = [col for col in data_df.columns.map(str) if 'Unnamed' in col]
    data_df = data_df.drop(columns=unnamed_cols)


    if name == 'lending-club':
        selected_columns = info["column_names"]
        data_df = data_df[selected_columns]
        data_df = data_df.dropna()

    if "preprocess_columns" in info: 
        for col, valid in info["preprocess_columns"].items(): 
            if name == 'lending-club':
                data_df[col] = data_df[col].apply(lambda x: x if x in valid else None)
            else:
                data_df[info["column_names"].index(col)] = data_df[info["column_names"].index(col)].apply(lambda x: x if x in valid else 'Other')

        
    if name == 'lending-club':
        data_df = data_df.dropna()
        data_df = data_df[:info["samples_keep"]]




    target_col = info['target_col']

    ignore_columns = info["ignore_columns"]
    ignore_columns_index = [info["column_names"].index(x) for x in ignore_columns]
    data_df = data_df.drop(ignore_columns_index, axis=1)

    if target_col in data_df:
        data_df = data_df[[col for col in data_df if col != target_col] + [target_col]]


    num_data = data_df.shape[0]

    column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()
    
    column_names = [x for x in column_names if x not in ignore_columns]

    if "num_col_idx" in info:
        num_col_idx = info['num_col_idx']
        cat_col_idx = info['cat_col_idx']
        target_col_idx = info['target_col_idx']
    else:
        num_col_idx = [idx for idx, col in enumerate(column_names) if info["column_info"][col] == "float"]
        cat_col_idx = [idx for idx, col in enumerate(column_names) if info["column_info"][col] == "str" and col != target_col]
        target_col_idx = [idx for idx, col in enumerate(column_names) if col == target_col]


    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]



    if info['test_path']:
        # if testing data is given
        test_path = info['test_path']

        with open(test_path, 'r') as f:
            lines = f.readlines()[1:]
            test_save_path = f'data/{name}/test.data'
            if not os.path.exists(test_save_path):
                with open(test_save_path, 'a') as f1:     
                    for line in lines:
                        save_line = line.strip('\n').strip('.')
                        f1.write(f'{save_line}\n')

        test_df = pd.read_csv(test_save_path, header = None)

        test_df = test_df.drop(ignore_columns_index, axis=1)

        
        if "preprocess_columns" in info: 
            for col, valid in info["preprocess_columns"].items(): 
                test_df[info["column_names"].index(col)] = test_df[info["column_names"].index(col)].apply(lambda x: x if x in valid else 'Other')


        train_df = data_df

    else:  
        # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)

        if "train_test_split" in info:
            num_train = int(num_data*float(info["train_test_split"]))
        else:
            num_train = int(num_data*0.9)
        num_test = num_data - num_train

        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
    

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))


    # print(name, train_df.shape, test_df.shape, data_df.shape)


    col_domain = {}

    col_info = {}
    
    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        col_info['max'] = float(train_df[col_idx].max())
        col_info['min'] = float(train_df[col_idx].min())
        col_name = column_names[col_idx]
        col_domain[col_name] = [float(train_df[col_idx].min()), float(train_df[col_idx].max())]
     
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        col_info['categorizes'] = list(set(train_df[col_idx]))    
        col_name = column_names[col_idx]
        col_domain[col_name] = train_df[col_idx].unique().tolist()

    for col_idx in target_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'

        col_info['categorizes'] = list(set(train_df[col_idx]))      




    if name=='adult':
        info["target_class"] = ' >50K'
        info["negative_class"] = ' <=50K'
    elif name == 'lending-club':
        info["target_class"] = 'Fully Paid'
        info["negative_class"] = 'Charged Off'
    elif name == 'bank':
        info["target_class"] = 'yes'
        info["negative_class"] = 'no'
    else:
        print("Warning: Setting target class for dataset to 1")
        info["target_class"] = 1
        info["negative_class"] = 0

    info['column_info'] = col_info
    # print(col_domain)
    info['column_domain'] = col_domain

    train_df.rename(columns = idx_name_mapping, inplace=True)
    test_df.rename(columns = idx_name_mapping, inplace=True)


    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'
    for col in num_columns:
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == '?', col] = 'nan'


    
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()


    input_col_order = num_columns + cat_columns

    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

 
    save_dir = f'data/{name}'
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
    np.save(f'{save_dir}/y_train.npy', y_train)

    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
    np.save(f'{save_dir}/y_test.npy', y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)


    train_df.to_csv(f'{save_dir}/train.csv', index = False)
    test_df.to_csv(f'{save_dir}/test.csv', index = False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['num_max'] = dict(zip(num_columns, X_num_train.max(axis=0).astype(str)))
    info['num_min'] = dict(zip(num_columns, X_num_train.min(axis=0).astype(str)))

    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]

    info["input_col_order"] = input_col_order


    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping 

    info['num_col_idx'] = num_col_idx
    info['cat_col_idx'] = cat_col_idx
    info['target_col_idx'] = target_col_idx

    # num_col_idx = info['num_col_idx']
    # cat_col_idx = info['cat_col_idx']
    # target_col_idx = info['target_col_idx']

    with open(f'{save_dir}/info.json', 'w') as file:
        json.dump(info, file, indent=4)

    print(f'Processing and Saving {name} Successfully!')

    print(name)
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'])
    print('Test', info['test_num'])

    cat = len(cat_col_idx)
    num = len(num_col_idx)
    print('Num', num)
    print('Cat', cat)

    print("Numeric", num_columns)
    print("Categorical", cat_columns)
    print("Classification Target", target_columns)


if __name__ == "__main__":

    for name in ([args.dataname] if args.dataname else DATASETS):
        process_data(name)

        

