"""Dataset plumbing shared by every method.

`preprocess` turns the .npy splits written by process_dataset.py into the encoded
matrices the black box was trained on, plus the inverse transforms a method needs
to map counterfactuals back to the input space.
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset

import data_encoding as enc

def get_name_form_args(args):
    opts_vae = []
    opts_vae.append(f"num_encoding[{args.num_encoding}]")
    opts_vae.append(f"num_loss[{args.num_reconstr_loss}]")

    if args.kl_weight > 0:
        opts_vae.append(f"kl_weight[{args.kl_weight}]")

    if args.gumbel_softmax:
        opts_vae.append(f"cat_actv[gumbel_softmax, tau={args.tau}]")

    if args.sigmoid:
        opts_vae.append("cont_actv[sigmoid]")
    else:
        opts_vae.append("cont_actv[linear]")

    if args.reparam:
        opts_vae.append("reparam")


    output_file_name_vae = "_".join(opts_vae)

    return output_file_name_vae

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat=None, y=None, info=None):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = None
        if y is not None:
            class_mapping = {info["target_class"]: 1, info["negative_class"]: 0}
            y_n = np.array([class_mapping[class_name[0]] for class_name in y])
            self.y = torch.tensor(y_n)

    def __getitem__(self, index):
        this_num = self.X_num[index]

        if self.X_cat is not None:
            this_cat = self.X_cat[index]
        else:
            this_cat = []


        if self.y is not None:
            this_y = self.y[index]
            return (this_num, this_cat, this_y)

        return (this_num, this_cat)

    def __len__(self):
        return self.X_num.shape[0]

def preprocess(dataset_path, task_type = 'binclass', inverse = False, cat_encoding = None, num_encoding='min_max_torch'):
    
    T_dict = {}

    T_dict['normalization'] = num_encoding
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = enc.Transformations(**T_dict)

    dataset = make_dataset(dataset_path, T, task_type)

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num['train'], X_num['test']
        if X_cat is None:
            X_train_cat = X_test_cat = None
        else:
            X_train_cat, X_test_cat = X_cat['train'], X_cat['test']

        categories = enc.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)

        y = (dataset.y['train'], dataset.y['test']) 


        if inverse:
            if dataset.num_transform is not None:
                num_inverse = dataset.num_transform.inverse_transform
            else:
                num_inverse = None
            if dataset.cat_transform is not None:
                cat_inverse = dataset.cat_transform.inverse_transform
            else:
                cat_inverse = None

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse, y
        else:
            return X_num, X_cat, categories, d_numerical, y
    elif cat_encoding=="one-hot" or cat_encoding=="one-hot-drop-bin":
        X_num = dataset.X_num

        X_train_num, X_test_num = X_num['train'], X_num['test']
        
        X_num = (X_train_num, X_test_num)

        y = (dataset.y['train'], dataset.y['test']) 


        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = (dataset.cat_transform.inverse_transform
                           if dataset.cat_transform is not None else None)

            return X_num, None, None, None, num_inverse, cat_inverse, y
        else:
            return X_num, None, None, None, y
    else:
        return dataset


def make_dataset(data_path: str, T: 'enc.Transformations', task_type):

    # classification
    X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
    X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
    y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

    for split in ['train', 'test']:
        X_num_t, X_cat_t, y_t = enc.read_pure_data(data_path, split)
        if X_num is not None:
            X_num[split] = X_num_t
        if X_cat is not None:
            X_cat[split] = X_cat_t
        if y is not None:
            y[split] = y_t

    # Continuous-only datasets still get a (n, 0) X_cat_*.npy from
    # process_dataset.py. Drop it so the pipeline takes its no-categoricals
    # path instead of handing a zero-width array to the one-hot encoder.
    if X_cat is not None and X_cat['train'].shape[1] == 0:
        X_cat = None

    info = enc.load_json(os.path.join(data_path, 'info.json'))

    D = enc.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=enc.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    return enc.transform_dataset(D, T)