"""Feature encoding shared by every method (from the TabDDPM/TABCF pipeline).

`utils_train.preprocess` builds, from the .npy splits written by
process_dataset.py, the encoded matrices the black box was trained on and every
method consumes: numerical features min-max or torch-min-max scaled, categorical
features one-hot encoded or left as ordinal codes, plus the fitted transforms so a
method can invert them.

This is a trimmed copy of the original pipeline, restricted to the transformations
the experiments use.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
import enum
import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np
import sklearn.preprocessing
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ArrayDict = Dict[str, np.ndarray]

CAT_MISSING_VALUE = 'nan'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['min_max_torch', 'standard_torch', 'standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'one-hot-drop-bin']
YPolicy = Literal['default']


class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def read_pure_data(path, split='train'):
    y = np.load(os.path.join(path, f'y_{split}.npy'), allow_pickle=True)
    X_num = X_cat = None
    if os.path.exists(os.path.join(path, f'X_num_{split}.npy')):
        X_num = np.load(os.path.join(path, f'X_num_{split}.npy'), allow_pickle=True)
    if os.path.exists(os.path.join(path, f'X_cat_{split}.npy')):
        X_cat = np.load(os.path.join(path, f'X_cat_{split}.npy'), allow_pickle=True)
    return X_num, X_cat, y


def get_categories(X_train_cat):
    return (None if X_train_cat is None
            else [len(set(X_train_cat[:, i])) for i in range(X_train_cat.shape[1])])


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'


# ── Numerical ─────────────────────────────────────────────────────────────────

class TorchMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = self.max = self.data_min = self.data_max = None

    def fit(self, x):
        x = self.check_make_tensor(x)
        self.data_min = torch.min(x, dim=0, keepdim=True).values
        self.data_max = torch.max(x, dim=0, keepdim=True).values
        self.min, self.max = self.feature_range

    def transform(self, x):
        x = self.check_make_tensor(x)
        x_std = (x - self.data_min) / (self.data_max - self.data_min + 1e-7)
        return x_std * (self.max - self.min) + self.min

    def inverse_transform(self, x):
        x = self.check_make_tensor(x)
        device = x.device
        x_std = (x - self.min) / (self.max - self.min + 1e-7)
        return x_std * (self.data_max.to(device) - self.data_min.to(device)) + self.data_min.to(device)

    @staticmethod
    def check_make_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


class TorchStandardScaler:
    def fit(self, x):
        x = self.check_make_tensor(x)
        self.mean = torch.mean(x, dim=0, keepdim=True)
        self.std = torch.std(x, dim=0, unbiased=False, keepdim=True)

    def transform(self, x):
        x = self.check_make_tensor(x)
        return (x - self.mean) / (self.std + 1e-7)

    def inverse_transform(self, x):
        x = self.check_make_tensor(x)
        device = x.device
        return x * (self.std.to(device) + 1e-7) + self.mean.to(device)

    @staticmethod
    def check_make_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):
        print('No NaNs in numerical features, skipping')
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks['test'].all(), \
            'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {k: v[valid_masks[k]] for k, v in data_dict.items()}
        return replace(dataset, **new_data)
    if policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        return replace(dataset, X_num=X_num)
    raise ValueError(f'unknown num_nan_policy: {policy}')


def normalize(X: ArrayDict, normalization: Normalization, seed: Optional[int],
              return_normalizer: bool = False):
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=int(1e9), random_state=seed)
    elif normalization == 'standard_torch':
        normalizer = TorchStandardScaler()
    elif normalization == 'min_max_torch':
        normalizer = TorchMinMaxScaler()
    else:
        raise ValueError(f'unknown normalization: {normalization}')

    normalizer.fit(X['train'])
    out = {k: normalizer.transform(v) for k, v in X.items()}
    return (out, normalizer) if return_normalizer else out


# ── Categorical ───────────────────────────────────────────────────────────────

def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if not any(x.any() for x in nan_masks.values()):
        assert policy is None
        return X
    if policy is None:
        return X
    if policy == 'most_frequent':
        imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)
        imputer.fit(X['train'])
        return {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
    raise ValueError(f'unknown categorical NaN policy: {policy}')


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append([(x if x in popular else CAT_RARE_VALUE)
                                for x in X[part][:, column_idx].tolist()])
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(X: ArrayDict, encoding: Optional[CatEncoding], seed: Optional[int]):
    """Returns (X, converted_to_numerical, encoder)."""
    if encoding is None:
        # Ordinal codes; unseen test categories become one past the training maximum.
        unknown_value = np.iinfo('int64').max - 3
        encoder = make_pipeline(sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=unknown_value, dtype='int64'))
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
        max_values = X['train'].max(axis=0)
        for part in X:
            if part == 'train':
                continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = \
                    max_values[column_idx] + 1
        return X, False, encoder

    if encoding in ('one-hot', 'one-hot-drop-bin'):
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, dtype=np.float32,
            **({'drop': 'if_binary'} if encoding == 'one-hot-drop-bin' else {}))
        encoder = make_pipeline(ohe)
        encoder.fit(X['train'])
        return {k: encoder.transform(v) for k, v in X.items()}, True, encoder

    raise ValueError(f'unknown cat_encoding: {encoding}')


def build_target(y: ArrayDict, policy: Optional[YPolicy],
                 task_type: TaskType) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy == 'default' and task_type == TaskType.REGRESSION:
        mean, std = float(y['train'].mean()), float(y['train'].std())
        y = {k: (v - mean) / std for k, v in y.items()}
        info['mean'], info['std'] = mean, std
    elif policy not in (None, 'default'):
        raise ValueError(f'unknown y_policy: {policy}')
    return y, info


def transform_dataset(dataset: Dataset, transformations: Transformations) -> Dataset:
    """Apply the transformations in order and stamp the fitted transforms on the
    dataset, so a method can map its counterfactuals back to the input space."""
    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None and X_num['train'].shape[-1] > 0:
        X_num, num_transform = normalize(
            X_num, transformations.normalization, transformations.seed, return_normalizer=True)

    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num, cat_transform = cat_encode(
            X_cat, transformations.cat_encoding, transformations.seed)
        if is_num:
            # one-hot columns are appended to the numerical block
            X_num = (X_cat if X_num is None
                     else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num})
            X_cat = None

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform
    return dataset
