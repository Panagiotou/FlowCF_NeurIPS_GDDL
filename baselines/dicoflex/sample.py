"""DiCoFlex [6] (`--method dicoflex`).

DiCoFlex trains a class-conditional Masked Autoregressive Flow and samples
counterfactuals from the target-class distribution conditioned on the factual.

Original code lives untouched under ./DiCoFlex/ (a vendored copy of
https://github.com/ofurman/DiCoFlex). We only IMPORT from it here; nothing in
that tree is modified. The only shims required are a `sys.path` insert and a
tiny `neptune` stub (DiCoFlex's maf.py/base.py do `import neptune` purely for a
type annotation).

This file mirrors the contract of the other baselines (baselines/dice/sample.py,
baselines/wachter/sample.py):

  * exposes `main(args)`, dispatched from main.py;
  * selects the first `--num_samples` test instances the black box assigns to the
    negative class;
  * writes `cfs.csv` (cols: id, <features>, target_prob) and
    `original_test_samples.csv` into
    counterfactual_results/<dataname>/DiCoFlex/<n>_samples/.

`target_prob` is the black box's target-class probability of each CF, obtained
through DiCE's `func="ohe-min-max"` model wrapper (the same encoder the black box
was trained with), so evaluate.py can recompute validity.
"""

import os
import sys
import time
import types
import logging
import warnings
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
)
from tqdm import tqdm

import dice_ml

from utils_train import preprocess
from black_box import BBMLPCLF

warnings.filterwarnings("ignore")

# ── Make the vendored DiCoFlex importable WITHOUT touching its code ──────────────
_DICOFLEX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DiCoFlex")
if _DICOFLEX_ROOT not in sys.path:
    sys.path.insert(0, _DICOFLEX_ROOT)

# DiCoFlex's maf.py / base.py `import neptune` only to type-annotate `neptune.Run`.
_neptune_stub = types.ModuleType("neptune")
_neptune_stub.Run = type("Run", (), {})
sys.modules.setdefault("neptune", _neptune_stub)

from counterfactuals.generative_models.maf import MaskedAutoregressiveFlow
from counterfactuals.datasets.generic_counterfactual import (
    MulticlassCounterfactualWrapper,
    train_multiclass_counterfactual_flow_model,
    generate_multiclass_counterfactuals,
)

logging.getLogger("counterfactual").setLevel(logging.WARNING)

# ── Paper hyperparameters (DiCoFlex Appendix E.1) ───────────────────────────────
N_NEAREST = 32
NUM_EPOCHS = 1000
PATIENCE = 300
HIDDEN_FEATURES = 64
NUM_LAYERS = 5
NUM_BLOCKS_PER_LAYER = 2
LEARNING_RATE = 1e-4
N_GEN_SAMPLES = 100        # candidates drawn per factual (DiCoFlex: "up to 100")
# DiCoFlex's own protocol (their §4.1): of the 100 candidates, keep only the ones the classifier
# says are VALID, then pick among those. One CF per factual is reported here, so the
# closest valid candidate is taken (falling back to the closest overall when none is
# valid, to keep coverage). The "closest" setting selects on proximity alone, ignoring
# validity, and is kept only for comparison.
SELECT = os.environ.get("DICOFLEX_SELECT", "valid")   # "valid" (DiCoFlex) | "closest"
# DiCoFlex §3.3: training targets are neighbours the classifier labels y' -- the reference
# driver filters them by posterior confidence. 0.0 disables the filter.
PROB_THRESHOLD = float(os.environ.get("DICOFLEX_PROB_THRESHOLD", 0.55))
TEMPERATURE = 0.8
P_TRAIN = (0.01, 2.0)      # norms conditioned on during training
P_INFERENCE = 2.0          # 2.0 = standard, 0.01 = sparse variant
MAX_TRAIN_SIZE = 3000      # per-class cap to bound DiCoFlex's distance matrix


class _BlackBoxAdapter:
    """DiCoFlex queries the classifier on points in ITS preprocessed space; the harness
    black box expects the original feature columns. This bridges the two so the paper's
    training-time confidence filter and validity-based selection can be applied."""

    def __init__(self, prep, dice_model, num_cols, cat_cols, feat_cols):
        self.prep, self.dice_model = prep, dice_model
        self.num_cols, self.cat_cols, self.feat_cols = num_cols, cat_cols, feat_cols

    def to_frame(self, X_prep):
        cont, cat = self.prep.inverse_transform(np.asarray(X_prep, dtype=np.float32))
        df = pd.DataFrame(index=range(len(cont)))
        for j, c in enumerate(self.num_cols):
            df[c] = cont[:, j]
        for j, c in enumerate(self.cat_cols):
            df[c] = cat[:, j]
        return df[self.feat_cols]

    def predict_proba(self, X_prep):
        return np.asarray(self.dice_model.get_output(self.to_frame(X_prep)))


class _DiCoFlexPreprocessor:
    """Converts the TABCF tabular format into DiCoFlex's continuous space.

    Layout produced: [ MinMax(continuous) | Quantile(noisy one-hot(categorical)) ],
    continuous columns first. Categoricals are integer-coded first (OrdinalEncoder)
    so we can map flow outputs back to the original string categories.
    """

    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.d_cont = len(num_cols)

        self.mm_scaler = MinMaxScaler()
        self.ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1)
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.qt = QuantileTransformer(output_distribution="normal",
                                      n_quantiles=1000, random_state=42)
        self._cat_slices = []
        self._cat_sizes = []

    def fit_transform(self, df):
        X_cont = df[self.num_cols].to_numpy(dtype=np.float32) if self.num_cols \
            else np.zeros((len(df), 0), dtype=np.float32)
        X_cat_int = self.ord_enc.fit_transform(df[self.cat_cols].astype(str)) \
            if self.cat_cols else np.zeros((len(df), 0))

        X_cont_scaled = (self.mm_scaler.fit_transform(X_cont)
                         if self.d_cont else X_cont)

        X_cat_ohe = self.ohe.fit_transform(X_cat_int) if self.cat_cols \
            else np.zeros((len(df), 0))
        col = 0
        for cats in self.ohe.categories_ if self.cat_cols else []:
            self._cat_sizes.append(len(cats))
            self._cat_slices.append(slice(col, col + len(cats)))
            col += len(cats)

        X_cat_qt = self._cat_to_gaussian(X_cat_ohe, fit=True)
        return np.concatenate([X_cont_scaled, X_cat_qt], axis=1).astype(np.float32)

    def transform(self, df):
        X_cont = df[self.num_cols].to_numpy(dtype=np.float32) if self.num_cols \
            else np.zeros((len(df), 0), dtype=np.float32)
        X_cat_int = self.ord_enc.transform(df[self.cat_cols].astype(str)) \
            if self.cat_cols else np.zeros((len(df), 0))

        X_cont_scaled = (self.mm_scaler.transform(X_cont)
                         if self.d_cont else X_cont)
        X_cat_ohe = self.ohe.transform(X_cat_int) if self.cat_cols \
            else np.zeros((len(df), 0))
        X_cat_qt = self._cat_to_gaussian(X_cat_ohe, fit=False)
        return np.concatenate([X_cont_scaled, X_cat_qt], axis=1).astype(np.float32)

    def _cat_to_gaussian(self, X_cat_ohe, fit):
        if not self.cat_cols:
            return np.zeros((len(X_cat_ohe), 0))
        noisy = X_cat_ohe + np.random.normal(0, 0.05, X_cat_ohe.shape)
        return self.qt.fit_transform(noisy) if fit else self.qt.transform(noisy)

    def inverse_transform(self, X_prep):
        """Flow output -> (cont float array, cat string DataFrame columns)."""
        X_prep = np.nan_to_num(X_prep, nan=0.0, posinf=1.0, neginf=0.0)
        X_cont_scaled = X_prep[:, :self.d_cont]
        X_cat_qt = X_prep[:, self.d_cont:]

        if self.d_cont:
            X_cont_scaled = np.clip(X_cont_scaled, 0.0, 1.0)
            cf_cont = self.mm_scaler.inverse_transform(X_cont_scaled).astype(np.float32)
            cf_cont = np.nan_to_num(cf_cont, nan=0.0)
        else:
            cf_cont = np.zeros((len(X_prep), 0), dtype=np.float32)

        if self.cat_cols:
            qt_min, qt_max = self.qt.quantiles_[0], self.qt.quantiles_[-1]
            X_cat_qt = np.clip(X_cat_qt, qt_min, qt_max)
            X_cat_soft = np.nan_to_num(self.qt.inverse_transform(X_cat_qt), nan=0.0)
            cat_int = np.stack(
                [X_cat_soft[:, sl].argmax(axis=1) for sl in self._cat_slices], axis=1
            )
            cf_cat = self.ord_enc.inverse_transform(cat_int)
        else:
            cf_cat = np.zeros((len(X_prep), 0))
        return cf_cont, cf_cat


def main(args):
    dataname = args.dataname
    data_dir = f"data/{dataname}"
    device = args.device
    num_samples = args.num_samples
    save_path = args.save_path

    import json
    with open(os.path.join(data_dir, "info.json")) as f:
        info = json.load(f)

    target_col = info["target_col"]
    target_class = info["target_class"]
    negative_class = info["negative_class"]
    target_class_index = 1

    full_cols = list(info["column_names"])
    feat_cols = [c for c in full_cols if c != target_col]
    num_cols = [full_cols[i] for i in info["num_col_idx"]]
    cat_cols = [full_cols[i] for i in info["cat_col_idx"]]

    # ── output folder, mirroring the other baselines ────────────────────────────
    output_folder_name = f"{dataname}/DiCoFlex/"
    if num_samples > 0:
        output_folder_name = f"{output_folder_name}/{num_samples}_samples"
    else:
        output_folder_name = f"{output_folder_name}/all_samples"
    save_path = os.path.join(save_path, output_folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    elif os.listdir(save_path):
        print("CFs for setting", output_folder_name, "already computed")
        return

    # ── black box + DiCE encoder (func="ohe-min-max"), exactly as dice/sample.py ─
    X_num_cat_ohe, _, _, _, _ = preprocess(
        data_dir, task_type=info["task_type"],
        cat_encoding="one-hot", num_encoding="minmax")
    input_shape = X_num_cat_ohe[0].shape[-1]

    model_filename = f"{data_dir}/black_box_mlp.pkl"

    black_box = BBMLPCLF(input_shape).to(device)
    black_box.load_state_dict(torch.load(model_filename))
    for p in black_box.parameters():
        p.requires_grad = False
    black_box.eval()
    print("Loaded black box model from", model_filename)

    train_df = pd.read_csv(f"data/{dataname}/train.csv")
    test_df = pd.read_csv(f"data/{dataname}/test.csv")

    train_df_dice = train_df.copy()
    train_df_dice[target_col] = train_df_dice[target_col].map(
        {target_class: 1, negative_class: 0})
    data_dice = dice_ml.data.Data(dataframe=train_df_dice, outcome_name=target_col,
                                  continuous_features=num_cols)
    dice_model = dice_ml.Model(black_box, backend="PYT", model_type="classifier",
                               func="ohe-min-max", device=device)
    # Constructing the explainer wires the ohe-min-max transformer onto dice_model,
    # which `get_output` relies on. We use the model only as an encoder/predictor.
    dice_ml.Dice(data_dice, dice_model, method="gradient")

    # ── select negative-class test factuals ─────────────────────────────────────
    test_feats = test_df[feat_cols]
    test_preds = np.argmax(dice_model.get_output(test_feats), axis=1)
    idx_neg = np.where(test_preds != target_class_index)[0][:num_samples] \
        if num_samples > 0 else np.where(test_preds != target_class_index)[0]
    factuals = test_df.iloc[idx_neg].reset_index(drop=True)

    # ── black-box labels for the (subsampled) training set ──────────────────────
    rng = np.random.default_rng(42)
    train_preds = np.argmax(dice_model.get_output(train_df[feat_cols]), axis=1)
    idx0 = np.where(train_preds == 0)[0]
    idx1 = np.where(train_preds == 1)[0]
    if len(idx0) > MAX_TRAIN_SIZE:
        idx0 = rng.choice(idx0, MAX_TRAIN_SIZE, replace=False)
    if len(idx1) > MAX_TRAIN_SIZE:
        idx1 = rng.choice(idx1, MAX_TRAIN_SIZE, replace=False)
    idx_sub = np.concatenate([idx0, idx1])
    train_sub = train_df.iloc[idx_sub].reset_index(drop=True)
    y_sub = train_preds[idx_sub].astype(int)

    # ── DiCoFlex preprocessing ──────────────────────────────────────────────────
    prep = _DiCoFlexPreprocessor(num_cols, cat_cols)
    X_all_prep = prep.fit_transform(train_sub)
    X_fact_prep = prep.transform(factuals)
    d_feat = X_all_prep.shape[1]

    bb = _BlackBoxAdapter(prep, dice_model, num_cols, cat_cols, feat_cols)
    dataset_cf = MulticlassCounterfactualWrapper(
        X=X_all_prep, y=y_sub, factual_classes=[0], p_values=list(P_TRAIN),
        masks=np.array([np.ones(d_feat) * 1e-3]), n_nearest=N_NEAREST,
        noise_level=0.01,
        classifier=bb if PROB_THRESHOLD > 0 else None, prob_threshold=PROB_THRESHOLD,
        log_level="WARNING", numerical_pos=prep.d_cont)

    # ── train flow ──────────────────────────────────────────────────────────────
    print(f"Training DiCoFlex flow ({NUM_EPOCHS} epochs) ...")
    start_time = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        flow_model = train_multiclass_counterfactual_flow_model(
            dataset=dataset_cf, flow_model_class=MaskedAutoregressiveFlow,
            mask_features=1, hidden_features=HIDDEN_FEATURES, num_layers=NUM_LAYERS,
            num_blocks_per_layer=NUM_BLOCKS_PER_LAYER, learning_rate=LEARNING_RATE,
            batch_size=256, num_epochs=NUM_EPOCHS, patience=PATIENCE,
            noise_level=0.01, device=device, save_dir=tmp_dir,
            log_interval=NUM_EPOCHS + 1, balanced=True, load_from_save_dir=False)

    # ── generate CFs (closest of N candidates per factual) ──────────────────────
    print(f"Generating CFs for {len(factuals)} factuals ...")
    all_cfs, _ = generate_multiclass_counterfactuals(
        model=flow_model, factual_points=X_fact_prep, target_class=1,
        p_value=P_INFERENCE, mask=np.array([1.0]), n_samples=N_GEN_SAMPLES,
        temperature=TEMPERATURE, device=device,
        num_classes=len(np.unique(y_sub)))
    n_fact, n_samp, d_dim = all_cfs.shape
    dists = np.linalg.norm(all_cfs - X_fact_prep[:, None, :], axis=2)
    dists[np.isnan(all_cfs).any(axis=2)] = np.inf        # unusable candidates

    if SELECT == "valid":
        probs = bb.predict_proba(np.nan_to_num(all_cfs.reshape(n_fact * n_samp, d_dim)))
        p_tgt = np.asarray(probs)[:, target_class_index].reshape(n_fact, n_samp)
        valid = (p_tgt > 0.5) & np.isfinite(dists)
        masked = np.where(valid, dists, np.inf)
        pick = masked.argmin(axis=1)
        none_valid = ~valid.any(axis=1)
        pick[none_valid] = dists[none_valid].argmin(axis=1)   # fall back, keep coverage
        print(f"  selection: {int((~none_valid).sum())}/{n_fact} factuals had >=1 valid "
              f"candidate of {n_samp} (mean valid per factual "
              f"{valid.sum(axis=1).mean():.1f})")
    else:
        pick = dists.argmin(axis=1)

    cf_prep = all_cfs[np.arange(n_fact), pick]

    cf_cont, cf_cat = prep.inverse_transform(cf_prep)

    # ── assemble CFs in original feature space ──────────────────────────────────
    cf_df = pd.DataFrame(index=range(len(factuals)))
    for j, c in enumerate(num_cols):
        cf_df[c] = cf_cont[:, j]
    for j, c in enumerate(cat_cols):
        cf_df[c] = cf_cat[:, j]
    cf_df = cf_df[feat_cols]

    target_probs = dice_model.get_output(cf_df)[:, target_class_index]
    cf_df["id"] = list(range(len(factuals)))
    cf_df["target_prob"] = target_probs

    cols = ["id"] + feat_cols
    cf_df[cols + ["target_prob"]].to_csv(
        os.path.join(save_path, "cfs.csv"), index=False)

    factuals_out = factuals.copy()
    factuals_out["id"] = list(range(len(factuals)))
    factuals_out[cols].to_csv(
        os.path.join(save_path, "original_test_samples.csv"), index=False)

    print(f"Time: {time.time() - start_time:.1f}s")
    print(f"Saving sampled data to {save_path}")
