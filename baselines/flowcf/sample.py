"""FlowCF in the benchmark harness (`--method flowcf`).

Trains the three steps of Section 3 on the training split, explains the same
`--num_samples` factual instances every other method is given, and writes the
cfs.csv / original_test_samples.csv that evaluate.py reads.

The flow only ever sees the black box's binary predictions on the training rows;
the black box itself is called just to label those rows, to pick the factuals, and
to score the finished counterfactuals for the validity metric.

Hyperparameters below are the single configuration used for every dataset
(paper Section 4.1).
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder

import dice_ml

from black_box import BBMLPCLF
from utils_train import preprocess

from flowcf import (
    DEVICE,
    MixedFlowModel,
    build_immutable_mask,
    build_knn_pairs,
    continuous_scaler,
    encode_features,
    generate,
    train_gate,
    train_joint,
    train_surrogate,
    train_transport,
)

warnings.filterwarnings("ignore")

# ── Hyperparameters (paper Section 4.1; identical for all datasets) ────────────
SEED = 42
K_KNN = 10               # K target-class neighbours paired with each factual
FLOW_HIDDEN, FLOW_LAYERS = 128, 3          # d_F
SURROGATE_HIDDEN, SURROGATE_LAYERS = 256, 4  # d_S
LAMBDA_CAT = 1.0
LAMBDA_VAL = 2.0
LAMBDA_SPARSE = 0.3
LAMBDA_PROX = 0.375
LAMBDA_DRIFT = 0.05
GATE_TEMP = 0.2          # T of Eq. 3 (annealed from 2.0 to this value)
TAU_STEP1 = 0.85         # surrogate-confidence margin of the validity hinge
TAU_GATE = 0.9999        # ... in Steps 2 and 3, where the gate must defend validity
BATCH_SIZE, LR = 512, 1e-3
LR_FLOW_JOINT = 5e-4
EPOCHS_SURROGATE, EPOCHS_STEP1, EPOCHS_STEP2, EPOCHS_STEP3 = 3000, 3000, 2500, 1500
VAL_FRAC = 0.1           # train rows held out of the surrogate fit, to report its
                         # agreement with the black box (the test set is never used)


def _build_col_spec(feat_cols, num_cols, cat_cols, ord_enc):
    col_spec = []
    for c in feat_cols:
        if c in num_cols:
            col_spec.append({"name": c, "type": "continuous"})
        else:
            n_cats = len(ord_enc.categories_[cat_cols.index(c)])
            col_spec.append({"name": c, "type": "categorical", "n_cats": n_cats})
    return col_spec


def _encode_raw(df, feat_cols, num_cols, cat_cols, ord_enc):
    """Raw array in `feat_cols` order: continuous as floats, categorical as codes."""
    X = np.zeros((len(df), len(feat_cols)), dtype=float)
    cat_codes = ord_enc.transform(df[cat_cols].astype(str)) if cat_cols else None
    for j, c in enumerate(feat_cols):
        if c in num_cols:
            X[:, j] = df[c].astype(float).values
        else:
            X[:, j] = cat_codes[:, cat_cols.index(c)]
    return X


def main(args):
    dataname = args.dataname
    data_dir = f"data/{dataname}"
    device = args.device
    num_samples = args.num_samples

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    with open(os.path.join(data_dir, "info.json")) as f:
        info = json.load(f)

    target_col = info["target_col"]
    target_class, negative_class = info["target_class"], info["negative_class"]
    target_class_index = 1

    full_cols = list(info["column_names"])
    feat_cols = [c for c in full_cols if c != target_col]
    num_cols = [full_cols[i] for i in info["num_col_idx"]]
    cat_cols = [full_cols[i] for i in info["cat_col_idx"]]

    save_path = os.path.join(args.save_path, dataname, "FlowCF",
                             f"{num_samples}_samples" if num_samples > 0 else "all_samples")
    os.makedirs(save_path, exist_ok=True)

    # ── black box + the one-hot/min-max encoder it was trained with ────────────
    X_num_cat_ohe, _, _, _, _ = preprocess(
        data_dir, task_type=info["task_type"], cat_encoding="one-hot", num_encoding="minmax")
    black_box = BBMLPCLF(X_num_cat_ohe[0].shape[-1]).to(device)
    black_box.load_state_dict(torch.load(f"{data_dir}/black_box_mlp.pkl"))
    for p in black_box.parameters():
        p.requires_grad = False
    black_box.eval()

    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    train_df_dice = train_df.copy()
    train_df_dice[target_col] = train_df_dice[target_col].map({target_class: 1, negative_class: 0})
    data_dice = dice_ml.data.Data(dataframe=train_df_dice, outcome_name=target_col,
                                  continuous_features=num_cols)
    dice_model = dice_ml.Model(black_box, backend="PYT", model_type="classifier",
                               func="ohe-min-max", device=device)
    dice_ml.Dice(data_dice, dice_model, method="gradient")  # wires the encoder

    # ── raw arrays + the black box's binary labels ─────────────────────────────
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        ord_enc.fit(train_df[cat_cols].astype(str))
    col_spec = _build_col_spec(feat_cols, num_cols, cat_cols, ord_enc)

    immutable_mask = None
    if getattr(args, "immutable", False):
        imm = [c for c in info.get("immutable", []) if c in feat_cols]
        immutable_mask = build_immutable_mask(col_spec, imm)
        print(f"  immutable (never changed): {imm}")

    X_train = _encode_raw(train_df, feat_cols, num_cols, cat_cols, ord_enc)
    y_train = np.argmax(dice_model.get_output(train_df[feat_cols]), axis=1)

    test_preds = np.argmax(dice_model.get_output(test_df[feat_cols]), axis=1)
    idx_neg = np.where(test_preds != target_class_index)[0]
    if num_samples > 0:
        idx_neg = idx_neg[:num_samples]
    factuals = test_df.iloc[idx_neg].reset_index(drop=True)
    X0_test = _encode_raw(factuals, feat_cols, num_cols, cat_cols, ord_enc)

    X0_train, X1_train = X_train[y_train == 0], X_train[y_train == 1]
    print(f"  train class-0: {len(X0_train)}  class-1: {len(X1_train)}  factuals: {len(X0_test)}")

    # The surrogate is fitted on a 90% slice of train; the rest measures how well it
    # reproduces the black box. Only the surrogate holds rows out -- the flow and the
    # gate train on all of train.
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X_train))
    val_idx, fit_idx = perm[:int(round(VAL_FRAC * len(X_train)))], perm[int(round(VAL_FRAC * len(X_train))):]
    X0_fit = X_train[fit_idx][y_train[fit_idx] == 0]
    X1_fit = X_train[fit_idx][y_train[fit_idx] == 1]

    # ── the three training steps of Section 3 ──────────────────────────────────
    start = time.time()
    cont_mu, cont_sd = continuous_scaler(X_train, col_spec)

    surrogate = train_surrogate(
        X0_fit, X1_fit, col_spec, cont_mu, cont_sd, n_epochs=EPOCHS_SURROGATE,
        batch_size=BATCH_SIZE, lr=LR, hidden=SURROGATE_HIDDEN, n_layers=SURROGATE_LAYERS)

    Xc, Xk, _ = encode_features(X_train[val_idx], col_spec)
    xc = ((torch.tensor(Xc, device=DEVICE) - cont_mu) / cont_sd
          if Xc.shape[1] else torch.zeros(len(val_idx), 0, device=DEVICE))
    with torch.no_grad():
        held_out = (surrogate(torch.cat([xc, torch.tensor(Xk, device=DEVICE)], 1)) > 0.5)
    print(f"  surrogate agreement on held-out train rows: "
          f"{(held_out.long().cpu().numpy() == y_train[val_idx]).mean():.4f}")

    X0_pairs, X1_pairs = build_knn_pairs(X0_train, X1_train, col_spec, K=K_KNN)

    model = MixedFlowModel(col_spec, hidden_dim=FLOW_HIDDEN, n_layers=FLOW_LAYERS).to(DEVICE)
    train_transport(model, X0_pairs, X1_pairs, col_spec, surrogate, cont_mu, cont_sd,
                    n_epochs=EPOCHS_STEP1, batch_size=BATCH_SIZE, lr=LR,
                    lambda_cat=LAMBDA_CAT, lambda_val=LAMBDA_VAL, tau_conf=TAU_STEP1)

    gate = train_gate(model, surrogate, X0_train, col_spec, cont_mu, cont_sd,
                      lambda_sparse=LAMBDA_SPARSE, lambda_prox=LAMBDA_PROX,
                      tau_conf=TAU_GATE, n_epochs=EPOCHS_STEP2, batch_size=BATCH_SIZE,
                      lr=LR, temp1=GATE_TEMP, immutable_mask=immutable_mask)

    model = train_joint(model, gate, surrogate, X0_train, col_spec, cont_mu, cont_sd,
                        lambda_prox=LAMBDA_PROX, lambda_drift=LAMBDA_DRIFT,
                        tau_conf=TAU_GATE, n_epochs=EPOCHS_STEP3, batch_size=BATCH_SIZE,
                        lr=LR_FLOW_JOINT, temp1=GATE_TEMP, immutable_mask=immutable_mask)

    cf_cont, cf_int = generate(model, gate, X0_test, col_spec, temp=GATE_TEMP,
                               immutable_mask=immutable_mask)

    # ── decode back to the original feature space ──────────────────────────────
    cont_names = [c["name"] for c in col_spec if c["type"] == "continuous"]
    cat_names = [c["name"] for c in col_spec if c["type"] == "categorical"]
    cf_df = pd.DataFrame(index=range(len(factuals)))
    for j, c in enumerate(cont_names):
        cf_df[c] = cf_cont[:, j]
    if cat_names:
        cf_cat_str = ord_enc.inverse_transform(cf_int)
        for j, c in enumerate(cat_names):
            cf_df[c] = cf_cat_str[:, j]
    cf_df = cf_df[feat_cols]

    cf_df["id"] = list(range(len(factuals)))
    cf_df["target_prob"] = dice_model.get_output(cf_df[feat_cols])[:, target_class_index]

    cols = ["id"] + feat_cols
    cf_df[cols + ["target_prob"]].to_csv(os.path.join(save_path, "cfs.csv"), index=False)

    factuals_out = factuals.copy()
    factuals_out["id"] = list(range(len(factuals)))
    factuals_out[cols].to_csv(os.path.join(save_path, "original_test_samples.csv"), index=False)

    print(f"Time: {time.time() - start:.1f}s")
    print(f"Saved counterfactuals to {save_path}")
