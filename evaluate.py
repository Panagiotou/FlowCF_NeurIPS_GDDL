"""Evaluation of saved counterfactuals — the metrics of paper Section 4.4.

Every method writes cfs.csv + original_test_samples.csv into
counterfactual_results/<dataset>/<Method>/<n>_samples/, and this module scores them
identically, in the input space:

    Validity      share of explained instances whose CF flips the black box
    Sparsity Cat  fraction of categorical features changed
    Sparsity Num  fraction of numerical features changed
    Prox Num      mean absolute displacement over the standard-normalised
                  numerical features

Validity is computed over all explained instances; every other metric over the
valid counterfactuals only. The two continuous-only datasets (MAGIC, HTRU2) have
no categorical block, so Sparsity Cat is reported as NaN there.

Run with:  python main.py --dataname <name> --method <method> --mode evaluate
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# Result folder written by each method's sample.py.
METHOD_FOLDERS = {
    "flowcf": "FlowCF",
    "tabcf": "TABCF",
    "dicoflex": "DiCoFlex",
    "wachter": "Wachter",
    "dice": "DiCE",
    "revise": "Revise",
    "cchvae": "CCHVAE",
}


def results_dir(save_path, dataname, method, num_samples):
    leaf = f"{num_samples}_samples" if num_samples > 0 else "all_samples"
    return os.path.join(save_path, dataname, METHOD_FOLDERS[method], leaf)


def l0_fraction(a, b):
    return (a != b).sum() / len(a)


def main(args):
    save_path = results_dir(args.save_path, args.dataname, args.method, args.num_samples)
    data_dir = f"data/{args.dataname}"

    with open(f"{data_dir}/info.json") as f:
        info = json.load(f)

    column_names = [c for c in info["column_names"] if c != info["target_col"]]
    num_cols = [column_names[i] for i in info["num_col_idx"]]
    cat_cols = [column_names[i] for i in info["cat_col_idx"]]

    cfs = pd.read_csv(os.path.join(save_path, "cfs.csv"))[["id"] + column_names + ["target_prob"]]
    factuals = pd.read_csv(os.path.join(save_path, "original_test_samples.csv"))

    # Numerical features are standard-normalised on the training split, so the
    # proximity metric is comparable across datasets and features.
    scaler = StandardScaler()
    if num_cols:
        scaler.fit(pd.read_csv(f"{data_dir}/train.csv")[num_cols])

    rows = []
    for i in range(len(factuals)):
        factual = factuals.iloc[i:i + 1]
        valid = cfs[(cfs["id"] == factual["id"].values[0]) & (cfs["target_prob"] > 0.5)]
        if len(valid) == 0:
            continue

        x_cat = factual[cat_cols].to_numpy()[0] if cat_cols else np.zeros(0)
        if num_cols:
            x_num = scaler.transform(factual[num_cols])[0]
            cf_num_all = scaler.transform(valid[num_cols])

        for j in range(len(valid)):
            cf_cat = valid[cat_cols].to_numpy()[j] if cat_cols else np.zeros(0)
            rows.append([
                l0_fraction(cf_cat, x_cat) if cat_cols else np.nan,
                l0_fraction(cf_num_all[j], x_num) if num_cols else np.nan,
                np.absolute(cf_num_all[j] - x_num).mean() if num_cols else np.nan,
            ])

    up, down = "↑", "↓"
    means = np.mean(rows, axis=0) if rows else [np.nan] * 3  # no valid CF at all
    results = pd.DataFrame([{
        f"Validity {up}": len(rows) / (args.total_CFs * len(factuals)),
        f"Sparsity Cat {down}": means[0],
        f"Sparsity Num {down}": means[1],
        f"Prox Num {down}": means[2],
    }], index=[args.method])

    rel = os.path.relpath(save_path)
    print(save_path if rel.startswith("..") else rel)
    print(tabulate(results.round(4), headers="keys", tablefmt="pretty"))
    results.to_csv(os.path.join(save_path, "results.csv"))
