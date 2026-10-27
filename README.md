# FlowCF: Sparse Counterfactual Explanations for Mixed-Type Tabular Data using Flow Matching

Code to reproduce the tabular experiments of the paper: Tables 3 and 4, i.e. FlowCF
and the six baselines on the six datasets of Table 2, scored with the metrics of
Section 4.4. (The qualitative CelebA-HQ experiment of Section 5.2 is not included.)

## Setup

```bash
conda create -n flowcf python=3.10
conda activate flowcf
pip install -r requirements.txt
```

Install the two vendored baseline frameworks in editable mode:

```bash
pip install -e baselines/dice/DiCE-main     # DiCE
pip install -e baselines/CARLA              # Wachter, REVISE, CCHVAE
```

## Data

```bash
python download_dataset.py                  # all six datasets
python process_dataset.py                   # train/test splits + data/<name>/info.json
```

`data/Info/<name>.json` declares each dataset's columns, types and target;
`download_dataset.py` fetches the raw files and `process_dataset.py` writes
`data/<name>/{train.csv,test.csv,info.json}`. Five datasets download without
credentials. **Lending Club** comes from Kaggle, so it needs an API token in
`~/.kaggle/kaggle.json` and acceptance of the dataset's terms
(<https://www.kaggle.com/datasets/wordsforthewise/lending-club>); the other five
run without it.

Both scripts take an optional dataset name, e.g. `python process_dataset.py --dataname magic`.

## Running the experiments

Run every method on one dataset and print its row of Table 4:

```bash
python run_benchmark.py --dataname adult --num_samples 1000
```

This first trains the shared black box the methods explain (a single 1000-unit MLP,
Section 4.1) and TABCF's VAE, reusing them if they already exist, then for each
method runs `sample` (generate counterfactuals) followed by `evaluate` (score them).
Results land in `counterfactual_results/adult/<Method>/1000_samples/` with a
`summary_1000_samples.csv` alongside, and per-method logs in `logs/adult/`. Repeat
for `bank`, `default`, `lending-club`, `magic` and `htru2` to obtain Table 4; Table 3
averages those rows.

`--methods flowcf,dicoflex` restricts the run to a subset, and `--mode evaluate`
re-scores counterfactuals that already exist.

Individual steps can also be run on their own:

```bash
python main.py --dataname adult --method blackbox --mode train
python main.py --dataname adult --method vae      --mode train      # TABCF only
python main.py --dataname adult --method flowcf   --mode sample   --num_samples 1000
python main.py --dataname adult --method flowcf   --mode evaluate --num_samples 1000
```

with `--method` one of `flowcf`, `wachter`, `dice`, `revise`, `cchvae`, `tabcf`,
`dicoflex`.

## Code

| File | Contents |
| --- | --- |
| `flowcf.py` | **Our method.** The mixed flow operator, the gate, and the three training steps of Section 3. Self-contained and black-box agnostic. |
| `baselines/flowcf/sample.py` | Wires FlowCF to the benchmark: the hyperparameters of Section 4.1, in one configuration for all datasets. |
| `black_box.py` | The shared black box every method explains. |
| `evaluate.py` | The metrics of Section 4.4 (validity, sparsity per feature type, numerical proximity). |
| `main.py` | Single entry point: train / sample / evaluate. |
| `run_benchmark.py` | Runs every method on a dataset and collects the results table. |
|  `download_dataset.py`, `process_dataset.py`, `utils_train.py`, `data_encoding.py` | Dataset download, splitting and encoding. |
| `baselines/<name>/sample.py` | Adapters that run each baseline against the same black box and factuals and write counterfactuals in the same format. |

The method reads its input as a `col_spec` (one entry per column: continuous, or
categorical with a category count) plus the black box's binary predictions on the
training rows. It never queries the black box at explanation time, and never sees
its gradients, probabilities or parameters: `flowcf.train_surrogate` fits the
surrogate of Eq. 2 on those predicted labels alone.

The three steps map onto `flowcf.train_transport` (Section 3.1),
`flowcf.train_gate` (Section 3.2) and `flowcf.train_joint` (Section 3.3);
`flowcf.generate` is inference under Eq. 3 with the gate binarised.

## Hyperparameters

FlowCF uses one configuration for every dataset, with no per-dataset tuning
(Section 4.1). It is set in `baselines/flowcf/sample.py`:

| Paper | Code | Value |
| --- | --- | --- |
| d_S | `SURROGATE_HIDDEN` x `SURROGATE_LAYERS` | 256 x 4 |
| d_F | `FLOW_HIDDEN` x `FLOW_LAYERS` | 128 x 3 |
| K | `K_KNN` | 10 |
| lambda_cat | `LAMBDA_CAT` | 1.0 |
| lambda_val | `LAMBDA_VAL` | 2.0 |
| lambda_sparse | `LAMBDA_SPARSE` | 0.3 |
| lambda_prox | `LAMBDA_PROX` | 0.375 |
| lambda_drift | `LAMBDA_DRIFT` | 0.05 |
| T | `GATE_TEMP` | 0.2 |

## Third-party code

The baselines are the authors' reference implementations, vendored so the
experiments run as reported. Their bundled datasets, checkpoints and docs have been
removed; only source is included.

* `baselines/CARLA/` — CARLA [19], MIT licence,
  <https://github.com/carla-recourse/CARLA>. Provides Wachter, REVISE and CCHVAE.
  One patch: `carla/recourse_methods/autoencoder/models/vae.py` handles datasets
  with no categorical features (MAGIC, HTRU2).
* `baselines/dice/DiCE-main/` — DiCE 0.11 [16], MIT licence,
  <https://github.com/interpretml/DiCE>.
* `baselines/dicoflex/DiCoFlex/` — DiCoFlex [6],
  <https://github.com/ofurman/DiCoFlex>, imported unmodified.
* `baselines/tabcf/` — TABCF [17], from the implementation released with that
  paper. The benchmark harness here (`main.py`, `process_dataset.py`,
  `data_encoding.py`, `utils_train.py`) also derives from it, so that factual
  selection, black-box training and metric computation are identical across methods,
  as stated in Section 4.1.
