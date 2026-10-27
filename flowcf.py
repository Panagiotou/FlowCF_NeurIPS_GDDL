"""FlowCF — sparse counterfactual explanations via gated mixed flow matching.

Reference implementation of the method described in Section 3 of the paper. The
three training steps of Section 3 map onto the three functions below:

    Step 1 (Sec. 3.1, Transport) -> train_transport
    Step 2 (Sec. 3.2, Sparsity)  -> train_gate
    Step 3 (Sec. 3.3, Joint)     -> train_joint

and inference (Eq. 3 with a binarised gate) onto `generate`.

The module is deliberately black-box agnostic: it never sees the classifier f. It
takes the raw feature matrix, a `col_spec` describing each column, and the binary
labels f assigns to the training rows; the surrogate S of Eq. 2 is fitted on those
labels alone. baselines/flowcf/sample.py wires it to the benchmark harness.

col_spec is a list, one entry per column of the raw array, in column order:
    {"name": str, "type": "continuous"}
    {"name": str, "type": "categorical", "n_cats": int}   # values are integer codes
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Encoding ──────────────────────────────────────────────────────────────────

def _col_indices(col_spec: list) -> tuple[list[int], list[tuple[int, int]]]:
    cont_idx = [i for i, c in enumerate(col_spec) if c["type"] == "continuous"]
    cat_specs = [(i, c["n_cats"]) for i, c in enumerate(col_spec) if c["type"] == "categorical"]
    return cont_idx, cat_specs


def encode_features(X_raw: np.ndarray, col_spec: list):
    """Split a raw array into (continuous floats, categorical one-hots, category ints)."""
    cont_idx, cat_specs = _col_indices(col_spec)

    X_cont = (X_raw[:, cont_idx].astype(np.float32) if cont_idx
              else np.zeros((len(X_raw), 0), np.float32))

    oh_parts, int_parts = [], []
    for col_i, K in cat_specs:
        cats = X_raw[:, col_i].astype(int)
        oh_parts.append(np.eye(K, dtype=np.float32)[cats])
        int_parts.append(cats)

    X_cat_oh = np.concatenate(oh_parts, axis=1) if oh_parts else np.zeros((len(X_raw), 0), np.float32)
    X_cat_int = np.stack(int_parts, axis=1) if int_parts else np.zeros((len(X_raw), 0), np.int64)
    return X_cont, X_cat_oh, X_cat_int


def continuous_scaler(X_raw: np.ndarray, col_spec: list) -> tuple[torch.Tensor, torch.Tensor]:
    """(mu, sd) over the continuous columns; the surrogate sees standardised inputs."""
    Xc, _, _ = encode_features(X_raw, col_spec)
    if Xc.shape[1] == 0:
        z = torch.zeros(0, device=DEVICE)
        return z, z
    return (torch.tensor(Xc.mean(axis=0), device=DEVICE),
            torch.tensor(Xc.std(axis=0) + 1e-8, device=DEVICE))


def build_immutable_mask(col_spec: list, immutable_names: list) -> np.ndarray:
    """Boolean (n_cols,) over the gate's column order [continuous | categorical];
    True = immutable, i.e. g_c == 0 in training and at inference (Sec. 3.2)."""
    cont_idx, cat_specs = _col_indices(col_spec)
    order = ([col_spec[i]["name"] for i in cont_idx] +
             [col_spec[i]["name"] for i, _ in cat_specs])
    want = {n.strip() for n in immutable_names if n.strip()}
    unknown = want - set(order)
    if unknown:
        raise ValueError(f"unknown immutable feature(s) {sorted(unknown)}; available: {order}")
    return np.array([name in want for name in order], dtype=bool)


# ── Models ────────────────────────────────────────────────────────────────────

class MixedFlowModel(nn.Module):
    """The mixed flow operator F_theta of Eq. 1.

    Input  [x_cont | x_cat_onehot | t]
    Output v_theta (numerical velocity) and per-column categorical endpoint logits.
    """

    def __init__(self, col_spec: list, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        cont_idx, cat_specs = _col_indices(col_spec)
        self.d_cont = len(cont_idx)
        self.cat_dims = [K for _, K in cat_specs]
        in_dim = self.d_cont + sum(self.cat_dims) + 1  # +1 for time

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)
        self.cont_head = nn.Linear(hidden_dim, self.d_cont) if self.d_cont > 0 else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden_dim, K) for K in self.cat_dims])

    def forward(self, x_cont, x_cat_oh, t):
        parts = [p for p in [x_cont, x_cat_oh, t] if p.shape[1] > 0]
        h = self.backbone(torch.cat(parts, dim=1))
        v_cont = self.cont_head(h) if self.cont_head is not None else None
        return v_cont, [head(h) for head in self.cat_heads]


class Surrogate(nn.Module):
    """The differentiable surrogate S of Eq. 2.

    Trained on [std(cont) | cat_onehot] against the black box's BINARY predictions
    only — never its probabilities, logits or gradients — then frozen, so gradients
    can flow through it into the flow and the gate.
    """

    def __init__(self, in_dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class GateNet(nn.Module):
    """The gate network a_psi of Eq. 3: one logit per continuous feature and per
    categorical COLUMN (not per one-hot dummy). g = sigmoid(a_psi(x) / T)."""

    def __init__(self, d_cont: int, d_cat_oh: int, n_cols: int,
                 hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(d_cont + d_cat_oh, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, n_cols))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat_oh):
        parts = [p for p in [x_cont, x_cat_oh] if p.shape[1] > 0]
        return self.net(torch.cat(parts, dim=1))


# ── Loss terms ────────────────────────────────────────────────────────────────

def _surrogate_input(cf_cont, cf_cat_soft, cont_mu, cont_sd):
    parts = []
    if cf_cont.shape[1] > 0:
        parts.append((cf_cont - cont_mu) / cont_sd)
    if cf_cat_soft.shape[1] > 0:
        parts.append(cf_cat_soft)
    return torch.cat(parts, dim=1)


def _validity_loss(prob, tau_logit):
    """L_val of Eq. 2: hinge on the surrogate logit, flat once the CF is past the
    target margin, so nothing pushes it deeper into the target class."""
    lg = torch.log(prob + 1e-8) - torch.log(1 - prob + 1e-8)
    return F.softplus(tau_logit - lg).mean()


def _sparsity_loss(gate, k_target: float = 1.0):
    """L_sparse of Eq. 5: charge for every column the gate leaves open. The
    relu(.-k) floor exempts the single column a valid CF must always change."""
    return F.relu(gate.sum(1) - k_target).mean()


def _proximity_loss(gcont, v, gcat, probs, orig_oh, cont_sd):
    """L_prox of Eq. 5: the displacement scale, charged identically for both types.

    continuous  |g . v| in STANDARDISED units, so a wide-range column cannot
                dominate the penalty by its scale alone.
    categorical g . (probability mass the predicted endpoint places outside the
                original category), in [0, 1].
    """
    total = None
    if v is not None and gcont.shape[1] > 0:
        total = ((gcont * v) / cont_sd).abs().sum(1)
    for j in range(len(probs)):
        p_orig = (probs[j] * orig_oh[j]).sum(1)
        term = gcat[:, j] * (1.0 - p_orig)
        total = term if total is None else total + term
    return total.mean() if total is not None else torch.zeros((), device=DEVICE)


# ── Pairing (Sec. 3.1) ────────────────────────────────────────────────────────

def build_knn_pairs(X0_raw: np.ndarray, X1_raw: np.ndarray, col_spec: list,
                    K: int = 10, chunk: int = 2000):
    """Pair each factual instance with its K nearest target-class neighbours, using
    L1 distance on the numerical block and Hamming distance on the categorical one.

    X0 is processed in chunks so the O(N0 x N1) distance matrix is never built.
    """
    cont_idx, cat_specs = _col_indices(col_spec)
    cat_idx = [i for i, _ in cat_specs]
    n_cont, n_cat = len(cont_idx), len(cat_idx)

    X0c = X0_raw[:, cont_idx].astype(np.float32)
    X1c = X1_raw[:, cont_idx].astype(np.float32)
    X0k = X0_raw[:, cat_idx].astype(np.int32) if cat_idx else None
    X1k = X1_raw[:, cat_idx].astype(np.int32) if cat_idx else None

    all_i, all_j = [], []
    for start in tqdm(range(0, len(X0_raw), chunk), desc="KNN pairing", ncols=80):
        end = min(start + chunk, len(X0_raw))
        dist = np.zeros((end - start, len(X1_raw)), dtype=np.float32)
        if n_cont > 0:
            dist += np.abs(X0c[start:end, None, :] - X1c[None, :, :]).sum(axis=2) / n_cont
        if n_cat > 0:
            dist += (X0k[start:end, None, :] != X1k[None, :, :]).sum(axis=2) / n_cat
        top_k = np.argpartition(dist, K, axis=1)[:, :K]
        all_i.append(np.repeat(np.arange(start, end), K))
        all_j.append(top_k.ravel())

    return X0_raw[np.concatenate(all_i)], X1_raw[np.concatenate(all_j)]


# ── Surrogate ─────────────────────────────────────────────────────────────────

def train_surrogate(X0_raw, X1_raw, col_spec, cont_mu, cont_sd,
                    n_epochs: int = 3000, batch_size: int = 512, lr: float = 1e-3,
                    hidden: int = 256, n_layers: int = 4) -> Surrogate:
    """Fit S on the black box's binary labels, then freeze it."""
    X = np.vstack([X0_raw, X1_raw])
    y = np.hstack([np.zeros(len(X0_raw)), np.ones(len(X1_raw))]).astype(np.float32)
    Xc, Xk, _ = encode_features(X, col_spec)
    xc = ((torch.tensor(Xc, device=DEVICE) - cont_mu) / cont_sd
          if Xc.shape[1] else torch.zeros(len(X), 0, device=DEVICE))
    Xin = torch.cat([xc, torch.tensor(Xk, dtype=torch.float32, device=DEVICE)], dim=1)
    yt = torch.tensor(y, device=DEVICE)

    S = Surrogate(Xin.shape[1], hidden=hidden, n_layers=n_layers).to(DEVICE)
    opt = torch.optim.Adam(S.parameters(), lr=lr)
    for _ in tqdm(range(n_epochs), desc="Surrogate", ncols=80):
        S.train()
        i = torch.randperm(len(Xin), device=DEVICE)[:batch_size]
        loss = F.binary_cross_entropy(S(Xin[i]), yt[i])
        opt.zero_grad(); loss.backward(); opt.step()

    S.eval()
    for p in S.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        acc = ((S(Xin) > 0.5).long().cpu().numpy() == y.astype(int)).mean()
    print(f"  surrogate agreement with the black box: {acc:.4f}")
    return S


# ── Step 1: Transport (Sec. 3.1) ──────────────────────────────────────────────

def train_transport(model: MixedFlowModel, X0_pairs, X1_pairs, col_spec,
                    surrogate: Surrogate, cont_mu, cont_sd,
                    n_epochs: int = 3000, batch_size: int = 512, lr: float = 1e-3,
                    lambda_cat: float = 1.0, lambda_val: float = 2.0,
                    tau_conf: float = 0.85) -> None:
    """Train F_theta with L_flow of Eq. 2.

    Squared error on the numerical velocity along the straight interpolant, plus
    cross-entropy on the categorical endpoint, plus the validity hinge evaluated on
    the one-step endpoint x + v(x, 0).
    """
    X0c, X0k_oh, _ = encode_features(X0_pairs, col_spec)
    X1c, X1k_oh, X1k_int = encode_features(X1_pairs, col_spec)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    x0c_t, x0k_t, x1c_t, x1k_t = to_t(X0c), to_t(X0k_oh), to_t(X1c), to_t(X1k_oh)
    x1k_int_t = torch.tensor(X1k_int, dtype=torch.long, device=DEVICE)
    tau_logit = math.log(tau_conf / (1.0 - tau_conf))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in tqdm(range(n_epochs), desc="Step 1: transport", ncols=80):
        model.train()
        idx = torch.randperm(len(x0c_t), device=DEVICE)[:batch_size]
        b0c, b0k = x0c_t[idx], x0k_t[idx]
        b1c, b1k = x1c_t[idx], x1k_t[idx]
        b1k_int = x1k_int_t[idx]

        t = torch.rand(len(idx), 1, device=DEVICE)
        v_cont, logits_cat = model((1 - t) * b0c + t * b1c, (1 - t) * b0k + t * b1k, t)

        loss = torch.zeros(1, device=DEVICE)
        if v_cont is not None:
            loss = loss + ((v_cont - (b1c - b0c)) ** 2).sum(dim=1).mean()
        for j, logit in enumerate(logits_cat):
            loss = loss + lambda_cat * F.cross_entropy(logit, b1k_int[:, j])

        # Validity of the one-step endpoint, through the frozen surrogate.
        v0, logits0 = model(b0c, b0k, torch.zeros(len(idx), 1, device=DEVICE))
        cf_cont = (b0c + v0) if v0 is not None else b0c
        cf_soft = (torch.cat([F.softmax(l, dim=1) for l in logits0], dim=1)
                   if logits0 else torch.zeros(len(idx), 0, device=DEVICE))
        prob = surrogate(_surrogate_input(cf_cont, cf_soft, cont_mu, cont_sd))
        loss = loss + lambda_val * _validity_loss(prob, tau_logit)

        opt.zero_grad(); loss.backward(); opt.step()


# ── Step 2: Sparsity (Sec. 3.2) ───────────────────────────────────────────────

def train_gate(model: MixedFlowModel, surrogate: Surrogate, X0_raw, col_spec,
               cont_mu, cont_sd, lambda_sparse: float = 0.3, lambda_prox: float = 0.375,
               tau_conf: float = 0.9999, n_epochs: int = 2500, batch_size: int = 512,
               lr: float = 1e-3, temp0: float = 2.0, temp1: float = 0.2,
               immutable_mask=None) -> GateNet:
    """Train G_psi with L_gate of Eq. 4, the transport F_theta FROZEN.

    The gate temperature is annealed temp0 -> temp1 (T of Eq. 3); temp1 is stamped
    on the gate as the inference temperature so training and inference agree.
    """
    X0c, X0k_oh, _ = encode_features(X0_raw, col_spec)
    x0c = torch.tensor(X0c, device=DEVICE)
    x0k = torch.tensor(X0k_oh, device=DEVICE)
    with torch.no_grad():
        v_full, logits = model(x0c, x0k, torch.zeros(len(x0c), 1, device=DEVICE))
        v_full = v_full.detach() if v_full is not None else None
        probs = [F.softmax(l, dim=1).detach() for l in logits]

    n_cont = X0c.shape[1]
    cat_dims = [p.shape[1] for p in probs]
    n_catcols = len(cat_dims)
    offs = np.cumsum([0] + cat_dims)
    orig_oh = [x0k[:, offs[j]:offs[j + 1]] for j in range(n_catcols)]
    tau_logit = math.log(tau_conf / (1.0 - tau_conf))
    keep = (torch.tensor(1.0 - np.asarray(immutable_mask, np.float32), device=DEVICE)
            if immutable_mask is not None else None)

    g = GateNet(n_cont, x0k.shape[1], n_cont + n_catcols).to(DEVICE)
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    for e in tqdm(range(n_epochs), desc="Step 2: sparsity gate", ncols=80):
        temp = temp0 + (temp1 - temp0) * (e / max(1, n_epochs - 1))
        i = torch.randperm(len(x0c), device=DEVICE)[:batch_size]
        gate = torch.sigmoid(g(x0c[i], x0k[i]) / temp)
        if keep is not None:
            gate = gate * keep
        gcont, gcat = gate[:, :n_cont], gate[:, n_cont:]

        cf_cont = (x0c[i] + gcont * v_full[i]) if n_cont > 0 else x0c[i]
        parts = [gcat[:, j:j + 1] * probs[j][i] + (1 - gcat[:, j:j + 1]) * orig_oh[j][i]
                 for j in range(n_catcols)]
        cf_soft = torch.cat(parts, 1) if parts else torch.zeros(len(i), 0, device=DEVICE)
        prob = surrogate(_surrogate_input(cf_cont, cf_soft, cont_mu, cont_sd))

        loss = (_validity_loss(prob, tau_logit)
                + lambda_sparse * _sparsity_loss(gate)
                + lambda_prox * _proximity_loss(gcont, v_full[i] if n_cont > 0 else None,
                                                gcat, [p[i] for p in probs],
                                                [o[i] for o in orig_oh], cont_sd))
        opt.zero_grad(); loss.backward(); opt.step()

    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    g._infer_temp = temp1
    g._immutable = immutable_mask
    return g


# ── Step 3: Joint (Sec. 3.3) ──────────────────────────────────────────────────

def train_joint(model: MixedFlowModel, gate: GateNet, surrogate: Surrogate, X0_raw,
                col_spec, cont_mu, cont_sd, lambda_prox: float = 0.375,
                lambda_drift: float = 0.05, tau_conf: float = 0.9999,
                n_epochs: int = 1500, batch_size: int = 512, lr: float = 5e-4,
                temp0: float = 2.0, temp1: float = 0.2, flow_ramp: float = 0.3,
                immutable_mask=None) -> MixedFlowModel:
    """Re-fit F_theta through the frozen gate with L_joint of Eq. 6.

    The drift term anchors the flow to its Step 1 solution; the learning rate is
    ramped from 0 over the first `flow_ramp` fraction of training. Returns the
    co-tuned transport (the gate is unchanged and is reused as given).
    """
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad_(True)
    step1 = copy.deepcopy(model).eval()
    for p in step1.parameters():
        p.requires_grad_(False)

    X0c, X0k_oh, _ = encode_features(X0_raw, col_spec)
    x0c = torch.tensor(X0c, device=DEVICE)
    x0k = torch.tensor(X0k_oh, device=DEVICE)
    N, n_cont = len(x0c), X0c.shape[1]
    with torch.no_grad():
        v_ref, logits_ref = step1(x0c, x0k, torch.zeros(N, 1, device=DEVICE))
        v_ref = v_ref.detach() if v_ref is not None else None
        logits_ref = [l.detach() for l in logits_ref]

    cat_dims = [l.shape[1] for l in logits_ref]
    n_catcols = len(cat_dims)
    offs = np.cumsum([0] + cat_dims)
    orig_oh = [x0k[:, offs[j]:offs[j + 1]] for j in range(n_catcols)]
    tau_logit = math.log(tau_conf / (1 - tau_conf))
    keep = (torch.tensor(1.0 - np.asarray(immutable_mask, np.float32), device=DEVICE)
            if immutable_mask is not None else None)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for e in tqdm(range(n_epochs), desc="Step 3: joint", ncols=80):
        frac = e / max(1, n_epochs - 1)
        temp = temp0 + (temp1 - temp0) * frac
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, frac / flow_ramp)

        i = torch.randperm(N, device=DEVICE)[:batch_size]
        v, logits = model(x0c[i], x0k[i], torch.zeros(len(i), 1, device=DEVICE))
        probs = [F.softmax(l, 1) for l in logits]
        gate_i = torch.sigmoid(gate(x0c[i], x0k[i]) / temp)
        if keep is not None:
            gate_i = gate_i * keep
        gcont, gcat = gate_i[:, :n_cont], gate_i[:, n_cont:]

        cf_cont = (x0c[i] + gcont * v) if n_cont > 0 else x0c[i]
        parts = [gcat[:, j:j + 1] * probs[j] + (1 - gcat[:, j:j + 1]) * orig_oh[j][i]
                 for j in range(n_catcols)]
        cf_soft = torch.cat(parts, 1) if parts else torch.zeros(len(i), 0, device=DEVICE)
        prob = surrogate(_surrogate_input(cf_cont, cf_soft, cont_mu, cont_sd))

        drift = torch.zeros((), device=DEVICE)
        if v is not None:
            drift = drift + ((v - v_ref[i]) ** 2).sum(1).mean()
        for j in range(n_catcols):
            drift = drift + ((logits[j] - logits_ref[j][i]) ** 2).sum(1).mean()

        loss = (_validity_loss(prob, tau_logit)
                + lambda_prox * _proximity_loss(gcont, v, gcat, probs,
                                                [o[i] for o in orig_oh], cont_sd)
                + lambda_drift * drift)
        opt.zero_grad(); loss.backward(); opt.step()

    return model.eval()


# ── Inference (Sec. 3.3) ──────────────────────────────────────────────────────

@torch.no_grad()
def generate(model: MixedFlowModel, gate: GateNet, X0_raw, col_spec,
             temp: float = 0.2, immutable_mask=None):
    """Explain instances with one forward pass of each network (Eq. 3, gate binarised).

        continuous, open   -> x + v ;   closed -> x exactly
        categorical, open  -> argmax of the predicted endpoint ;  closed -> original

    Returns (cf_continuous, cf_category_codes). The black box is never called.
    """
    model.eval()
    X0c, X0k_oh, X0k_int = encode_features(X0_raw, col_spec)
    x0c = torch.tensor(X0c, dtype=torch.float32, device=DEVICE)
    x0k = torch.tensor(X0k_oh, dtype=torch.float32, device=DEVICE)

    v_cont, logits = model(x0c, x0k, torch.zeros(len(x0c), 1, device=DEVICE))
    mask = (torch.sigmoid(gate(x0c, x0k) / temp) > 0.5).float()
    imm = immutable_mask if immutable_mask is not None else getattr(gate, "_immutable", None)
    if imm is not None:
        mask = mask * torch.tensor(1.0 - np.asarray(imm, np.float32), device=DEVICE)

    n_cont = X0c.shape[1]
    mcont, mcat = mask[:, :n_cont], mask[:, n_cont:]
    cf_cont = (x0c + mcont * v_cont).cpu().numpy() if v_cont is not None else X0c

    cf_cat_int = X0k_int.copy()
    for j, l in enumerate(logits):
        open_col = mcat[:, j].cpu().numpy() > 0
        cf_cat_int[open_col, j] = l.argmax(dim=1).cpu().numpy()[open_col]
    return cf_cont, cf_cat_int
