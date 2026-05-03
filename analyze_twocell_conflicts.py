#!/usr/bin/env python3
"""
Two-Cell Conflict Scenario analysis.

Implements two levels from:
  "Learning and Reconstructing Conflicts in O-RAN: A Graph Neural Network Approach"
  IEEE WCNC 2025

Level 1: Pearson correlation baseline → binarized adjacency matrix
Level 2: GNN-based graph structure detection (numpy-only implementation)
         - Time-series node embeddings via sliding window
         - Iterative message-passing adjacency refinement (GCN-style)
         - Gradient descent structure learning (NOTEARS-inspired)

Usage:
  python3 analyze_twocell_conflicts.py <csv_file> [options]
  python3 analyze_twocell_conflicts.py two-cell-conflict-log.csv --level 2 --output-dir ./tc_results
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# ─── Column definitions for TwoCellConflictScenario ──────────────────────────
#
# ICPs (controlled parameters, prefixed by cell)
PARAMS = [
    "c0_Param_TxPower", "c1_Param_TxPower",
    "c0_Param_RET", "c1_Param_RET",
    "c0_Param_CIO", "c1_Param_CIO",
]
# KPIs (observed metrics)
KPIS = [
    "c0_SINR", "c1_SINR",
    "c0_dlThroughput", "c1_dlThroughput",
    "c0_trafficLoad", "c1_trafficLoad",
    "c0_rbUtil", "c1_rbUtil",
]
FEATURES = PARAMS + KPIS

# Short display names for plots
SHORT_NAME = {
    "c0_Param_TxPower": "c0.TxP", "c1_Param_TxPower": "c1.TxP",
    "c0_Param_RET": "c0.RET", "c1_Param_RET": "c1.RET",
    "c0_Param_CIO": "c0.CIO", "c1_Param_CIO": "c1.CIO",
    "c0_SINR": "c0.SINR", "c1_SINR": "c1.SINR",
    "c0_dlThroughput": "c0.DLThp", "c1_dlThroughput": "c1.DLThp",
    "c0_trafficLoad": "c0.Load", "c1_trafficLoad": "c1.Load",
    "c0_rbUtil": "c0.RButil", "c1_rbUtil": "c1.RButil",
    "c0_AvgCqi": "c0.CQI", "c1_AvgCqi": "c1.CQI",
    "c0_RSRP": "c0.RSRP", "c1_RSRP": "c1.RSRP",
    "c0_RSRQ": "c0.RSRQ", "c1_RSRQ": "c1.RSRQ",
    "c0_hoAttempts": "c0.HOatt", "c1_hoAttempts": "c1.HOatt",
}

# ─── Ground-truth adjacency ───────────────────────────────────────────────────
#
# Known causal relationships from the current measurable KPI design:
#   CCO  adjusts TxPower  -> affects SINR and DL Throughput
#   MLB  adjusts CIO proxy -> affects trafficLoad and RB utilization
#
# Deferred until measured accurately:
#   ES   Energy Efficiency, Power Consumption
#   MRO  Handover Success Rate, Call Drop Rate, Call Block Rate
#
# xApp conflicts (ground truth):
#   Direct : CCO ↔ ES via TxPower
#   Indirect: CCO ↔ MLB (coverage/quality vs load balancing)
PARAM_KPI_GT: dict[str, list[str]] = {
    "c0_Param_TxPower":   ["c0_SINR", "c0_dlThroughput", "c0_trafficLoad", "c0_rbUtil"],
    "c1_Param_TxPower":   ["c1_SINR", "c1_dlThroughput", "c1_trafficLoad", "c1_rbUtil"],
    "c0_Param_RET":       ["c0_SINR", "c0_dlThroughput", "c0_trafficLoad", "c0_rbUtil"],
    "c1_Param_RET":       ["c1_SINR", "c1_dlThroughput", "c1_trafficLoad", "c1_rbUtil"],
    "c0_Param_CIO":       ["c0_trafficLoad", "c1_trafficLoad", "c0_rbUtil", "c1_rbUtil"],
    "c1_Param_CIO":       ["c0_trafficLoad", "c1_trafficLoad", "c0_rbUtil", "c1_rbUtil"],
}

# xApp -> ICPs ground truth (for conflict reporting)
XAPP_PARAMS_GT: dict[str, list[str]] = {
    "CCO": [
        "c0_Param_TxPower", "c1_Param_TxPower",
        "c0_Param_RET", "c1_Param_RET",
    ],
    "ES": ["c0_Param_TxPower", "c1_Param_TxPower"],
    "MLB": ["c0_Param_CIO", "c1_Param_CIO"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        # Try with subset of available features
        avail = [c for c in FEATURES if c in df.columns]
        print(f"  [warn] Missing columns: {missing}")
        print(f"  Using available: {avail}")
        if len(avail) < 4:
            raise ValueError("Too few feature columns found in CSV")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Deduplicate steps (one row per step), return (X, feature_list)."""
    avail = [f for f in FEATURES if f in df.columns]
    # Group by step and take mean (many action rows per step → aggregate)
    df_agg = df.groupby("step")[avail].mean().reset_index().sort_values("step")
    X = df_agg[avail].values.astype(float)
    return X, avail


def derive_xapp_param(df: pd.DataFrame) -> dict[str, list[str]]:
    """Infer xApp → ICP from CSV action rows."""
    xapp_param: dict[str, list[str]] = {}
    for xapp, group in df.groupby("xapp"):
        raw_params = group["adjusted_param"].dropna().unique().tolist()
        # Map delta columns back to ICP param names
        icps = []
        for rp in raw_params:
            if "_TxPower_delta" in rp:
                cell = rp.split("_")[0]
                icp = f"{cell}_Param_TxPower"
            elif "_RET_delta" in rp:
                cell = rp.split("_")[0]
                icp = f"{cell}_Param_RET"
            elif "_CIO_delta" in rp:
                cell = rp.split("_")[0]
                icp = f"{cell}_Param_CIO"
            else:
                icp = rp
            if icp not in icps:
                icps.append(icp)
        xapp_param[xapp] = sorted(icps)
    return xapp_param


def aggregate_by_step(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return one numeric row per simulation step."""
    avail = [c for c in columns if c in df.columns]
    return df.groupby("step")[avail].mean().reset_index().sort_values("step")


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(x.corr(y))


def safe_lag_corr(x: pd.Series, y: pd.Series, lag: int = 1) -> float:
    if lag <= 0:
        return safe_corr(x, y)
    if len(x) <= lag:
        return 0.0
    return safe_corr(x.iloc[:-lag].reset_index(drop=True),
                     y.iloc[lag:].reset_index(drop=True))


def default_value_for_kpi(name: str) -> float:
    if "RSRP" in name:
        return -100.0
    if "RSRQ" in name:
        return -15.0
    if "AvgCqi" in name:
        return 7.0
    return 0.0


def map_action_param_to_icp(adjusted_param: str) -> str | None:
    if not isinstance(adjusted_param, str):
        return None
    cell = adjusted_param.split("_")[0]
    if "_TxPower_delta" in adjusted_param:
        return f"{cell}_Param_TxPower"
    if "_RET_delta" in adjusted_param:
        return f"{cell}_Param_RET"
    if "_CIO_delta" in adjusted_param:
        return f"{cell}_Param_CIO"
    return None


def build_action_consistency(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Compare requested action deltas with actual clamped parameter deltas."""
    if "adjusted_param" not in df.columns or "adjusted_value" not in df.columns:
        return pd.DataFrame()

    rows = []
    action_df = df[["step", "xapp", "adjusted_param", "adjusted_value"]].dropna().copy()
    param_deltas = agg[["step"] + [p for p in PARAMS if p in agg.columns]].copy()
    for param in PARAMS:
        if param in param_deltas.columns:
            param_deltas[f"actual_{param}"] = param_deltas[param].diff().fillna(0.0)

    action_df["icp"] = action_df["adjusted_param"].map(map_action_param_to_icp)
    action_df = action_df[action_df["icp"].notna()]
    for _, action in action_df.iterrows():
        actual_col = f"actual_{action['icp']}"
        step = int(action["step"])
        actual_row = param_deltas[param_deltas["step"] == step + 1]
        if actual_row.empty or actual_col not in actual_row.columns:
            continue
        requested = float(action["adjusted_value"])
        actual = float(actual_row.iloc[0][actual_col])
        rows.append({
            "step": step,
            "xapp": action["xapp"],
            "adjusted_param": action["adjusted_param"],
            "icp": action["icp"],
            "requested_delta": requested,
            "actual_delta": actual,
            "clamped_or_noop": abs(requested - actual) > 1e-9,
        })
    return pd.DataFrame(rows)


def validation_report(df: pd.DataFrame,
                      features: list[str],
                      warmup_step: int,
                      output_dir: str) -> dict[str, object]:
    """Validate scenario/KPM sanity before interpreting GNN output."""
    agg = aggregate_by_step(df, features)
    warm = agg[agg["step"] >= warmup_step].copy()
    if warm.empty:
        warm = agg.copy()

    lines: list[str] = []
    lines.append("TwoCell scenario validation")
    lines.append("=" * 55)
    lines.append(f"rows={len(df)} steps={df['step'].nunique()} warmup_step={warmup_step}")
    lines.append(f"validation_steps={len(warm)}")

    lines.append("\nKPM default-like counts")
    default_summary = []
    for kpi in [k for k in KPIS if k in agg.columns]:
        default = default_value_for_kpi(kpi)
        total_count = int(np.isclose(agg[kpi], default).sum())
        warm_count = int(np.isclose(warm[kpi], default).sum())
        default_summary.append((kpi, default, total_count, warm_count))
        lines.append(f"  {SHORT_NAME.get(kpi, kpi):<12} default={default:>7g} "
                     f"all={total_count:>3} warm={warm_count:>3}")

    lines.append("\nKPM ranges after warm-up")
    inactive_kpis = []
    for kpi in [k for k in KPIS if k in warm.columns]:
        series = warm[kpi]
        if series.nunique(dropna=True) <= 1:
            inactive_kpis.append(kpi)
        lines.append(f"  {SHORT_NAME.get(kpi, kpi):<12} "
                     f"min={series.min():>9.4g} max={series.max():>9.4g} "
                     f"mean={series.mean():>9.4g} unique={series.nunique(dropna=True)}")

    lines.append("\nEmpty-cell-like steps after warm-up")
    empty_rows = []
    for cell in ["c0", "c1"]:
        needed = [f"{cell}_dlThroughput", f"{cell}_rbUtil"]
        if not all(c in warm.columns for c in needed):
            continue
        mask = (
            np.isclose(warm[f"{cell}_dlThroughput"], 0.0) |
            np.isclose(warm[f"{cell}_rbUtil"], 0.0)
        )
        steps = warm.loc[mask, "step"].astype(int).tolist()
        empty_rows.append((cell, steps))
        preview = ", ".join(map(str, steps[:20])) if steps else "none"
        if len(steps) > 20:
            preview += ", ..."
        lines.append(f"  {cell}: count={len(steps)} steps={preview}")

    consistency = build_action_consistency(df, agg)
    lines.append("\nAction delta consistency")
    mismatch_count = 0
    if consistency.empty:
        lines.append("  no action consistency data available")
    else:
        for icp, group in consistency.groupby("icp"):
            mismatches = group[group["clamped_or_noop"]]
            mismatch_count += len(mismatches)
            max_abs_err = (group["requested_delta"] - group["actual_delta"]).abs().max()
            lines.append(f"  {SHORT_NAME.get(icp, icp):<8} "
                         f"mismatch={len(mismatches):>3}/{len(group):<3} "
                         f"max_abs_err={max_abs_err:g}")
            for _, row in mismatches.head(5).iterrows():
                lines.append(f"    step={int(row['step']):>3} {row['adjusted_param']} "
                             f"requested={row['requested_delta']:g} actual={row['actual_delta']:g}")

    lines.append("\nExpected relationship correlations after warm-up")
    expected_pairs = [
        (src, dst)
        for src, dsts in PARAM_KPI_GT.items()
        for dst in dsts
    ]
    corr_summary = {}
    for src, dst in expected_pairs:
        if src not in warm.columns or dst not in warm.columns:
            continue
        pearson = safe_corr(warm[src], warm[dst])
        lag1 = safe_lag_corr(warm[src], warm[dst], lag=1)
        corr_summary[(src, dst)] = (pearson, lag1)
        lines.append(f"  {SHORT_NAME.get(src, src):<8} -> {SHORT_NAME.get(dst, dst):<10} "
                     f"pearson={pearson:>7.3f} lag1={lag1:>7.3f}")

    lines.append("\nInactive KPI warning")
    if inactive_kpis:
        for kpi in inactive_kpis:
            lines.append(f"  {SHORT_NAME.get(kpi, kpi)} is constant after warm-up")
    else:
        lines.append("  none")

    report_path = os.path.join(output_dir, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n── Scenario/KPM Validation ───────────────────────────────")
    print("\n".join(lines[:80]))
    if len(lines) > 80:
        print("  ...")
    print(f"  Saved: {report_path}")

    return {
        "path": report_path,
        "warmup_step": warmup_step,
        "inactive_kpis": inactive_kpis,
        "action_mismatch_count": mismatch_count,
        "empty_rows": empty_rows,
        "corr_summary": corr_summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — Pearson correlation baseline
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlation(X: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def binarize(corr: np.ndarray, threshold: float) -> np.ndarray:
    adj = (np.abs(corr) >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def build_ground_truth_adj(features: list[str]) -> np.ndarray:
    n = len(features)
    adj = np.zeros((n, n), dtype=int)
    for param, kpis in PARAM_KPI_GT.items():
        if param not in features:
            continue
        i = features.index(param)
        for kpi in kpis:
            if kpi not in features:
                continue
            j = features.index(kpi)
            adj[i, j] = 1
            adj[j, i] = 1
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — GNN-based conflict graph detection (numpy implementation)
#
# Based on: "Learning and Reconstructing Conflicts in O-RAN:
#            A Graph Neural Network Approach" (IEEE WCNC 2025)
#
# Architecture:
#   1. Temporal encoder: sliding-window statistics → node feature vectors
#   2. Edge encoder:    pairwise node features → initial edge logits (MLP)
#   3. GCN rounds:      iterative message passing to refine node embeddings
#   4. Structure learner: NOTEARS-inspired differentiable adjacency estimation
#      — minimizes reconstruction loss + DAG acyclicity penalty via gradient descent
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def temporal_node_features(X: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Encode each feature (node) as a vector of temporal statistics computed
    over a sliding window: [mean, std, trend, autocorr, range].

    Returns shape (N_nodes, 5) where N_nodes = X.shape[1].
    """
    T, N = X.shape
    feats = np.zeros((N, 5), dtype=float)

    # Standardize X per column for stable statistics
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma

    for j in range(N):
        col = Xn[:, j]
        # Mean of last `window` steps
        feats[j, 0] = col[-window:].mean()
        # Std of last `window` steps
        feats[j, 1] = col[-window:].std()
        # Linear trend (slope) over full series
        t = np.arange(T)
        feats[j, 2] = np.polyfit(t, col, 1)[0]
        # Lag-1 autocorrelation
        if T > 1:
            feats[j, 3] = np.corrcoef(col[:-1], col[1:])[0, 1]
        # Range (max - min) over last `window`
        feats[j, 4] = col[-window:].max() - col[-window:].min()

    return np.nan_to_num(feats, nan=0.0)


def edge_logits_from_features(node_feats: np.ndarray) -> np.ndarray:
    """
    Compute pairwise edge logits from node feature vectors.
    Uses a two-layer MLP applied to (fᵢ ⊕ fⱼ) for each pair (i,j).

    MLP weights are random but seeded — serves as a fixed encoder.
    In the full paper these would be learned end-to-end.
    """
    N, d = node_feats.shape
    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    # Layer 1: (2d) → hidden_dim
    hidden = 16
    W1 = rng.standard_normal((2 * d, hidden)) * np.sqrt(2.0 / (2 * d))
    b1 = np.zeros(hidden)
    # Layer 2: hidden_dim → 1
    W2 = rng.standard_normal((hidden, 1)) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(1)

    logits = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            pair = np.concatenate([node_feats[i], node_feats[j]])
            h = _relu(pair @ W1 + b1)
            logits[i, j] = (h @ W2 + b2)[0]

    return logits


def gcn_message_passing(X: np.ndarray, A_init: np.ndarray,
                         n_rounds: int = 2) -> np.ndarray:
    """
    GCN-style message passing to refine node embeddings.

    Each round: H' = σ(D⁻½ A D⁻½ H W)
    where A = current adjacency estimate, H = node feature matrix over time.

    Returns updated node features after `n_rounds` of message passing.
    """
    T, N = X.shape
    # Use time-series as node features: each node has embedding = its time series
    H = X.T  # (N, T)

    A = np.copy(A_init)
    np.fill_diagonal(A, 1.0)  # self-loops

    rng = np.random.default_rng(7)
    d_hidden = max(4, T // 4)

    for _ in range(n_rounds):
        # Normalized adjacency
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        A_norm = A / np.sqrt(deg) / np.sqrt(deg.T)

        # Aggregate neighbor features
        H_agg = A_norm @ H  # (N, T)

        # Linear projection + ReLU
        W = rng.standard_normal((T, d_hidden)) * np.sqrt(2.0 / T)
        H = _relu(H_agg @ W)  # (N, d_hidden)
        T = d_hidden  # next round uses new embedding dim

    return H  # (N, d_hidden)


def notears_structure_learning(X: np.ndarray,
                                n_iter: int = 500,
                                lr: float = 0.01,
                                lambda1: float = 0.1,
                                h_tol: float = 1e-6,
                                rho_init: float = 1.0) -> np.ndarray:
    """
    NOTEARS-inspired differentiable structure learning.

    Minimizes:  ||X - X·W||²_F + λ||W||₁  subject to  h(W) = 0
    where h(W) = tr(e^{W∘W}) - N  is the DAG acyclicity constraint.

    Uses augmented Lagrangian to handle the constraint.
    Returns thresholded |W| as detected adjacency matrix.

    Reference: Zheng et al., "DAGs with NO TEARS" (NeurIPS 2018).
    The GNN paper uses a learned version; this is the baseline variant.
    """
    T, N = X.shape

    # Standardize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sigma

    # Initialize W near zero
    rng = np.random.default_rng(0)
    W = rng.standard_normal((N, N)) * 0.01
    np.fill_diagonal(W, 0.0)

    alpha = 0.0   # Lagrange multiplier
    rho = rho_init
    h_prev = np.inf

    def h_func(W: np.ndarray) -> float:
        """Acyclicity constraint: tr(e^{W∘W}) - N"""
        M = np.exp(W * W)
        return float(np.trace(M)) - N

    def grad_h(W: np.ndarray) -> np.ndarray:
        """Gradient of h w.r.t. W"""
        M = np.exp(W * W)
        return 2.0 * W * M

    def loss_and_grad(W: np.ndarray) -> tuple[float, np.ndarray]:
        # Reconstruction loss
        res = Xn - Xn @ W
        rec_loss = 0.5 * np.sum(res ** 2) / T
        grad_rec = -Xn.T @ res / T

        # L1 penalty (proximal handled separately, use smooth approx here)
        l1_loss = lambda1 * np.sum(np.abs(W))
        grad_l1 = lambda1 * np.sign(W)

        # Acyclicity penalty
        h = h_func(W)
        h_loss = (alpha * h + 0.5 * rho * h ** 2)
        grad_h_val = grad_h(W)
        grad_h_total = (alpha + rho * h) * grad_h_val

        # Diagonal: no self-loops
        np.fill_diagonal(grad_rec, 0.0)
        np.fill_diagonal(grad_l1, 0.0)
        np.fill_diagonal(grad_h_total, 0.0)

        total_loss = rec_loss + l1_loss + h_loss
        total_grad = grad_rec + grad_l1 + grad_h_total
        return total_loss, total_grad

    # Augmented Lagrangian iterations
    for outer in range(5):
        # Inner gradient descent
        for it in range(n_iter):
            loss, grad = loss_and_grad(W)
            W -= lr * grad
            np.fill_diagonal(W, 0.0)

        h = h_func(W)
        if h <= h_tol:
            break

        # Update multipliers
        alpha += rho * h
        if h > 0.25 * h_prev:
            rho = min(rho * 10, 1e8)
        h_prev = h

    # Threshold absolute weights to produce binary adjacency
    W_abs = np.abs(W)
    np.fill_diagonal(W_abs, 0.0)
    threshold = np.percentile(W_abs[W_abs > 0], 70) if np.any(W_abs > 0) else 0.1
    adj = (W_abs >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj, W


def gnn_detect_conflict_graph(X: np.ndarray,
                               features: list[str],
                               n_iter: int = 500) -> np.ndarray:
    """
    Full GNN pipeline for conflict graph detection.

    Steps:
      1. Temporal encoding → node feature vectors
      2. Initial edge logit estimation (pairwise MLP)
      3. GCN message passing (2 rounds) with soft adjacency from logits
      4. NOTEARS structure learning on GCN-refined embeddings
      5. Combine correlation signal + structure learning for final adjacency
    """
    T, N = X.shape
    print(f"  [GNN] Temporal node encoding  ({N} nodes, {T} steps)")
    node_feats = temporal_node_features(X)

    print(f"  [GNN] Initial edge logit estimation")
    logits = edge_logits_from_features(node_feats)
    A_soft = _sigmoid(logits)
    np.fill_diagonal(A_soft, 0.0)

    print(f"  [GNN] GCN message passing (2 rounds)")
    H_refined = gcn_message_passing(X, A_soft, n_rounds=2)

    # Use refined node embeddings as input to structure learning
    # Build a (T × N) matrix from the GCN output for NOTEARS
    # Map back: H_refined is (N, d), use pairwise similarity as GNN-reconstructed X
    H_T = H_refined.T  # (d, N) → use as (T', N) where T' = d_hidden
    if H_T.shape[0] < 2:
        H_T = X  # fallback to original

    print(f"  [GNN] NOTEARS structure learning ({n_iter} iterations)")
    adj_notears, W = notears_structure_learning(H_T, n_iter=n_iter)

    # Combine: union of strong correlation signal + NOTEARS
    corr = compute_correlation(X)
    adj_corr = binarize(corr, threshold=0.7)

    # Only keep edges where BOTH methods agree (high precision) or either (high recall)
    # Use intersection for Param→KPI region (causal), union elsewhere
    N_params = sum(1 for f in features if f in PARAMS)
    adj_combined = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            is_param_i = features[i] in PARAMS
            is_kpi_j = features[j] in KPIS
            if is_param_i and is_kpi_j:
                # Param→KPI: require at least one method
                adj_combined[i, j] = int(adj_notears[i, j] or adj_corr[i, j])
            else:
                # Other edges: require both methods (reduce false positives)
                adj_combined[i, j] = int(adj_notears[i, j] and adj_corr[i, j])

    return adj_combined, adj_notears, W


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(pred_adj: np.ndarray, gt_adj: np.ndarray,
             features: list[str], label: str = ""):
    avail_params = [p for p in PARAMS if p in features]
    avail_kpis   = [k for k in KPIS   if k in features]
    param_idx = [features.index(p) for p in avail_params]
    kpi_idx   = [features.index(k) for k in avail_kpis]

    y_true, y_pred, edge_labels = [], [], []
    for i in param_idx:
        for j in kpi_idx:
            y_true.append(gt_adj[i, j])
            y_pred.append(pred_adj[i, j])
            sn = SHORT_NAME.get(features[i], features[i])
            sk = SHORT_NAME.get(features[j], features[j])
            edge_labels.append(f"{sn}→{sk}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tag = f"[{label}] " if label else ""
    print(f"\n── {tag}Param→KPI Edge Detection ─────────────────────────")
    print(f"  Precision : {p:.3f}")
    print(f"  Recall    : {r:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print()
    print(f"  {'Edge':<38} GT  Pred")
    for lbl, gt, pred in zip(edge_labels, y_true, y_pred):
        mark = "✓" if gt == pred else "✗"
        print(f"  {lbl:<38} {gt}   {pred}  {mark}")
    return {"precision": p, "recall": r, "f1": f1}


def report_conflicts(pred_adj: np.ndarray, features: list[str],
                     xapp_param: dict[str, list[str]], label: str = ""):
    tag = f"[{label}] " if label else ""
    print(f"\n── {tag}xApp→ICP Observed from data ──────────────────────")
    for xapp, params in sorted(xapp_param.items()):
        short_p = [SHORT_NAME.get(p, p) for p in params]
        print(f"  {xapp}: {short_p}")

    print(f"\n── {tag}Detected Conflicts ────────────────────────────────")

    # Direct conflicts: two xApps touching the same ICP
    param_to_xapps: dict[str, list[str]] = {}
    for xapp, params in xapp_param.items():
        for p in params:
            param_to_xapps.setdefault(p, [])
            if xapp not in param_to_xapps[p]:
                param_to_xapps[p].append(xapp)

    direct_found = [(xapps, p) for p, xapps in param_to_xapps.items() if len(xapps) > 1]
    if direct_found:
        print("  Direct (same ICP adjusted by ≥2 xApps):")
        for xapps, param in direct_found:
            print(f"    {' ↔ '.join(sorted(xapps))} via {SHORT_NAME.get(param, param)}")
    else:
        print("  Direct: none detected")

    # Indirect conflicts: two xApps influence same KPI via different ICPs
    kpi_to_xapps: dict[str, list[str]] = {}
    avail_params = [p for p in PARAMS if p in features]
    avail_kpis   = [k for k in KPIS   if k in features]
    for fp in avail_params:
        if fp not in features:
            continue
        i = features.index(fp)
        for fk in avail_kpis:
            if fk not in features:
                continue
            j = features.index(fk)
            if pred_adj[i, j] == 1:
                for xapp, params in xapp_param.items():
                    if fp in params:
                        kpi_to_xapps.setdefault(fk, [])
                        if xapp not in kpi_to_xapps[fk]:
                            kpi_to_xapps[fk].append(xapp)

    indirect_shown = set()
    indirect_found = []
    for kpi, xapps in kpi_to_xapps.items():
        if len(xapps) < 2:
            continue
        for a in sorted(xapps):
            for b in sorted(xapps):
                if a >= b:
                    continue
                key = (a, b)
                if key not in indirect_shown:
                    indirect_shown.add(key)
                    indirect_found.append((a, b, SHORT_NAME.get(kpi, kpi)))

    if indirect_found:
        print("  Indirect (same KPI reachable via different ICPs):")
        for a, b, kpi in indirect_found:
            print(f"    {a} ↔ {b} via {kpi}")
    else:
        print("  Indirect: none detected")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — original style
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "CCO": "#e74c3c",  # red
    "ES": "#f39c12",   # orange
    "MLB": "#2ecc71",  # green
}

def plot_time_series(df: pd.DataFrame, features: list[str], path: str):
    """Time-series plot for all features, with xApp action markers."""
    df_agg = df.groupby("step")[features].mean().reset_index().sort_values("step")
    steps = df_agg["step"].values

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    # Collect xApp action events per step
    xapp_steps: dict[str, list[int]] = {}
    if "xapp" in df.columns:
        for _, row in df.drop_duplicates(subset=["step", "xapp"]).iterrows():
            xapp_steps.setdefault(row["xapp"], []).append(int(row["step"]))

    for ax, feat in zip(axes, features):
        vals = df_agg[feat].values
        ax.plot(steps, vals, color="#2c3e50", lw=1.5, marker=".", markersize=4, zorder=3)
        ax.set_ylabel(SHORT_NAME.get(feat, feat), fontsize=8, rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_tick_params(labelsize=7)

        # xApp action markers
        for xapp, s_list in xapp_steps.items():
            col = PALETTE.get(xapp, "#888")
            for s in s_list:
                ax.axvline(s, color=col, alpha=0.25, lw=0.8)

    # Legend for xApp colors
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=PALETTE.get(x, "#888"), lw=1.5, label=x)
               for x in sorted(xapp_steps.keys())]
    if handles:
        axes[0].legend(handles=handles, fontsize=7, loc="upper right", ncol=len(handles))

    axes[-1].set_xlabel("Step")
    fig.suptitle("TwoCell Conflict Scenario — Feature Time Series", fontsize=11, y=1.001)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_param_kpi_heatmap(mat: np.ndarray, features: list[str], ax, title: str,
                             fmt: str = ".2f", vmin=-1, vmax=1, cmap="RdBu_r",
                             annot=True, cbar=True):
    """Heatmap with params on Y-axis (rows) and KPIs on X-axis (columns)."""
    avail_params = [f for f in PARAMS if f in features]
    avail_kpis   = [f for f in KPIS   if f in features]
    param_idx    = [features.index(p) for p in avail_params]
    kpi_idx      = [features.index(k) for k in avail_kpis]

    sub = mat[np.ix_(param_idx, kpi_idx)]
    row_labels = [SHORT_NAME.get(p, p) for p in avail_params]
    col_labels = [SHORT_NAME.get(k, k) for k in avail_kpis]

    sns.heatmap(
        sub, annot=annot, fmt=fmt, cmap=cmap,
        vmin=vmin, vmax=vmax,
        xticklabels=col_labels, yticklabels=row_labels,
        ax=ax, linewidths=0.3, cbar=cbar,
        annot_kws={"size": 7},
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("KPI", fontsize=9)
    ax.set_ylabel("Parameter (ICP)", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)


def plot_correlation_heatmap(corr: np.ndarray, features: list[str], path: str):
    """Param×KPI submatrix of the Pearson correlation — Y=params, X=KPIs."""
    avail_params = [f for f in PARAMS if f in features]
    avail_kpis   = [f for f in KPIS   if f in features]
    n_p, n_k = len(avail_params), len(avail_kpis)

    fig, ax = plt.subplots(figsize=(max(8, n_k * 0.9), max(4, n_p * 0.85)))
    _plot_param_kpi_heatmap(corr, features, ax,
                             "Param↔KPI Pearson Correlation")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_adjacency_comparison(pred_adj: np.ndarray, gt_adj: np.ndarray,
                               features: list[str], title_pred: str, path: str):
    """Side-by-side GT vs detected adjacency — Y=params, X=KPIs."""
    avail_params = [f for f in PARAMS if f in features]
    avail_kpis   = [f for f in KPIS   if f in features]
    n_p, n_k = len(avail_params), len(avail_kpis)
    w = max(10, n_k * 1.0)
    h = max(4, n_p * 0.85)

    fig, axes = plt.subplots(1, 2, figsize=(w * 2 + 1, h))
    _plot_param_kpi_heatmap(gt_adj.astype(float), features, axes[0],
                             "Ground Truth", fmt=".0f", vmin=0, vmax=1,
                             cmap="Blues", cbar=False)
    _plot_param_kpi_heatmap(pred_adj.astype(float), features, axes[1],
                             title_pred, fmt=".0f", vmin=0, vmax=1,
                             cmap="Blues", cbar=False)
    fig.suptitle("Conflict Graph: Ground Truth vs Detected\n(Y=ICP Parameters, X=KPIs)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_gnn_adjacency_triple(adj_notears: np.ndarray, adj_gnn: np.ndarray,
                               gt_adj: np.ndarray, features: list[str], path: str):
    """Three-panel: GT | NOTEARS | GNN combined — Y=params, X=KPIs."""
    avail_params = [f for f in PARAMS if f in features]
    avail_kpis   = [f for f in KPIS   if f in features]
    n_p, n_k = len(avail_params), len(avail_kpis)
    w = max(8, n_k * 0.95)
    h = max(4, n_p * 0.85)

    fig, axes = plt.subplots(1, 3, figsize=(w * 3 + 2, h))
    pairs = [
        (gt_adj.astype(float), "Ground Truth"),
        (adj_notears.astype(float), "NOTEARS Structure"),
        (adj_gnn.astype(float), "GNN Combined"),
    ]
    for ax, (mat, title) in zip(axes, pairs):
        _plot_param_kpi_heatmap(mat, features, ax, title,
                                 fmt=".0f", vmin=0, vmax=1, cmap="Blues", cbar=False)
    fig.suptitle("GNN-Based Conflict Graph Detection\n(Y=ICP Parameters, X=KPIs)",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_xapp_actions(df: pd.DataFrame, path: str):
    """Per-xApp ICP delta plot over steps."""
    if "xapp" not in df.columns or "adjusted_param" not in df.columns:
        return

    xapps = sorted(df["xapp"].dropna().unique())
    n = len(xapps)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, xapp in zip(axes, xapps):
        sub = df[df["xapp"] == xapp].copy()
        params = sorted(sub["adjusted_param"].dropna().unique())
        for param in params:
            ps = sub[sub["adjusted_param"] == param].copy()
            ax.plot(ps["step"], ps["adjusted_value"],
                    marker="o", markersize=4, lw=1.2,
                    label=param.replace("_delta", "").replace("_ms", "ms"))
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_ylabel("Delta", fontsize=9)
        ax.set_title(f"{xapp} xApp actions", fontsize=10, color=PALETTE.get(xapp, "#333"))
        ax.legend(fontsize=7, loc="upper right", ncol=3)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Step")
    fig.suptitle("xApp Actions (ICP Deltas) Over Steps", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_param_evolution(df: pd.DataFrame, features: list[str], path: str):
    """ICP parameter value evolution over steps."""
    param_feats = [f for f in features if f in PARAMS]
    if not param_feats:
        return

    df_agg = df.groupby("step")[param_feats].mean().reset_index().sort_values("step")
    steps = df_agg["step"].values

    n = len(param_feats)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    cell_colors = {"c0": "#e74c3c", "c1": "#3498db"}
    for ax, feat in zip(axes, param_feats):
        cell = feat.split("_")[0]
        col = cell_colors.get(cell, "#555")
        ax.plot(steps, df_agg[feat].values, color=col, lw=1.8, marker=".", markersize=4)
        ax.set_ylabel(SHORT_NAME.get(feat, feat), fontsize=9, rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.25)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2, label=cell)
               for cell, c in cell_colors.items()]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Step")
    fig.suptitle("ICP Parameter Evolution Over Steps\n(red=cell0, blue=cell1)", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TwoCell O-RAN xApp conflict detection (Pearson + GNN)"
    )
    parser.add_argument("csv", help="TwoCellConflictScenario output CSV")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Pearson threshold for Level-1 binarization (default: 0.5)")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="Analysis level: 1=Pearson only, 2=Pearson+GNN (default: 2)")
    parser.add_argument("--gnn-iter", type=int, default=500,
                        help="NOTEARS gradient iterations (default: 500)")
    parser.add_argument("--warmup-step", type=int, default=9,
                        help="First step used for scenario/KPM validation summaries (default: 9)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only write validation and time-series/action/parameter plots")
    parser.add_argument("--output-dir", default="./twocell_results",
                        help="Output directory (default: ./twocell_results)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {args.csv}")
    df = load_csv(args.csv)
    print(f"  Rows: {len(df)}  |  Steps: {df['step'].nunique()}")
    print(f"  xApps: {sorted(df['xapp'].dropna().unique()) if 'xapp' in df.columns else 'N/A'}")

    X, features = build_feature_matrix(df)
    print(f"  Feature matrix: {X.shape}  ({len(features)} features)")
    print(f"  Features: {features}")

    xapp_param = derive_xapp_param(df)
    gt_adj = build_ground_truth_adj(features)
    validation = validation_report(df, features, args.warmup_step, args.output_dir)

    if args.plots_only:
        print(f"\n── Saving time-series-only plots to {args.output_dir}/ ──")
        plot_time_series(df, features,
                         os.path.join(args.output_dir, "time_series.png"))
        plot_xapp_actions(df,
                          os.path.join(args.output_dir, "xapp_actions.png"))
        plot_param_evolution(df, features,
                             os.path.join(args.output_dir, "param_evolution.png"))
        print("\nDone.")
        return

    # ── Level 1: Pearson correlation ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Level 1: Pearson Correlation Baseline")
    print(f"{'='*55}")
    corr = compute_correlation(X)
    adj_l1 = binarize(corr, args.threshold)

    print(f"\n── Correlation Matrix (threshold={args.threshold}) ──────────")
    short_feats = [SHORT_NAME.get(f, f) for f in features]
    corr_df = pd.DataFrame(np.round(corr, 3), index=short_feats, columns=short_feats)
    print(corr_df.to_string())

    metrics_l1 = evaluate(adj_l1, gt_adj, features, label="L1-Pearson")
    report_conflicts(adj_l1, features, xapp_param, label="L1-Pearson")

    # ── Level 2: GNN ─────────────────────────────────────────────────────────
    metrics_l2 = None
    adj_l2 = None
    adj_notears = None

    if args.level == 2:
        print(f"\n{'='*55}")
        print("Level 2: GNN-Based Conflict Graph Detection")
        print(f"{'='*55}")
        adj_l2, adj_notears, W = gnn_detect_conflict_graph(X, features, n_iter=args.gnn_iter)
        metrics_l2 = evaluate(adj_l2, gt_adj, features, label="L2-GNN")
        report_conflicts(adj_l2, features, xapp_param, label="L2-GNN")

        # NOTEARS-only evaluation
        print()
        _ = evaluate(adj_notears, gt_adj, features, label="L2-NOTEARS-only")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n── Saving plots to {args.output_dir}/ ──")

    plot_time_series(df, features,
                     os.path.join(args.output_dir, "time_series.png"))
    plot_xapp_actions(df,
                      os.path.join(args.output_dir, "xapp_actions.png"))
    plot_param_evolution(df, features,
                         os.path.join(args.output_dir, "param_evolution.png"))
    plot_correlation_heatmap(corr, features,
                             os.path.join(args.output_dir, "correlation_heatmap.png"))
    plot_adjacency_comparison(adj_l1, gt_adj, features,
                              "Detected (Pearson L1)",
                              os.path.join(args.output_dir, "adjacency_L1.png"))

    if args.level == 2 and adj_l2 is not None:
        plot_adjacency_comparison(adj_l2, gt_adj, features,
                                  "Detected (GNN L2)",
                                  os.path.join(args.output_dir, "adjacency_L2.png"))
        plot_gnn_adjacency_triple(adj_notears, adj_l2, gt_adj, features,
                                  os.path.join(args.output_dir, "adjacency_GNN_triple.png"))

    # ── Metrics summary ───────────────────────────────────────────────────────
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"csv={args.csv}\n")
        f.write(f"threshold={args.threshold}\n")
        f.write(f"level={args.level}\n")
        f.write(f"warmup_step={args.warmup_step}\n")
        f.write(f"validation_report={validation['path']}\n")
        f.write(f"action_mismatch_count={validation['action_mismatch_count']}\n")
        f.write("inactive_kpis=" + ",".join(validation["inactive_kpis"]) + "\n")
        f.write("\n[Level1-Pearson]\n")
        for k, v in metrics_l1.items():
            f.write(f"{k}={v:.4f}\n")
        if metrics_l2:
            f.write("\n[Level2-GNN]\n")
            for k, v in metrics_l2.items():
                f.write(f"{k}={v:.4f}\n")
    print(f"  Saved: {metrics_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Summary")
    print(f"{'='*55}")
    print(f"  L1 Pearson  — P={metrics_l1['precision']:.3f}  R={metrics_l1['recall']:.3f}  F1={metrics_l1['f1']:.3f}")
    if metrics_l2:
        print(f"  L2 GNN      — P={metrics_l2['precision']:.3f}  R={metrics_l2['recall']:.3f}  F1={metrics_l2['f1']:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
