#!/usr/bin/env python3
"""
Conflict detection analysis based on:
  "Learning and Reconstructing Conflicts in O-RAN: A Graph Neural Network Approach"
  IEEE WCNC 2025

Level 1: Pearson correlation on time-series param+KPI data -> binarized adjacency matrix
         Evaluated against ground-truth conflict graph derived from simulation design.

Usage:
  python3 analyze_conflicts.py <csv_file> [--threshold 0.7] [--output-dir ./results]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# ─── Ground Truth ───────────────────────────────────────────────────────────
#
# Param → KPI (causal, known from physics/design — evaluation only)
PARAM_KPI: dict[str, list[str]] = {
    # eNB DL TxPower → DL signal quality + power consumption
    "TxPower": ["AvgCqi", "dlThroughput", "EstimatedPower_W", "rbUtil", "RSRP", "RSRQ"],
    # TxMode2Gain: multiplies SINR in CQI computation — effect too small to detect in ns3-LTE
    # UeTxPower: UL TxPower — SRS SINR path does not reflect UeTxPower changes in ns3-LTE FDD
}

# xApp → Param assignment (one param per xApp — no co-variation by design)
#   CoverageXapp → TxPower      (raises TxPower when CQI low)
#   EnergyXapp   → TxPower      (lowers TxPower when throughput high) ← direct conflict
#   QoSXapp      → TxMode2Gain  (sweeps CQI SINR multiplier)          ← indirect conflict
#   LoadXapp     → UeTxPower    (sweeps UE UL power — UL domain)

PARAMS = ["TxPower", "TxMode2Gain", "UeTxPower"]
KPIS   = ["rbUtil", "dlThroughput", "AvgCqi", "EstimatedPower_W",
          "RSRP", "RSRQ", "ulThroughput", "ulSinr", "ulInterference",
          "ServedUes", "FarUes"]
FEATURES = PARAMS + KPIS


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def derive_xapp_param(df: pd.DataFrame) -> dict[str, list[str]]:
    """Infer xApp→param relationships directly from CSV observations."""
    xapp_param: dict[str, list[str]] = {}
    for xapp, group in df.groupby("xapp"):
        params = sorted(group["adjusted_param"].dropna().unique().tolist())
        xapp_param[xapp] = params
    return xapp_param


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return X shape [T, len(FEATURES)] sorted by step."""
    df_sorted = df.sort_values("step").reset_index(drop=True)
    return df_sorted[FEATURES].values.astype(float)


def compute_correlation(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix [F x F]. Constant columns yield 0 correlation."""
    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def binarize(corr: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize by absolute correlation >= threshold."""
    adj = (np.abs(corr) >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def build_ground_truth_adj(features: list[str]) -> np.ndarray:
    """
    Build ground-truth adjacency matrix for param-KPI relationships.
    adj[i,j] = 1 means features[i] causally influences features[j] (or vice versa).
    """
    n = len(features)
    adj = np.zeros((n, n), dtype=int)
    for param, kpis in PARAM_KPI.items():
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


def evaluate(pred_adj: np.ndarray, gt_adj: np.ndarray, features: list[str]):
    """Print precision/recall/F1 for param-KPI edges only."""
    param_idx = [features.index(p) for p in PARAMS if p in features]
    kpi_idx   = [features.index(k) for k in KPIS   if k in features]

    y_true, y_pred = [], []
    edge_labels = []
    for i in param_idx:
        for j in kpi_idx:
            y_true.append(gt_adj[i, j])
            y_pred.append(pred_adj[i, j])
            edge_labels.append(f"{features[i]}→{features[j]}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n── Param→KPI Edge Detection ──────────────────────────")
    print(f"  Precision : {p:.3f}")
    print(f"  Recall    : {r:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print()
    print("  Edge                              GT  Pred")
    for label, gt, pred in zip(edge_labels, y_true, y_pred):
        mark = "✓" if gt == pred else "✗"
        print(f"  {label:<38} {gt}   {pred}  {mark}")

    return {"precision": p, "recall": r, "f1": f1}


def report_conflicts(pred_adj: np.ndarray, features: list[str],
                     xapp_param: dict[str, list[str]]):
    """Derive and print xApp-level conflict types from detected param-KPI edges."""
    print("\n── Observed xApp→Param (from data) ──────────────────")
    for xapp, params in sorted(xapp_param.items()):
        print(f"  {xapp}: {params}")

    print("\n── Detected Conflicts ────────────────────────────────")

    # Direct: two xApps that adjusted the same parameter
    param_to_xapps: dict[str, list[str]] = {}
    for xapp, params in xapp_param.items():
        for p in params:
            param_to_xapps.setdefault(p, [])
            if xapp not in param_to_xapps[p]:
                param_to_xapps[p].append(xapp)

    direct_found = [(xapps, p) for p, xapps in param_to_xapps.items() if len(xapps) > 1]
    if direct_found:
        print("  Direct (same param adjusted by multiple xApps):")
        for xapps, param in direct_found:
            print(f"    {' ↔ '.join(sorted(xapps))} via {param}")
    else:
        print("  Direct: none")

    # Indirect: two xApps affecting the same KPI through different detected params
    kpi_to_xapps: dict[str, list[str]] = {}
    for i, fi in enumerate(features):
        if fi not in PARAMS:
            continue
        for j, fj in enumerate(features):
            if fj not in KPIS:
                continue
            if pred_adj[i, j] == 1:
                for xapp, params in xapp_param.items():
                    if fi in params:
                        kpi_to_xapps.setdefault(fj, [])
                        if xapp not in kpi_to_xapps[fj]:
                            kpi_to_xapps[fj].append(xapp)

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
                    indirect_found.append((a, b, kpi))

    if indirect_found:
        print("  Indirect (same KPI via different params):")
        for a, b, kpi in indirect_found:
            print(f"    {a} ↔ {b} via {kpi}")
    else:
        print("  Indirect: none detected")


def plot_correlation_heatmap(corr: np.ndarray, features: list[str], path: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(features), dtype=bool)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=features,
        yticklabels=features,
        ax=ax,
        linewidths=0.5,
    )
    # Draw box around param-KPI submatrix
    n_params = sum(1 for f in features if f in PARAMS)
    n_kpis   = sum(1 for f in features if f in KPIS)
    ax.add_patch(plt.Rectangle(
        (n_params, 0), n_kpis, n_params,
        fill=False, edgecolor="gold", lw=2.5, label="Param→KPI region"
    ))
    ax.set_title("Feature Correlation Matrix\n(gold box = Param↔KPI region)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_adjacency_comparison(pred_adj: np.ndarray, gt_adj: np.ndarray,
                              features: list[str], path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in zip(axes, [gt_adj, pred_adj],
                               ["Ground Truth", "Detected (Pearson)"]):
        sns.heatmap(
            mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            vmin=0, vmax=1,
            xticklabels=features,
            yticklabels=features,
            ax=ax,
            linewidths=0.5,
            cbar=False,
        )
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle("Conflict Graph: Ground Truth vs Detected")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_time_series(df: pd.DataFrame, path: str):
    df_sorted = df.sort_values("step").reset_index(drop=True)
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(12, 2 * len(FEATURES)), sharex=True)
    for ax, feat in zip(axes, FEATURES):
        if feat not in df_sorted.columns:
            continue
        ax.plot(df_sorted["step"], df_sorted[feat], marker=".", markersize=3)
        ax.set_ylabel(feat, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    fig.suptitle("Feature Time Series")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="O-RAN xApp conflict detection from CSV")
    parser.add_argument("csv", help="Path to simulation output CSV")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Correlation threshold for binarization (default: 0.7)")
    parser.add_argument("--output-dir", default="./conflict_results",
                        help="Directory for output files (default: ./conflict_results)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {args.csv}")
    df = load_csv(args.csv)
    print(f"  Steps: {len(df)}  |  Columns: {list(df.columns)}")

    xapp_param = derive_xapp_param(df)

    X = build_feature_matrix(df)
    print(f"  Feature matrix: {X.shape}")

    corr = compute_correlation(X)
    pred_adj = binarize(corr, args.threshold)
    gt_adj   = build_ground_truth_adj(FEATURES)

    print(f"\n── Correlation Matrix (threshold={args.threshold}) ──────────")
    corr_df = pd.DataFrame(np.round(corr, 3), index=FEATURES, columns=FEATURES)
    print(corr_df.to_string())

    metrics = evaluate(pred_adj, gt_adj, FEATURES)
    report_conflicts(pred_adj, FEATURES, xapp_param)

    print("\n── Saving plots ──────────────────────────────────────")
    plot_correlation_heatmap(corr, FEATURES,
                             os.path.join(args.output_dir, "correlation_heatmap.png"))
    plot_adjacency_comparison(pred_adj, gt_adj, FEATURES,
                              os.path.join(args.output_dir, "adjacency_comparison.png"))
    plot_time_series(df, os.path.join(args.output_dir, "time_series.png"))

    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"threshold={args.threshold}\n")
        for k, v in metrics.items():
            f.write(f"{k}={v:.4f}\n")
    print(f"  Saved: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
