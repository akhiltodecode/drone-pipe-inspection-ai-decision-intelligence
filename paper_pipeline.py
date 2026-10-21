#!/usr/bin/env python3
"""Reproduce the locked synthetic experiment for the IEEE paper.

The script generates all evidence locally. It never downloads data and never exposes
latent simulator variables to a fitted model. See checkpoints/06_protocol_v1.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


DATA_SEEDS = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]
MODEL_SEEDS = [17, 29, 43, 59, 71, 83, 97, 109, 127, 139]
ACTIONS = ["monitor", "reinspect", "planned_repair", "urgent_shutdown"]
RESIDUAL = np.array([1.00, 0.72, 0.25, 0.05], dtype=float)
FAILURE_COST = {"low": 80.0, "medium": 160.0, "high": 320.0}
DIRECT_COST = {
    "low": np.array([0.0, 4.0, 16.0, 38.0]),
    "medium": np.array([0.0, 6.0, 25.0, 60.0]),
    "high": np.array([0.0, 9.0, 40.0, 95.0]),
}
CAT_COLS = ["material", "environment", "criticality"]
NUM_COLS = [
    "age_years", "diameter_mm", "pressure_ratio", "temperature_duty",
    "rgb_corrosion", "rgb_crack", "rgb_coating", "thermal_delta",
    "thermal_hotspot", "thermal_gradient", "wind_ms", "standoff_m",
    "illumination", "blur", "thermal_confidence", "flight_distance_m",
    "battery_remaining",
]
FEATURE_COLS = CAT_COLS + NUM_COLS
PROHIBITED = {
    "asset_id", "segment_id", "failure_12m", "true_probability",
    "latent_corrosion", "latent_crack", "latent_leak", "asset_effect",
    "failure_cost", "direct_cost", "oracle_action",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def generate_replication(seed: int, replicate: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_assets, n_segments = 300, 12
    asset = np.arange(n_assets)
    age = np.clip(rng.gamma(3.2, 6.5, n_assets), 1, 48)
    material = rng.choice(["carbon_steel", "stainless", "alloy"], n_assets,
                          p=[0.58, 0.27, 0.15])
    diameter = np.clip(rng.lognormal(np.log(410), 0.38, n_assets), 100, 1200)
    pressure = np.clip(rng.normal(0.68, 0.16, n_assets), 0.25, 1.12)
    temp = np.clip(rng.beta(2.1, 2.4, n_assets), 0.02, 0.99)
    environment = rng.choice(["indoor", "coastal", "industrial", "arid"], n_assets,
                             p=[0.25, 0.24, 0.34, 0.17])
    criticality = rng.choice(["low", "medium", "high"], n_assets,
                             p=[0.31, 0.45, 0.24])
    asset_effect = rng.normal(0, 0.62, n_assets)

    idx = np.repeat(asset, n_segments)
    seg = np.tile(np.arange(n_segments), n_assets)
    n = len(idx)
    age_r, pressure_r, temp_r = age[idx], pressure[idx], temp[idx]
    mat_r, env_r, crit_r = material[idx], environment[idx], criticality[idx]
    aeff = asset_effect[idx]
    seg_wave = np.sin(2 * np.pi * seg / n_segments)

    mat_cor = pd.Series(mat_r).map({"carbon_steel": 0.65, "stainless": -0.45, "alloy": -0.15}).to_numpy()
    env_cor = pd.Series(env_r).map({"indoor": -0.45, "coastal": 0.85, "industrial": 0.40, "arid": -0.05}).to_numpy()
    mat_crack = pd.Series(mat_r).map({"carbon_steel": 0.20, "stainless": -0.10, "alloy": 0.05}).to_numpy()

    latent_corrosion = sigmoid(-3.1 + 0.085 * age_r + 0.95 * pressure_r + mat_cor + env_cor +
                               0.38 * aeff + 0.28 * seg_wave + rng.normal(0, 0.62, n))
    latent_crack = sigmoid(-3.5 + 0.050 * age_r + 1.25 * pressure_r * temp_r + mat_crack +
                           0.32 * aeff + 0.55 * latent_corrosion + rng.normal(0, 0.70, n))
    latent_leak = sigmoid(-4.0 + 1.85 * latent_corrosion + 1.65 * latent_crack +
                          0.95 * pressure_r + 0.65 * temp_r + 0.25 * aeff + rng.normal(0, 0.62, n))
    failure_logit = (-4.85 + 2.15 * latent_corrosion + 1.85 * latent_crack +
                     2.35 * latent_leak + 0.72 * pressure_r + 0.50 * temp_r +
                     0.72 * latent_corrosion * pressure_r + 0.36 * aeff)
    true_p = np.clip(sigmoid(failure_logit), 0.002, 0.82)
    failure = rng.binomial(1, true_p, n)

    wind = np.clip(rng.gamma(2.0, 1.45, n), 0, 11)
    standoff = np.clip(rng.normal(2.9 + 0.08 * wind, 0.62, n), 1.1, 6.2)
    illumination = np.clip(rng.beta(3.2, 2.0, n) - 0.018 * wind, 0.05, 1)
    blur = np.clip(sigmoid(-3.0 + 0.38 * wind + 0.45 * (standoff - 2.8) + rng.normal(0, 0.55, n)), 0, 1)
    thermal_conf = np.clip(sigmoid(2.5 - 0.24 * wind - 0.32 * (standoff - 2.5) + rng.normal(0, 0.5, n)), 0, 1)
    rgb_sigma = 0.055 + 0.095 * blur + 0.060 * (1 - illumination) + 0.010 * wind
    thermal_sigma = 0.40 + 1.05 * (1 - thermal_conf) + 0.10 * wind

    rgb_corrosion = np.clip(latent_corrosion + rng.normal(0, rgb_sigma), 0, 1)
    rgb_crack = np.clip(latent_crack + rng.normal(0, rgb_sigma * 1.15), 0, 1)
    rgb_coating = np.clip(0.70 * latent_corrosion + 0.16 * (age_r / 48) + rng.normal(0, rgb_sigma), 0, 1)
    thermal_delta = np.clip(1.0 + 8.5 * latent_leak + 2.2 * temp_r + rng.normal(0, thermal_sigma), -2, 15)
    thermal_hotspot = np.clip(0.08 + 0.76 * latent_leak + rng.normal(0, thermal_sigma / 11), 0, 1)
    thermal_gradient = np.clip(0.25 + 3.5 * latent_leak + 0.85 * latent_corrosion + rng.normal(0, thermal_sigma / 2.2), 0, 6)

    p_tmiss = np.clip(sigmoid(-3.0 + 2.2 * (1 - thermal_conf) + 0.16 * wind + 0.28 * (standoff - 3)), 0.02, 0.45)
    tmiss = rng.random(n) < p_tmiss
    p_rmiss = np.clip(0.012 + 0.10 * blur + 0.06 * (1 - illumination), 0.01, 0.24)
    rmiss = rng.random(n) < p_rmiss
    thermal_delta[tmiss] = np.nan
    thermal_hotspot[tmiss] = np.nan
    thermal_gradient[tmiss] = np.nan
    rgb_crack[rmiss] = np.nan

    flight_distance = np.clip(35 + 4.0 * seg + rng.normal(0, 10, n), 8, 120)
    battery = np.clip(0.93 - 0.0040 * flight_distance - 0.013 * wind + rng.normal(0, 0.035, n), 0.12, 0.98)

    return pd.DataFrame({
        "replicate": replicate,
        "asset_id": [f"r{replicate:02d}_a{x:03d}" for x in idx],
        "segment_id": seg,
        "age_years": age_r,
        "material": mat_r,
        "diameter_mm": diameter[idx],
        "pressure_ratio": pressure_r,
        "temperature_duty": temp_r,
        "environment": env_r,
        "criticality": crit_r,
        "rgb_corrosion": rgb_corrosion,
        "rgb_crack": rgb_crack,
        "rgb_coating": rgb_coating,
        "thermal_delta": thermal_delta,
        "thermal_hotspot": thermal_hotspot,
        "thermal_gradient": thermal_gradient,
        "wind_ms": wind,
        "standoff_m": standoff,
        "illumination": illumination,
        "blur": blur,
        "thermal_confidence": thermal_conf,
        "flight_distance_m": flight_distance,
        "battery_remaining": battery,
        "failure_12m": failure,
        "true_probability": true_p,
        "latent_corrosion": latent_corrosion,
        "latent_crack": latent_crack,
        "latent_leak": latent_leak,
        "asset_effect": aeff,
    })


def grouped_split(df: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed + 70001)
    assets = df["asset_id"].unique().copy()
    rng.shuffle(assets)
    train_a, cal_a, test_a = assets[:210], assets[210:255], assets[255:]
    parts = tuple(df[df.asset_id.isin(a)].copy() for a in (train_a, cal_a, test_a))
    sets = [set(x.asset_id) for x in parts]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert [len(x) for x in parts] == [2520, 540, 540]
    return parts


def preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_COLS),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT_COLS),
    ], remainder="drop")


class SigmoidCalibrator:
    def __init__(self):
        self.model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")

    def fit(self, probability: np.ndarray, y: np.ndarray):
        p = np.clip(probability, 1e-6, 1 - 1e-6)
        self.model.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(probability, 1e-6, 1 - 1e-6)
        return np.clip(self.model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def expert_probability(df: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    cols = ["rgb_corrosion", "rgb_crack", "rgb_coating", "thermal_delta",
            "thermal_hotspot", "thermal_gradient"]
    med = train[cols].median(numeric_only=True)
    x = df[cols].fillna(med)
    z = (-5.15 + 1.75 * x.rgb_corrosion + 1.40 * x.rgb_crack + 0.70 * x.rgb_coating +
         0.115 * x.thermal_delta + 0.85 * x.thermal_hotspot + 0.18 * x.thermal_gradient +
         0.55 * df.pressure_ratio + 0.30 * df.temperature_duty + 0.35 * (df.age_years / 48) -
         0.20 * df.blur - 0.12 * (1 - df.thermal_confidence))
    return np.clip(sigmoid(z.to_numpy()), 1e-6, 1 - 1e-6)


def fixed_actions(p: np.ndarray) -> np.ndarray:
    return np.select([p < 0.10, p < 0.25, p < 0.55], [0, 1, 2], default=3).astype(int)


def cost_arrays(criticality: pd.Series, failure_scale=1.0, residual_scale=1.0):
    direct = np.vstack([DIRECT_COST[c] for c in criticality])
    fail = np.array([FAILURE_COST[c] for c in criticality], dtype=float) * failure_scale
    residual = np.clip(RESIDUAL * residual_scale, 0, 1)
    residual[0] = 1.0
    return direct, fail, residual


def optimal_actions(p: np.ndarray, criticality: pd.Series, failure_scale=1.0, residual_scale=1.0):
    direct, fail, residual = cost_arrays(criticality, failure_scale, residual_scale)
    losses = direct + p[:, None] * fail[:, None] * residual[None, :]
    return np.argmin(losses, axis=1)


def realized_expected_loss(true_p: np.ndarray, action: np.ndarray, criticality: pd.Series,
                           failure_scale=1.0, residual_scale=1.0) -> np.ndarray:
    direct, fail, residual = cost_arrays(criticality, failure_scale, residual_scale)
    row = np.arange(len(action))
    return direct[row, action] + true_p * fail * residual[action]


def ece(y: np.ndarray, p: np.ndarray, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def metric_rows(rep: int, scenario: str, method: str, df: pd.DataFrame,
                p: np.ndarray, action: np.ndarray, inference_ms: float = math.nan,
                failure_scale=1.0, residual_scale=1.0):
    y, true_p = df.failure_12m.to_numpy(), df.true_probability.to_numpy()
    loss = realized_expected_loss(true_p, action, df.criticality, failure_scale, residual_scale)
    oracle_a = optimal_actions(true_p, df.criticality, failure_scale, residual_scale)
    oracle_loss = realized_expected_loss(true_p, oracle_a, df.criticality, failure_scale, residual_scale)
    top_n = max(1, math.ceil(0.10 * len(df)))
    true_top = set(np.argsort(true_p)[-top_n:])
    pred_top = set(np.argsort(p)[-top_n:])
    vals = {
        "MEL": loss.mean(),
        "regret": (loss - oracle_loss).mean(),
        "PR_AUC": average_precision_score(y, p) if np.unique(y).size == 2 else math.nan,
        "Brier": brier_score_loss(y, p),
        "ECE10": ece(y, p),
        "high_risk_recall": len(true_top & pred_top) / top_n,
        "urgent_rate": np.mean(action == 3),
        "n": len(df),
        "failure_prevalence": y.mean(),
    }
    rows = [{"replicate": rep, "scenario": scenario, "method": method,
             "metric": k, "value": float(v)} for k, v in vals.items()]
    return rows, loss, oracle_loss


def stress_features(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 900003)
    x = df.copy()
    # Adding independent noise with SD sqrt(1.5^2-1) times the generator SD
    # increases total heteroscedastic measurement noise to 1.5x in variance sum.
    factor = math.sqrt(1.5**2 - 1.0)
    rgb_sigma = 0.055 + 0.095 * x.blur + 0.060 * (1 - x.illumination) + 0.010 * x.wind_ms
    thermal_sigma = 0.40 + 1.05 * (1 - x.thermal_confidence) + 0.10 * x.wind_ms
    for c, scale in [("rgb_corrosion", 1.0), ("rgb_crack", 1.15), ("rgb_coating", 1.0)]:
        x[c] = np.clip(x[c] + rng.normal(0, factor * scale * rgb_sigma, len(x)), 0, 1)
    x["thermal_delta"] = np.clip(x.thermal_delta + rng.normal(0, factor * thermal_sigma, len(x)), -2, 15)
    x["thermal_hotspot"] = np.clip(x.thermal_hotspot + rng.normal(0, factor * thermal_sigma / 11, len(x)), 0, 1)
    x["thermal_gradient"] = np.clip(x.thermal_gradient + rng.normal(0, factor * thermal_sigma / 2.2, len(x)), 0, 6)
    x["blur"] = np.clip(x.blur + 0.15, 0, 1)
    x["thermal_confidence"] = np.clip(x.thermal_confidence - 0.18, 0, 1)
    thermal_cols = ["thermal_delta", "thermal_hotspot", "thermal_gradient"]
    current_missing = x[thermal_cols].isna().all(axis=1).to_numpy()
    desired = math.ceil(0.25 * len(x))
    candidates = np.flatnonzero(~current_missing)
    if current_missing.sum() < desired:
        chosen = rng.choice(candidates, desired - current_missing.sum(), replace=False)
        x.loc[x.index[chosen], thermal_cols] = np.nan
    return x


def fit_models(train: pd.DataFrame, cal: pd.DataFrame, seed: int):
    assert not (set(FEATURE_COLS) & PROHIBITED)
    ytr, ycal = train.failure_12m.to_numpy(), cal.failure_12m.to_numpy()
    lr = Pipeline([("prep", preprocessor(True)),
                   ("model", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs"))])
    rf = Pipeline([("prep", preprocessor(False)),
                   ("model", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                                    class_weight="balanced", random_state=seed, n_jobs=1))])
    gb = Pipeline([("prep", preprocessor(False)),
                   ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                            max_leaf_nodes=15, l2_regularization=1.0,
                                                            random_state=seed))])
    lr.fit(train[FEATURE_COLS], ytr)
    rf.fit(train[FEATURE_COLS], ytr)
    gb.fit(train[FEATURE_COLS], ytr, model__sample_weight=compute_sample_weight("balanced", ytr))
    bases = {"Logistic + calibration": lr, "Random forest + calibration": rf,
             "Gradient boosting (uncalibrated)": gb}
    calibrators = {}
    for name, model in bases.items():
        if "uncalibrated" not in name:
            calibrators[name] = SigmoidCalibrator().fit(model.predict_proba(cal[FEATURE_COLS])[:, 1], ycal)
    calibrators["Gradient boosting + calibration"] = SigmoidCalibrator().fit(
        gb.predict_proba(cal[FEATURE_COLS])[:, 1], ycal)
    return lr, rf, gb, calibrators


def predict_all(df: pd.DataFrame, train: pd.DataFrame, lr, rf, gb, calibrators):
    timed = {}
    pred = {}
    t0 = time.perf_counter(); pred["Fixed rule heuristic"] = expert_probability(df, train); timed["Fixed rule heuristic"] = (time.perf_counter()-t0)*1000*1000/len(df)
    for name, model in [("Logistic + calibration", lr), ("Random forest + calibration", rf),
                        ("Gradient boosting (uncalibrated)", gb)]:
        t0 = time.perf_counter(); base = model.predict_proba(df[FEATURE_COLS])[:, 1]; elapsed = (time.perf_counter()-t0)*1000*1000/len(df)
        pred[name] = base if "uncalibrated" in name else calibrators[name].predict(base)
        timed[name] = elapsed
    gb_base = gb.predict_proba(df[FEATURE_COLS])[:, 1]
    gb_cal = calibrators["Gradient boosting + calibration"].predict(gb_base)
    pred["Gradient boosting + fixed thresholds"] = gb_cal
    pred["Proposed calibrated GB + optimizer"] = gb_cal
    timed["Gradient boosting + fixed thresholds"] = timed["Gradient boosting (uncalibrated)"]
    timed["Proposed calibrated GB + optimizer"] = timed["Gradient boosting (uncalibrated)"]
    return pred, timed


def audit_data(df: pd.DataFrame, train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame):
    assert len(df) == 3600 and df.asset_id.nunique() == 300
    assert (df.groupby("asset_id").size() == 12).all()
    assert set(df.failure_12m.unique()).issubset({0, 1})
    assert np.isfinite(df.true_probability).all() and df.true_probability.between(0, 1).all()
    group_sets = [set(x.asset_id) for x in (train, cal, test)]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    assert not (set(FEATURE_COLS) & PROHIBITED)
    hashes = [set(pd.util.hash_pandas_object(x[FEATURE_COLS], index=False).astype(str))
              for x in (train, cal, test)]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    for c in ["age_years", "diameter_mm", "pressure_ratio", "temperature_duty"]:
        assert np.isfinite(df[c]).all()


def public_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return the analysis-ready synthetic data without simulator-only fields.

    Asset and segment identifiers are retained so users can reproduce the
    grouped split.  The target is retained for model development.  Latent
    states, the data-generating probability, and the asset random effect are
    deliberately excluded because they are unavailable to fitted models.
    """
    columns = ["replicate", "asset_id", "segment_id", *FEATURE_COLS, "failure_12m"]
    exported = df.loc[:, columns].copy()
    simulator_only = {"true_probability", "latent_corrosion", "latent_crack",
                      "latent_leak", "asset_effect"}
    assert not (set(exported.columns) & simulator_only)
    assert exported.columns.is_unique
    return exported


def bootstrap_comparisons(loss_records: pd.DataFrame, n_boot=2000):
    rng = np.random.default_rng(440021)
    proposed = "Proposed calibrated GB + optimizer"
    out = []
    main = loss_records[loss_records.scenario == "main"]
    for comparator in sorted(set(main.method) - {proposed}):
        a = main[main.method == proposed][["replicate", "asset_id", "loss"]]
        b = main[main.method == comparator][["replicate", "asset_id", "loss"]]
        paired = a.merge(b, on=["replicate", "asset_id"], suffixes=("_proposed", "_other"))
        paired["reduction"] = paired.loss_other - paired.loss_proposed
        by_rep = {r: g.reduction.to_numpy() for r, g in paired.groupby("replicate")}
        boots = np.empty(n_boot)
        for i in range(n_boot):
            boots[i] = np.mean([rng.choice(v, len(v), replace=True).mean() for v in by_rep.values()])
        mean = paired.groupby("replicate").reduction.mean().mean()
        lo, hi = np.quantile(boots, [0.025, 0.975])
        other_mel = main[main.method == comparator].groupby("replicate").loss.mean().mean()
        out.append({"comparison": f"Proposed vs {comparator}", "mel_reduction": mean,
                    "relative_reduction_pct": 100 * mean / other_mel, "ci95_low": lo,
                    "ci95_high": hi, "bootstrap_resamples": n_boot})
    return pd.DataFrame(out)


def workflow_figure(path: Path):
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.axis("off")
    labels = ["Drone-observable RGB + thermal", "Acquisition quality + pipe context",
              "Calibrated 12-month failure risk", "Expected-loss maintenance action"]
    colors = ["#DDEBF7", "#E2F0D9", "#FFF2CC", "#FCE4D6"]
    ys = [3.15, 2.20, 1.25, 0.30]
    for i, (y, label, color) in enumerate(zip(ys, labels, colors)):
        rect = plt.Rectangle((0.25, y), 3.50, 0.55, facecolor=color, edgecolor="#2F5597", lw=1.0)
        ax.add_patch(rect); ax.text(2.0, y+0.275, label, ha="center", va="center", fontsize=8.4)
        if i < 3:
            ax.annotate("", xy=(2.0, y-0.12), xytext=(2.0, y-0.02),
                        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0))
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--export-synthetic-data",
        action="store_true",
        help=("also write synthetic_inspections.csv containing model-observable "
              "features, grouping identifiers, and the simulated outcome"),
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    project = Path(__file__).parent.resolve()
    if args.clean and out.exists():
        if out.parent != project or out.name != "outputs":
            raise ValueError("--clean is restricted to this project's outputs directory")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    raw_rows, pred_rows, loss_rows, split_rows, action_rows, timing_rows = [], [], [], [], [], []
    exported_datasets = []
    data_audits = []
    for rep, (dseed, mseed) in enumerate(zip(DATA_SEEDS, MODEL_SEEDS), start=1):
        df = generate_replication(dseed, rep)
        if args.export_synthetic_data:
            exported_datasets.append(public_dataset(df))
        train, cal, test = grouped_split(df, dseed)
        audit_data(df, train, cal, test)
        lr, rf, gb, calibrators = fit_models(train, cal, mseed)
        predictions, timing = predict_all(test, train, lr, rf, gb, calibrators)
        oracle = optimal_actions(test.true_probability.to_numpy(), test.criticality)
        for method, p in predictions.items():
            action = fixed_actions(p) if method in {"Fixed rule heuristic", "Gradient boosting + fixed thresholds"} else optimal_actions(p, test.criticality)
            rows, losses, _ = metric_rows(rep, "main", method, test, p, action, timing[method])
            raw_rows.extend(rows)
            timing_rows.append({"replicate": rep, "scenario": "main", "method": method,
                                "inference_ms_per_1000": timing[method],
                                "note": "environment-dependent; excluded from scientific checksums"})
            for i, (_, row) in enumerate(test.iterrows()):
                pred_rows.append({"replicate": rep, "asset_id": row.asset_id, "segment_id": int(row.segment_id),
                                  "method": method, "y": int(row.failure_12m), "true_probability": row.true_probability,
                                  "predicted_probability": p[i], "action": ACTIONS[action[i]],
                                  "oracle_action": ACTIONS[oracle[i]], "expected_loss": losses[i]})
            per_asset = pd.DataFrame({"asset_id": test.asset_id.to_numpy(), "loss": losses}).groupby("asset_id", as_index=False).loss.mean()
            for x in per_asset.itertuples(index=False):
                loss_rows.append({"replicate": rep, "scenario": "main", "asset_id": x.asset_id, "method": method, "loss": x.loss})
            counts = pd.Series(action).value_counts()
            for ai, action_name in enumerate(ACTIONS):
                action_rows.append({"replicate": rep, "scenario": "main", "method": method,
                                    "action": action_name, "count": int(counts.get(ai, 0))})

        stressed = stress_features(test, dseed)
        stress_predictions, stress_timing = predict_all(stressed, train, lr, rf, gb, calibrators)
        for method, p in stress_predictions.items():
            action = fixed_actions(p) if method in {"Fixed rule heuristic", "Gradient boosting + fixed thresholds"} else optimal_actions(p, stressed.criticality)
            rows, _, _ = metric_rows(rep, "sensor_stress", method, stressed, p, action, stress_timing[method])
            raw_rows.extend(rows)

        quality = (test.illumination + test.thermal_confidence + (1-test.blur) +
                   (1-test.wind_ms/11) + (1-(test.standoff_m-1.1)/5.1)) / 5
        regimes = {
            "lowest_quality_quintile": quality <= quality.quantile(0.20),
            "high_wind_thermal_missing": (test.wind_ms >= test.wind_ms.quantile(0.75)) & test.thermal_delta.isna(),
        }
        for scenario, mask in regimes.items():
            subset = test[mask].copy()
            if len(subset) < 10:
                continue
            pos = np.flatnonzero(mask.to_numpy())
            for method, p_all in predictions.items():
                p = p_all[pos]
                action = fixed_actions(p) if method in {"Fixed rule heuristic", "Gradient boosting + fixed thresholds"} else optimal_actions(p, subset.criticality)
                rows, _, _ = metric_rows(rep, scenario, method, subset, p, action)
                raw_rows.extend(rows)

        for fscale in [0.75, 1.00, 1.25]:
            for rscale in [0.80, 1.00, 1.20]:
                scenario = f"cost_f{fscale:.2f}_r{rscale:.2f}"
                for method, p in predictions.items():
                    action = fixed_actions(p) if method in {"Fixed rule heuristic", "Gradient boosting + fixed thresholds"} else optimal_actions(p, test.criticality, fscale, rscale)
                    rows, _, _ = metric_rows(rep, scenario, method, test, p, action,
                                             failure_scale=fscale, residual_scale=rscale)
                    raw_rows.extend([x for x in rows if x["metric"] in {"MEL", "regret", "urgent_rate"}])

        for split_name, part in [("train", train), ("calibration", cal), ("test", test)]:
            split_rows.append({"replicate": rep, "split": split_name, "rows": len(part),
                               "assets": part.asset_id.nunique(), "failure_prevalence": part.failure_12m.mean(),
                               "thermal_missing_rate": part.thermal_delta.isna().mean(),
                               "rgb_crack_missing_rate": part.rgb_crack.isna().mean()})
        data_audits.append({"replicate": rep, "data_seed": dseed, "model_seed": mseed,
                            "rows": len(df), "assets": df.asset_id.nunique(),
                            "failure_prevalence": df.failure_12m.mean(),
                            "true_probability_mean": df.true_probability.mean()})

    raw = pd.DataFrame(raw_rows)
    predictions_df = pd.DataFrame(pred_rows)
    loss_df = pd.DataFrame(loss_rows)
    final = (raw.groupby(["scenario", "method", "metric"], as_index=False)
             .agg(mean=("value", "mean"), sd=("value", "std"), replications=("value", "count")))
    comparisons = bootstrap_comparisons(loss_df, 2000)
    raw.to_csv(out / "raw_results.csv", index=False)
    final.to_csv(out / "final_results.csv", index=False)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    predictions_df.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out / "split_audit.csv", index=False)
    pd.DataFrame(action_rows).to_csv(out / "action_counts.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(out / "timing_results.csv", index=False)
    pd.DataFrame(data_audits).to_csv(out / "data_audit.csv", index=False)
    if args.export_synthetic_data:
        pd.concat(exported_datasets, ignore_index=True).to_csv(
            out / "synthetic_inspections.csv", index=False
        )
    workflow_figure(out / "fig1_workflow.png")

    versions = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__, "data_seeds": DATA_SEEDS, "model_seeds": MODEL_SEEDS,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "not set; all numerical RNGs explicitly seeded"),
    }
    (out / "environment.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    dictionary = {
        "unit": "one simulated drone inspection of one pipe segment",
        "group": "asset_id (12 segments)", "target": "failure_12m within 12 months",
        "model_features": FEATURE_COLS, "prohibited_model_fields": sorted(PROHIBITED),
        "actions": ACTIONS, "cost_unit": "dimensionless simulation cost unit",
    }
    (out / "data_dictionary.json").write_text(json.dumps(dictionary, indent=2) + "\n", encoding="utf-8")
    manifest = []
    for path in sorted(out.iterdir()):
        if path.name not in {"checksums.sha256", "timing_results.csv"} and path.is_file():
            manifest.append(f"{file_sha256(path)}  {path.name}")
    (out / "checksums.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    main_table = final[(final.scenario == "main") & final.metric.isin(["MEL", "regret", "PR_AUC", "Brier", "ECE10"])]
    print(main_table.to_string(index=False))
    print("\nPaired primary comparisons (positive = proposed lower MEL):")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
