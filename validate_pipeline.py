#!/usr/bin/env python3
"""Fast contract validation for the drone-inspection research pipeline."""

from __future__ import annotations

import numpy as np

from paper_pipeline import (
    ACTIONS,
    FEATURE_COLS,
    PROHIBITED,
    fit_models,
    generate_replication,
    grouped_split,
    optimal_actions,
    predict_all,
    public_dataset,
)


def main() -> None:
    data = generate_replication(seed=101, replicate=1)
    repeated = generate_replication(seed=101, replicate=1)
    assert data.equals(repeated), "simulation is not deterministic"
    assert len(data) == 3600 and data["asset_id"].nunique() == 300
    assert (data.groupby("asset_id").size() == 12).all()

    train, calibration, test = grouped_split(data, seed=101)
    asset_sets = [set(frame["asset_id"]) for frame in (train, calibration, test)]
    assert not (asset_sets[0] & asset_sets[1])
    assert not (asset_sets[0] & asset_sets[2])
    assert not (asset_sets[1] & asset_sets[2])
    assert [len(frame) for frame in (train, calibration, test)] == [2520, 540, 540]
    assert not (set(FEATURE_COLS) & PROHIBITED), "model feature leakage detected"

    exported = public_dataset(data)
    assert len(exported) == len(data)
    assert not ({"true_probability", "latent_corrosion", "latent_crack",
                 "latent_leak", "asset_effect"} & set(exported.columns))

    lr, rf, gb, calibrators = fit_models(train, calibration, seed=17)
    predictions, _ = predict_all(test, train, lr, rf, gb, calibrators)
    assert len(predictions) == 6
    for name, probability in predictions.items():
        assert probability.shape == (len(test),), name
        assert np.isfinite(probability).all(), name
        assert ((probability >= 0) & (probability <= 1)).all(), name
        action = optimal_actions(probability, test["criticality"])
        assert action.shape == (len(test),), name
        assert set(np.unique(action)).issubset(range(len(ACTIONS))), name

    print("PASS: deterministic simulation")
    print("PASS: grouped asset split (2520/540/540 rows)")
    print("PASS: no prohibited model features")
    print("PASS: leakage-safe public dataset schema")
    print("PASS: six prediction policies and valid decision actions")


if __name__ == "__main__":
    main()
