# Code Package: Drone Inspection Data, AI/ML, and Decision Intelligence

## Working research title

**Data-Driven AI and Machine Learning for Drone-Based Industrial Pipe Inspection: A Simulation Framework for Maintenance Decision Intelligence**

This package implements the computational study only. It does not generate or revise the final paper.

## What the code does

1. Simulates 10 reproducible datasets containing 300 pipe assets and 12 inspected segments per asset (36,000 segment inspections total).
2. Creates RGB, thermal, pipe-context, flight-quality, and 12-month failure variables with realistic noise and missingness.
3. Separates each replication by pipe asset into training, calibration, and held-out test sets (70%/15%/15%).
4. Fits logistic regression, random forest, and histogram gradient boosting models, with sigmoid probability calibration where specified.
5. Converts predicted risk into monitor, reinspect, planned repair, or urgent shutdown actions by minimizing simulated expected loss.
6. Evaluates probability quality, ranking, maintenance loss, regret, stress conditions, cost sensitivity, and fixed-rule baselines.

## Files

- `paper_pipeline.py`: complete deterministic experiment.
- `validate_pipeline.py`: one-replication contract and leakage smoke test.
- `requirements-lock.txt`: exact Python dependency versions.
- `outputs/synthetic_inspections.csv`: analysis-ready simulated dataset; latent generator variables are excluded.
- `outputs/final_results.csv`: mean and standard deviation across 10 replications.
- `outputs/raw_results.csv`: replication-level metrics.
- `outputs/paired_comparisons.csv`: paired bootstrap comparisons for maintenance expected loss.
- `outputs/test_predictions.csv`: held-out probabilities, actions, and losses.
- `outputs/split_audit.csv` and `outputs/data_audit.csv`: split and data-quality evidence.
- `outputs/checksums.sha256`: deterministic output hashes; timing is intentionally excluded.

## Run

Create an isolated Python environment, install the locked requirements, and execute:

```bash
python -m pip install -r requirements-lock.txt
python validate_pipeline.py
PYTHONHASHSEED=0 python paper_pipeline.py --clean --export-synthetic-data
```

The `--clean` option is deliberately restricted to this package's `outputs` directory.

## Leakage controls

Models receive only the variables in `FEATURE_COLS`. They do not receive asset or segment identifiers, the outcome, the data-generating probability, latent corrosion/crack/leak states, or the asset random effect. Pipe assets cannot cross training, calibration, and test partitions.

## Interpretation boundary

All observations, failures, costs, and maintenance effects are simulated. The code supports reproducible methodological evidence, not claims of field deployment, real operational savings, or autonomous shutdown authority.
