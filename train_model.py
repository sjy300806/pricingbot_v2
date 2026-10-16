
"""Train the stock-ranking model using technical, SEC and M2 data."""

from pathlib import Path
import json

import pandas as pd
import joblib
from lightgbm import LGBMRegressor

from build_features import FEATURES, FUND, DATA, ROOT

OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

M2_FEATURES = [
    "m2_billions",
    "m2_growth_3m",
    "m2_growth_6m",
    "m2_growth_12m",
    "m2_acceleration",
]

MODEL_ARGS = dict(
    n_estimators=100,
    num_leaves=7,
    learning_rate=0.03,
    min_child_samples=100,
    reg_lambda=20,
    verbosity=-1,
    random_state=42,
    n_jobs=2,
)


def main():
    # 1. Load the combined technical + SEC + M2 dataset
    file = DATA / "weekly_ranking_with_fundamentals_m2.csv"
    df = pd.read_csv(file, parse_dates=["date"])

    # 2. Check that all required features exist
    cols = FEATURES + FUND + M2_FEATURES
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    # 3. Use only observations with known future outcomes
    if "target_excess" not in df.columns:
        raise SystemExit(
            "Missing target_excess. Training labels must be built first."
        )

    train = df[df["target_excess"].notna()].copy()

    if train["date"].nunique() < 100:
        raise SystemExit("Insufficient matured weekly training history")

    # 4. Train the model
    model = LGBMRegressor(**MODEL_ARGS)
    model.fit(train[cols], train["target_excess"])

    # 5. Save model and feature definitions
    joblib.dump(
        {
            "model": model,
            "features": cols,
            "trained_through": str(train["date"].max().date()),
            "params": MODEL_ARGS,
        },
        OUT / "ranker_m2.joblib",
    )

    info = {
        "training_weeks": int(train["date"].nunique()),
        "training_rows": len(train),
        "trained_through": str(train["date"].max().date()),
        "features": cols,
        "parameters": MODEL_ARGS,
        "m2_enabled": True,
    }

    (OUT / "model_info_m2.json").write_text(
        json.dumps(info, indent=2)
    )

    print("M2 MODEL TRAINED SUCCESSFULLY")
    print(f"Training observations: {len(train):,}")
    print(f"Training weeks: {train['date'].nunique()}")
    print(f"Features used: {len(cols)}")
    print(f"Trained through: {train['date'].max().date()}")
    print("Saved model: models/ranker_m2.joblib")


if __name__ == "__main__":
    main()
