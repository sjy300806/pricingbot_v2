
"""Walk-forward comparison: original vs M2 stock-ranking model."""

from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from build_features import FEATURES, FUND

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

M2 = [
    "m2_billions",
    "m2_growth_3m",
    "m2_growth_6m",
    "m2_growth_12m",
    "m2_acceleration",
]

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-10-16")

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


def make_model():
    return LGBMRegressor(**MODEL_ARGS)


def main():
    path = DATA / "weekly_ranking_with_fundamentals_m2.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    base_cols = FEATURES + FUND
    m2_cols = base_cols + M2

    required = set(
        ["date", "symbol", "future_return", "target_excess"]
        + m2_cols
    )
    missing = sorted(required - set(df.columns))

    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    results = []
    test_dates = sorted(
        df.loc[
            (df["date"] >= START) &
            (df["date"] <= END) &
            df["target_excess"].notna(),
            "date"
        ].unique()
    )

    print(f"Testing {len(test_dates)} unseen weeks...")

    for test_date in test_dates:
        test_date = pd.Timestamp(test_date)

        # Conservative label-maturity buffer:
        # training labels must come from at least 10
        # calendar days before the test snapshot.
        cutoff = test_date - pd.Timedelta(days=10)

        train = df[
            (df["date"] < cutoff) &
            df["target_excess"].notna()
        ].copy()

        test = df[df["date"] == test_date].copy()

        if train["date"].nunique() < 100:
            continue

        if len(test) != 25 or test["symbol"].nunique() != 25:
            continue

        if test["future_return"].isna().any():
            continue

        original = make_model()
        original.fit(
            train[base_cols],
            train["target_excess"]
        )

        with_m2 = make_model()
        with_m2.fit(
            train[m2_cols],
            train["target_excess"]
        )

        test["original_score"] = original.predict(
            test[base_cols]
        )
        test["m2_score"] = with_m2.predict(
            test[m2_cols]
        )

        original_top = test.nlargest(
            5, "original_score"
        )

        m2_top = test.nlargest(
            5, "m2_score"
        )

        benchmark = test["future_return"].mean()

        results.append({
            "date": test_date,
            "original_return": original_top[
                "future_return"
            ].mean(),
            "m2_return": m2_top[
                "future_return"
            ].mean(),
            "benchmark_return": benchmark,
            "original_top5": ",".join(
                original_top["symbol"]
            ),
            "m2_top5": ",".join(
                m2_top["symbol"]
            ),
        })

    if not results:
        raise SystemExit("No valid test weeks found.")

    out = pd.DataFrame(results).sort_values("date")

    strategies = {
        "Original AI": "original_return",
        "M2 AI": "m2_return",
        "Equal-weight benchmark": "benchmark_return",
    }

    print("\nPROVISIONAL WALK-FORWARD BACKTEST")
    print("Period:", out["date"].min().date(),
          "to", out["date"].max().date())
    print("Weeks tested:", len(out))
    print("\nReturns exclude transaction costs.")
    print("M2 historical revisions NOT YET corrected.\n")

    for name, col in strategies.items():
        returns = out[col]

        equity = (1 + returns).cumprod()
        cumulative = equity.iloc[-1] - 1
        drawdown = equity / equity.cummax() - 1

        win_rate = (
            returns > out["benchmark_return"]
        ).mean()

        print(name)
        print(f"  Cumulative return: {cumulative:.2%}")
        print(f"  Average weekly return: {returns.mean():.2%}")
        print(f"  Maximum drawdown: {drawdown.min():.2%}")

        if col != "benchmark_return":
            print(f"  Beat benchmark: {win_rate:.1%} of weeks")

        print()

    output_file = OUT / "backtest_2025_2026.csv"
    out.to_csv(output_file, index=False)

    print("Saved:", output_file)
    print("\nResearch only. Not a validated trading strategy.")


if __name__ == "__main__":
    main()
