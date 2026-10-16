
"""Rank the latest 25 stocks using technical, SEC and M2 indicators.
Research only. Does not execute trades.
"""

from pathlib import Path
import pandas as pd
import joblib

from build_features import DATA, ROOT

OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def main():
    # 1. Load the M2-trained model
    model_path = ROOT / "models" / "ranker_m2.joblib"
    payload = joblib.load(model_path)

    # 2. Load the combined technical + SEC + M2 dataset
    data_path = DATA / "weekly_ranking_with_fundamentals_m2.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 3. Select the latest completed weekly snapshot
    latest = df["date"].max()
    snapshot = df[df["date"] == latest].copy()

    # 4. Check data freshness
    today = pd.Timestamp.now(
        tz="America/New_York"
    ).tz_localize(None).normalize()

    if (today - latest).days > 10:
        raise SystemExit(
            f"Latest weekly snapshot ({latest.date()}) is stale. "
            "Update stock data before predicting."
        )

    # 5. Check that all 25 stocks are present
    if len(snapshot) != 25 or snapshot["symbol"].nunique() != 25:
        raise SystemExit(
            "Latest weekly snapshot must contain 25 unique stocks."
        )

    # 6. Check model features
    model = payload["model"]
    cols = payload["features"]

    missing = [c for c in cols if c not in snapshot.columns]

    if missing:
        raise SystemExit(f"Missing model features: {missing}")

    # 7. Predict relative stock performance
    snapshot["score"] = model.predict(snapshot[cols])

    # 8. Rank stocks from highest to lowest score
    snapshot = snapshot.sort_values(
        ["score", "symbol"],
        ascending=[False, True]
    ).reset_index(drop=True)

    snapshot["rank"] = snapshot.index + 1

    # 9. Prepare the results
    result = snapshot[
        [
            "date",
            "rank",
            "symbol",
            "score",
            "r20",
            "vol20",
            "days_since_filing",
            "m2_growth_3m",
            "m2_growth_6m",
            "m2_growth_12m",
            "m2_acceleration",
        ]
    ].copy()

    result = result.rename(
        columns={"score": "predicted_relative_return"}
    )

    # 10. Save the M2 rankings separately
    filename = OUT / f"rankings_m2_{latest:%Y%m%d}.csv"
    result.to_csv(filename, index=False)

    print(
        f"\nM2 STOCK RANKINGS — {latest.date()}"
    )
    print(
        "\nScores estimate relative performance, "
        "not guaranteed returns.\n"
    )

    print(
        result.head(10).to_string(
            index=False,
            float_format=lambda n: f"{n:.4f}"
        )
    )

    print(
        "\nTOP FIVE:",
        ", ".join(result.head(5)["symbol"])
    )

    print(f"\nSaved: {filename}")
    print("No trades executed. Research only.")


if __name__ == "__main__":
    main()
