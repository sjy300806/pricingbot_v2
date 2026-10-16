
"""Build weekly stock rankings, keeping the latest unlabeled week."""

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

FEATURES = [
    "r5", "r20", "r60", "r120",
    "vol20", "vol60", "volume_ratio20",
    "ma20_gap", "ma60_gap", "range20",
    "relative_r20"
]

HORIZON = 5


def build(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

    raw = (
        raw.sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
    )

    calendar = pd.DatetimeIndex(sorted(raw["date"].unique()))

    dates = pd.DataFrame({"date": calendar})
    dates["week"] = dates["date"].dt.to_period("W-FRI")

    weekly = dates.groupby("week", sort=True)["date"].max()

    # Exclude the current, unfinished calendar week.
    today_ny = datetime.now(ZoneInfo("America/New_York"))
    current_week = pd.Timestamp(today_ny.date()).to_period("W-FRI")

    signal_dates = set(weekly[weekly.index < current_week])

    date_to_index = {date: i for i, date in enumerate(calendar)}

    raw["market_idx"] = raw["date"].map(date_to_index)
    raw["logclose"] = np.log(raw["close"])

    parts = []

    for symbol, g in raw.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()

        for n in (5, 20, 60, 120):
            g[f"r{n}"] = g["close"].pct_change(
                n, fill_method=None
            )

        daily = g["logclose"].diff()

        g["vol20"] = daily.rolling(20).std() * np.sqrt(5)
        g["vol60"] = daily.rolling(60).std() * np.sqrt(5)

        g["volume_ratio20"] = (
            g["volume"] / g["volume"].rolling(20).mean()
        )

        g["ma20_gap"] = (
            g["close"] / g["close"].rolling(20).mean() - 1
        )

        g["ma60_gap"] = (
            g["close"] / g["close"].rolling(60).mean() - 1
        )

        g["range20"] = (
            (g["high"] - g["low"]) / g["close"]
        ).rolling(20).mean()

        # Future returns use five global market sessions.
        future = g[["market_idx", "close"]].rename(
            columns={
                "market_idx": "future_idx",
                "close": "future_close"
            }
        )

        g["future_idx"] = g["market_idx"] + HORIZON

        g = g.merge(
            future,
            on="future_idx",
            how="left"
        )

        g["future_return"] = (
            g["future_close"] / g["close"] - 1
        )

        parts.append(g)

    full = pd.concat(parts, ignore_index=True)

    full["relative_r20"] = (
        full["r20"]
        - full.groupby("date")["r20"].transform("mean")
    )

    full = full[full["date"].isin(signal_dates)].copy()

    full = full.replace([np.inf, -np.inf], np.nan)
    full = full.dropna(subset=FEATURES)

    # Require all stocks to have valid technical features.
    symbols = raw["symbol"].nunique()
    counts = full.groupby("date")["symbol"].transform("nunique")

    full = full[counts == symbols].copy()

    # Only label weeks where every stock has a future return.
    complete_future = full.groupby("date")[
        "future_return"
    ].transform(lambda x: x.notna().all())

    full["target_excess"] = np.nan

    full.loc[complete_future, "target_excess"] = (
        full.loc[complete_future, "future_return"]
        - full.loc[complete_future].groupby("date")[
            "future_return"
        ].transform("mean")
    )

    full = full.sort_values(["date", "symbol"])

    return full[
        ["date", "symbol"]
        + FEATURES
        + ["future_return", "target_excess"]
    ].reset_index(drop=True)


def main():
    raw = pd.read_csv(DATA / "stocks_daily_adjusted.csv")
    ds = build(raw)

    if ds.empty:
        raise SystemExit("No complete weekly samples.")

    out = DATA / "weekly_ranking_dataset.csv"
    ds.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Total observations: {len(ds):,}")
    print(f"Total weeks: {ds['date'].nunique()}")
    print(f"Stocks: {ds['symbol'].nunique()}")
    print(f"Latest week: {ds['date'].max().date()}")
    print(
        "Labeled training weeks:",
        ds.loc[
            ds["target_excess"].notna(), "date"
        ].nunique()
    )


if __name__ == "__main__":
    main()
