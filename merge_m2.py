
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

STOCK_FILE = DATA / "weekly_ranking_with_fundamentals.csv"
M2_FILE = DATA / "m2_features.csv"
OUTPUT_FILE = DATA / "weekly_ranking_with_fundamentals_m2.csv"

M2_FEATURES = [
    "m2_billions",
    "m2_growth_3m",
    "m2_growth_6m",
    "m2_growth_12m",
    "m2_acceleration",
]


def main():
    # 1. Load stock + SEC data and M2 data
    stocks = pd.read_csv(STOCK_FILE)
    m2 = pd.read_csv(M2_FILE)

    # 2. Convert dates into the correct format
    stocks["date"] = pd.to_datetime(stocks["date"])
    m2["available_date"] = pd.to_datetime(m2["available_date"])

    # 3. Give the M2 availability date a unique name
    m2 = m2.rename(
        columns={"available_date": "m2_available_date"}
    )

    # 4. Sort both datasets by date
    stocks = stocks.sort_values("date").reset_index(drop=True)
    m2 = m2.sort_values("m2_available_date").reset_index(drop=True)

    # 5. Match each stock observation to the latest available M2 data
    combined = pd.merge_asof(
        stocks,
        m2[["m2_available_date"] + M2_FEATURES],
        left_on="date",
        right_on="m2_available_date",
        direction="backward",
    )

    # 6. Check that no future M2 data was used
    valid = (
        combined["m2_available_date"].isna()
        | (combined["m2_available_date"] <= combined["date"])
    )
    assert valid.all(), "ERROR: Future M2 data detected!"

    # 7. Check that no stock observations were lost
    assert len(combined) == len(stocks), "Stock rows were lost!"

    # 8. Save the combined dataset
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total stock observations: {len(combined)}")
    print(
        "M2 coverage:",
        round(combined["m2_billions"].notna().mean() * 100, 1),
        "%"
    )
    print(combined[["date", "symbol"] + M2_FEATURES].tail(5))


if __name__ == "__main__":
    main()

