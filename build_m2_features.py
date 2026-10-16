
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

INPUT_FILE = DATA / "m2_money_supply.csv"
OUTPUT_FILE = DATA / "m2_features.csv"


def main():
    # 1. Load monthly M2 money supply data
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df["m2_billions"] = pd.to_numeric(
        df["m2_billions"], errors="coerce"
    )

    df = (
        df.dropna(subset=["date", "m2_billions"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    # 2. Calculate M2 growth rates
    df["m2_growth_3m"] = df["m2_billions"].pct_change(3)
    df["m2_growth_6m"] = df["m2_billions"].pct_change(6)
    df["m2_growth_12m"] = df["m2_billions"].pct_change(12)

    # 3. Calculate M2 growth acceleration
    df["m2_acceleration"] = (
        df["m2_growth_3m"]
        - df["m2_growth_3m"].shift(3)
    )

    # 4. Conservative provisional publication delay
    # This is NOT an exact historical release calendar.
    # It prevents treating the observation date as release date.
    df["available_date"] = (
        df["date"] + pd.DateOffset(months=2)
    )

    # 5. Save the calculated indicators
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Monthly observations: {len(df)}")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()

