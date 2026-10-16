
from pathlib import Path
from io import StringIO
import urllib.request

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"

DATA.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"

CSV_PATH = DATA / "m2_money_supply.csv"
GRAPH_PATH = OUTPUT / "m2_money_supply.png"


def download_m2():
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "TradingBotResearch/1.0"}
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        csv_text = response.read().decode("utf-8-sig")

    df = pd.read_csv(StringIO(csv_text))
    df.columns = ["date", "m2_billions"]

    df["date"] = pd.to_datetime(df["date"])
    df["m2_billions"] = pd.to_numeric(
        df["m2_billions"], errors="coerce"
    )

    return (
        df.dropna()
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


def main():
    print("Checking FRED for updated M2 data...")

    new_df = download_m2()

    if CSV_PATH.exists():
        old_df = pd.read_csv(CSV_PATH)
        old_df["date"] = pd.to_datetime(old_df["date"])

        old_df = old_df.sort_values("date").reset_index(drop=True)

        old_latest = old_df["date"].max()
        new_latest = new_df["date"].max()

        if new_df.equals(old_df):
            print("M2 is already up to date.")
            print(f"Latest observation: {new_latest.date()}")
            return

        if new_latest > old_latest:
            print("NEW M2 OBSERVATION DETECTED!")
        else:
            print("M2 historical values have changed (revision detected).")

    else:
        print("No previous M2 dataset found. Downloading full history.")

    new_df.to_csv(CSV_PATH, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(new_df["date"], new_df["m2_billions"])
    plt.title("US M2 Money Supply — Historical")
    plt.xlabel("Date")
    plt.ylabel("Billions of US Dollars")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=160)
    plt.close()

    print(f"Saved data: {CSV_PATH}")
    print(f"Saved graph: {GRAPH_PATH}")
    print(f"Monthly observations: {len(new_df)}")
    print(f"Latest M2 observation: {new_df['date'].max().date()}")


if __name__ == "__main__":
    main()
