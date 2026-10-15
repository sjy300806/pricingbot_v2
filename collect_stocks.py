"""Download adjusted daily OHLCV bars for a fixed 25-stock research universe.
Run from any directory; outputs beside this script.
"""
from pathlib import Path
from datetime import datetime, timezone
import os
import time
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
load_dotenv(ROOT / '.env')

# Fixed CURRENT liquid-stock universe: engineering prototype, NOT survivorship-bias-free.
TICKERS = [
    'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','AVGO','NFLX',
    'JPM','BAC','GS','V','MA','UNH','JNJ','LLY','PFE','XOM',
    'CVX','CAT','HD','COST','WMT'
]
START = datetime(2017, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 10, 1, tzinfo=timezone.utc) # exclusive; 2026-09-30 last full day


def main():
    key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
    secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
    if not key or not secret:
        raise SystemExit('Missing Alpaca keys in .env next to this script.')
    client = StockHistoricalDataClient(key, secret)
    feed_name = os.getenv('ALPACA_FEED', 'iex').lower()
    feed = {'iex': DataFeed.IEX, 'sip': DataFeed.SIP}.get(feed_name)
    if feed is None:
        raise SystemExit('ALPACA_FEED must be iex or sip')
    all_rows = []
    for i, symbol in enumerate(TICKERS, 1):
        print(f'[{i}/{len(TICKERS)}] {symbol}', flush=True)
        request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                   start=START, end=END, adjustment=Adjustment.ALL, feed=feed)
        try:
            bars = client.get_stock_bars(request).df.reset_index()
        except Exception as exc:
            raise RuntimeError(f'Error downloading {symbol}. Check subscription/feed and API limits.') from exc
        if bars.empty:
            print(f'  WARNING: no data for {symbol}')
            continue
        bars['symbol'] = symbol
        bars['date'] = pd.to_datetime(bars['timestamp'], utc=True).dt.strftime('%Y-%m-%d')
        all_rows.append(bars[['symbol','date','open','high','low','close','volume']])
        time.sleep(0.15)
    if not all_rows:
        raise SystemExit('No bars downloaded')
    out = pd.concat(all_rows, ignore_index=True)
    out = out.drop_duplicates(['symbol','date']).sort_values(['symbol','date'])
    path = DATA / 'stocks_daily_adjusted.csv'
    out.to_csv(path, index=False)
    print(f'Saved {len(out):,} rows from {out.symbol.nunique()} stocks to {path}')
    print('WARNING: fixed present-day universe has survivorship/selection bias.')

if __name__ == '__main__':
    main()
