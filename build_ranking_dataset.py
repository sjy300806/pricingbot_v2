"""Create weekly cross-sectional stock ranking observations (one row per stock per week)."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FEATURES = ['r5','r20','r60','r120','vol20','vol60','volume_ratio20','ma20_gap','ma60_gap','range20','relative_r20']
HORIZON = 5


def build(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw.sort_values(['symbol','date']).drop_duplicates(['symbol','date'])
    # Weekly signals: last available trading date of each calendar week.
    calendar = sorted(raw.date.unique())
    dates = pd.DataFrame({'date':calendar})
    dates['week'] = dates.date.dt.to_period('W-FRI')
    signal_dates = set(dates.groupby('week').date.max())
    # Market index of trading days; needed for exact five SESSION horizon.
    date_to_index = {date:i for i,date in enumerate(calendar)}
    raw['market_idx'] = raw.date.map(date_to_index)
    raw['week'] = raw.date.dt.to_period('W-FRI')
    raw['logclose'] = np.log(raw.close)
    raw['logvol'] = np.log1p(raw.volume)
    parts=[]
    for symbol, g in raw.groupby('symbol', sort=False):
        g=g.sort_values('date').copy()
        for n in (5,20,60,120):
            g[f'r{n}'] = g.close.pct_change(n, fill_method=None)
        daily = g.logclose.diff()
        g['vol20'] = daily.rolling(20).std() * np.sqrt(5)
        g['vol60'] = daily.rolling(60).std() * np.sqrt(5)
        g['volume_ratio20'] = g.volume / g.volume.rolling(20).mean()
        g['ma20_gap'] = g.close/g.close.rolling(20).mean()-1
        g['ma60_gap'] = g.close/g.close.rolling(60).mean()-1
        g['range20'] = ((g.high-g.low)/g.close).rolling(20).mean()
        # Map future 5 GLOBAL trading sessions, not just shift(-5), so suspended/missing bars never masquerade as 5-day returns.
        future = g[['market_idx','close']].rename(columns={'market_idx':'future_idx','close':'future_close'})
        g['future_idx'] = g.market_idx + HORIZON
        g = g.merge(future, on='future_idx', how='left')
        g['future_return'] = g.future_close/g.close-1
        parts.append(g)
    full=pd.concat(parts, ignore_index=True)
    # Relative momentum to contemporaneous cross-section (known at close).
    full['relative_r20'] = full['r20'] - full.groupby('date')['r20'].transform('mean')
    full = full[full.date.isin(signal_dates)].copy()
    full = full.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+['future_return'])
    # Only complete cross-sectional weeks so every evaluated date compares the same symbols.
    symbols = raw.symbol.nunique()
    count = full.groupby('date').symbol.transform('nunique')
    full = full[count == symbols].copy()
    full['target_excess'] = full.future_return - full.groupby('date').future_return.transform('mean')
    full = full.sort_values(['date','symbol'])
    return full[['date','symbol']+FEATURES+['future_return','target_excess']].reset_index(drop=True)


def main():
    raw=pd.read_csv(DATA/'stocks_daily_adjusted.csv')
    ds=build(raw)
    if ds.empty: raise SystemExit('No complete weekly samples. Check missing bars / universe.')
    out=DATA/'weekly_ranking_dataset.csv'
    ds.to_csv(out,index=False)
    print(f'Wrote {len(ds):,} observations, {ds.date.nunique()} weeks, {ds.symbol.nunique()} stocks -> {out}')
    print('First/last week:',ds.date.min().date(),ds.date.max().date())

if __name__=='__main__': main()
