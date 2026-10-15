"""Training + latest inference features. Latest row has no future label and is retained."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
FEATURES=['r5','r20','r60','r120','vol20','vol60','volume_ratio20','ma20_gap','ma60_gap','range20','relative_r20']
FUND=['revenue_growth_yoy','net_margin','debt_to_assets','days_since_filing']
TICKERS=25; HORIZON=5

def build(prices, fund, now_ny=None):
    raw=prices.copy();raw['date']=pd.to_datetime(raw.date).dt.normalize()
    raw=raw.sort_values(['symbol','date']).drop_duplicates(['symbol','date'],keep='last')
    calendar=pd.DatetimeIndex(sorted(raw.date.unique()))
    dates=pd.DataFrame({'date':calendar});dates['week']=dates.date.dt.to_period('W-FRI')
    weekly=dates.groupby('week',sort=True).date.max()
    now_ny=now_ny or datetime.now(ZoneInfo('America/New_York'))
    # Do not use an in-progress calendar week: its 'last trading day' may change.
    current_week=pd.Timestamp(now_ny.date()).to_period('W-FRI')
    signal_dates=set(weekly[weekly.index < current_week])
    lookup={d:i for i,d in enumerate(calendar)}
    raw['market_idx']=raw.date.map(lookup);raw['logclose']=np.log(raw.close)
    pieces=[]
    for sym,g in raw.groupby('symbol',sort=False):
        g=g.sort_values('date').copy()
        for n in (5,20,60,120):g[f'r{n}']=g.close.pct_change(n,fill_method=None)
        dr=g.logclose.diff()
        for n in (20,60):g[f'vol{n}']=dr.rolling(n).std()*np.sqrt(5)
        g['volume_ratio20']=g.volume/g.volume.rolling(20).mean()
        for n in (20,60):g[f'ma{n}_gap']=g.close/g.close.rolling(n).mean()-1
        g['range20']=((g.high-g.low)/g.close).rolling(20).mean()
        # Global sessions: no phantom five-day return through a missing stock bar.
        fut=g[['market_idx','close']].rename(columns={'market_idx':'future_idx','close':'future_close'})
        g['future_idx']=g.market_idx+HORIZON
        g=g.merge(fut,on='future_idx',how='left')
        g['future_return']=g.future_close/g.close-1
        pieces.append(g)
    full=pd.concat(pieces,ignore_index=True)
    full['relative_r20']=full.r20-full.groupby('date').r20.transform('mean')
    full=full[full.date.isin(signal_dates)].replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES).copy()
    counts=full.groupby('date').symbol.transform('nunique')
    full=full[counts==TICKERS].copy()
    # Requiring full universe on a weekly date avoids comparing incomparable rankings.
    if full.empty: raise SystemExit('No weekly observations with all 25 stocks')
    f=fund.copy();f['available_date']=pd.to_datetime(f.available_date).dt.normalize()
    joined=[]
    for sym,g in full.groupby('symbol',sort=False):
        history=f[f.symbol==sym].sort_values('available_date')
        g=g.sort_values('date')
        if history.empty:
            g=g.copy();g['available_date']=pd.NaT
            for col in FUND[:3]:g[col]=np.nan
            joined.append(g)
        else:
            joined.append(pd.merge_asof(g,history[['available_date']+FUND[:3]],left_on='date',right_on='available_date',direction='backward'))
    out=pd.concat(joined,ignore_index=True).sort_values(['date','symbol']).reset_index(drop=True)
    assert (out.available_date.isna() | (out.available_date<=out.date)).all()
    out['days_since_filing']=(out.date-out.available_date).dt.days
    # A training week is valid only if every symbol's future five-session close exists.
    eligible=out.groupby('date').future_return.transform(lambda s:s.notna().all())
    out['target_excess']=np.nan
    out.loc[eligible,'target_excess']=out.loc[eligible,'future_return']-out.loc[eligible].groupby('date').future_return.transform('mean')
    cols=['date','symbol']+FEATURES+FUND+['available_date','future_return','target_excess']
    return out[cols]

def main():
    raw=pd.read_csv(DATA/'stocks_daily_adjusted.csv')
    fund=pd.read_csv(DATA/'fundamental_features.csv')
    out=build(raw,fund)
    out.to_csv(DATA/'weekly_features.csv',index=False)
    print(f'Wrote {len(out):,} stock-weeks; latest: {out.date.max().date()}; labeled weeks: {out.loc[out.target_excess.notna(),"date"].nunique()}')
if __name__=='__main__':main()
