"""As-of join SEC filing features onto original weekly ranking observations.
Preserves original rows; missing fundamentals stay missing.
"""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
FUND_FEATURES=['revenue_growth_yoy','net_margin','debt_to_assets','operating_cashflow_margin']

def merge(weekly,updates):
    weekly=weekly.copy(); updates=updates.copy()
    weekly['date']=pd.to_datetime(weekly.date).dt.normalize()
    updates['available_date']=pd.to_datetime(updates.available_date).dt.normalize()
    result=[]
    for symbol,grp in weekly.groupby('symbol',sort=False):
        history=updates[updates.symbol==symbol].sort_values('available_date')
        grp=grp.sort_values('date')
        if history.empty:
            part=grp.copy()
            for f in FUND_FEATURES: part[f]=np.nan
            part['available_date']=pd.NaT
        else:
            part=pd.merge_asof(grp,history[['available_date']+FUND_FEATURES],left_on='date',right_on='available_date',direction='backward')
        result.append(part)
    out=pd.concat(result,ignore_index=True).sort_values(['date','symbol']).reset_index(drop=True)
    assert len(out)==len(weekly)
    assert (out.available_date.isna() | (out.available_date<=out.date)).all(), 'Future filing leaked into training!'
    out['days_since_filing']=(out.date-out.available_date).dt.days
    return out

def main():
    weekly=pd.read_csv(DATA/'weekly_ranking_dataset.csv')
    updates=pd.read_csv(DATA/'fundamental_features.csv')
    out=merge(weekly,updates)
    path=DATA/'weekly_ranking_with_fundamentals.csv';out.to_csv(path,index=False)
    print('Saved',len(out),'weekly stock rows to',path)
    print('Coverage per SEC feature (% nonmissing):')
    print((out[FUND_FEATURES].notna().mean()*100).round(1).to_string())
    print('Historical values joined according to conservative post-filing available dates.')
if __name__=='__main__':main()
