"""Score the latest completed weekly snapshot. Does not execute trades."""
from pathlib import Path
import pandas as pd
import joblib
from build_features import DATA,ROOT
OUT=ROOT/'output';OUT.mkdir(exist_ok=True)

def main():
    payload=joblib.load(ROOT/'models'/'ranker.joblib') # Load only a model created locally / from a trusted source.
    df=pd.read_csv(DATA/'weekly_features.csv',parse_dates=['date'])
    latest=df.date.max(); snapshot=df[df.date==latest].copy()
    if (pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()-latest).days>10:
        raise SystemExit(f'Latest weekly snapshot ({latest.date()}) is stale. Check Alpaca downloads before issuing signals.')
    if len(snapshot)!=25:raise SystemExit('Latest weekly snapshot has fewer than 25 stocks')
    model=payload['model']; cols=payload['features']
    if not set(cols).issubset(snapshot.columns): raise SystemExit('Feature schema mismatch')
    snapshot['score']=model.predict(snapshot[cols])
    snapshot=snapshot.sort_values(['score','symbol'],ascending=[False,True]).reset_index(drop=True)
    snapshot['rank']=snapshot.index+1
    result=snapshot[['date','rank','symbol','score','r20','vol20','days_since_filing']].copy()
    result.rename(columns={'score':'predicted_relative_return'},inplace=True)
    filename=OUT/f'rankings_{latest:%Y%m%d}.csv'
    result.to_csv(filename,index=False)
    print(f'\nResearch rankings at completed week {latest.date()} (scores are estimates, not guaranteed returns)')
    print(result.head(10).to_string(index=False,float_format=lambda n:f'{n:.4f}'))
    print(f'\nTOP FIVE: {", ".join(result.head(5).symbol)}\nSaved {filename}')
    print('No orders placed. Research-only, fixed retrospectively selected 25-stock universe.')
if __name__=='__main__':main()
