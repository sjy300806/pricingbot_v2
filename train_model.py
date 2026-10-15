"""Refit frozen LightGBM cross-sectional model from fully matured historical labels."""
from pathlib import Path
import json
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from build_features import FEATURES,FUND,DATA,ROOT
OUT=ROOT/'models';OUT.mkdir(exist_ok=True)
MODEL_ARGS=dict(n_estimators=100,num_leaves=7,learning_rate=0.03,min_child_samples=100,reg_lambda=20,verbosity=-1,random_state=42,n_jobs=2)

def main():
    df=pd.read_csv(DATA/'weekly_features.csv',parse_dates=['date'])
    train=df[df.target_excess.notna()].copy()
    if train.date.nunique()<100:raise SystemExit('Insufficient matured weekly training history')
    cols=FEATURES+FUND
    model=LGBMRegressor(**MODEL_ARGS)
    model.fit(train[cols],train.target_excess)
    joblib.dump({'model':model,'features':cols,'trained_through':str(train.date.max().date()),'params':MODEL_ARGS},OUT/'ranker.joblib')
    (OUT/'model_info.json').write_text(json.dumps({'training_weeks':int(train.date.nunique()),'training_rows':len(train),'trained_through':str(train.date.max().date()),'features':cols,'parameters':MODEL_ARGS},indent=2))
    print(f'Saved model with {len(train):,} labeled observations through {train.date.max().date()}')
if __name__=='__main__':main()
