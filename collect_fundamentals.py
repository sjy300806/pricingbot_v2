"""Download SEC companyfacts for symbols in data/stocks_daily_adjusted.csv.
Set SEC_USER_AGENT in .env, e.g. 'StudentResearch contact@example.com'.
Caches raw responses in data/sec_raw/. Reruns reuse cache unless --refresh.
"""
import argparse
import json
import os
import time
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
CACHE=DATA/'sec_raw'
TICKERS_URL='https://www.sec.gov/files/company_tickers.json'


def get_json(url, headers):
    for attempt in range(5):
        try:
            resp=requests.get(url,headers=headers,timeout=45)
            if resp.status_code in (429,500,502,503,504):
                time.sleep(min(20,2**attempt+1));continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt==4: raise
            time.sleep(min(20,2**attempt+1))
    raise RuntimeError(f'Failed to download {url}')


def main():
    load_dotenv(ROOT/'.env')
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh',action='store_true',help='Redownload cached data')
    args=parser.parse_args()
    agent=os.getenv('SEC_USER_AGENT','').strip()
    if not agent or '@' not in agent:
        raise SystemExit('Set SEC_USER_AGENT="ResearchProject you@example.com" in .env (real contact email).')
    headers={'User-Agent':agent,'Accept-Encoding':'gzip, deflate','Accept':'application/json'}
    raw_path=DATA/'stocks_daily_adjusted.csv'
    if not raw_path.exists(): raise SystemExit(f'Missing {raw_path}. Run collect_stocks.py first.')
    symbols=sorted(pd.read_csv(raw_path,usecols=['symbol']).symbol.dropna().unique())
    CACHE.mkdir(parents=True,exist_ok=True)
    mapping_path=CACHE/'company_tickers.json'
    if args.refresh or not mapping_path.exists():
        mapping=get_json(TICKERS_URL,headers)
        mapping_path.write_text(json.dumps(mapping),encoding='utf-8')
        time.sleep(.2)
    else: mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
    lookup={entry['ticker'].upper():entry for entry in mapping.values()}
    manifest=[]
    for sym in symbols:
        entry=lookup.get(sym)
        if not entry:
            print(f'WARNING: No current SEC CIK mapping for {sym}; skipping')
            continue
        cik=int(entry['cik_str']); path=CACHE/f'{sym}.json'
        if args.refresh or not path.exists():
            print('Downloading',sym,'CIK',cik,flush=True)
            obj=get_json(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json',headers)
            path.write_text(json.dumps(obj),encoding='utf-8')
            time.sleep(.25)
        manifest.append({'symbol':sym,'cik':cik,'company':entry['title'],'raw_file':path.name})
    pd.DataFrame(manifest).to_csv(CACHE/'manifest.csv',index=False)
    print(f'Cached {len(manifest)}/{len(symbols)} SEC companyfacts files in {CACHE}')
    if len(manifest)!=len(symbols): print('WARNING: uncovered symbols; do not silently treat missing symbols as complete coverage.')

if __name__=='__main__': main()
