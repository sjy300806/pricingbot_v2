"""Incrementally refresh adjusted daily Alpaca stock bars; exclude today's possibly unfinished bar."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, time
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
TICKERS=['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','AVGO','NFLX','JPM','BAC','GS','V','MA','UNH','JNJ','LLY','PFE','XOM','CVX','CAT','HD','COST','WMT']
OUT=DATA/'stocks_daily_adjusted.csv'

def main():
    load_dotenv(ROOT/'.env',override=True)
    key=os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
    secret=os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
    if not key or not secret: raise SystemExit('Provide Alpaca credentials in .env')
    feed_name=os.getenv('ALPACA_FEED','iex').lower()
    if feed_name not in ('iex','sip'): raise SystemExit('ALPACA_FEED must be iex or sip')
    feed=DataFeed.IEX if feed_name=='iex' else DataFeed.SIP
    client=StockHistoricalDataClient(key,secret)
    existing=pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    if not existing.empty:
        required={'date','symbol','open','high','low','close','volume'}
        if not required.issubset(existing.columns): raise SystemExit('Invalid existing daily CSV; please check data file')
        existing['date']=pd.to_datetime(existing['date']).dt.strftime('%Y-%m-%d')
    # The exclusive UTC end at today's midnight guarantees only earlier calendar-day bars.
    end=datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    start_default=datetime(2017,1,1,tzinfo=timezone.utc)
    updates=[]
    for i,sym in enumerate(TICKERS,1):
        prev=existing[existing.symbol==sym] if not existing.empty else pd.DataFrame()
        start=start_default if prev.empty else max(start_default,pd.Timestamp(prev.date.max()).to_pydatetime().replace(tzinfo=timezone.utc)-timedelta(days=14))
        if start>=end: continue
        print(f'[{i}/{len(TICKERS)}] Download {sym} from {start.date()} to {end.date()} (exclusive)',flush=True)
        request=StockBarsRequest(symbol_or_symbols=sym,timeframe=TimeFrame.Day,start=start,end=end,adjustment=Adjustment.ALL,feed=feed)
        frame=client.get_stock_bars(request).df.reset_index()
        if frame.empty: raise SystemExit(f'No Alpaca bars for {sym}; refusing incomplete refresh')
        frame['symbol']=sym
        frame['date']=pd.to_datetime(frame['timestamp'],utc=True).dt.strftime('%Y-%m-%d')
        updates.append(frame[['symbol','date','open','high','low','close','volume']])
        time.sleep(.15)
    joined=pd.concat(([existing] if not existing.empty else [])+updates,ignore_index=True) if updates else existing
    if joined.empty: raise SystemExit('No data available')
    joined=joined.sort_values(['symbol','date']).drop_duplicates(['symbol','date'],keep='last')
    if set(joined.symbol.unique())!=set(TICKERS):raise SystemExit('Incomplete symbols; refusing to save')
    if (joined[['open','high','low','close']]<=0).any().any():raise SystemExit('Nonpositive prices detected')
    if ((joined.high < joined[['open','close','low']].max(axis=1)) | (joined.low > joined[['open','close','high']].min(axis=1))).any():raise SystemExit('Inconsistent OHLC detected')
    # Do not mix historical feeds: changing ALPACA_FEED requires rebuilding the CSV.
    temp=OUT.with_suffix('.tmp'); joined.to_csv(temp,index=False);temp.replace(OUT)
    print(f'Saved {len(joined):,} daily bars through {joined.date.max()} to {OUT}')
if __name__=='__main__': main()
