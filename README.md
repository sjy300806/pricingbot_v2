The bot collects the latest daily price and trading volume data for 25 selected stocks using the Alpaca API, while retrieving publicly available company financial information from SEC EDGAR. It then processes these datasets to generate technical indicators, such as historical returns, volatility, and momentum, alongside fundamental indicators, such as revenue growth, profit margins, and financial ratios. These features are combined into a dataset used to train a LightGBM machine learning model on historical stock performance. Once trained, the model evaluates the latest market data and predicts each stock's relative performance over the next five trading days. Finally, the bot ranks all 25 stocks according to their predicted returns and generates a CSV file containing the rankings and the five highest-ranked stocks as potential investment candidates.

Files:

1, collect_stocks.py            
Download adjusted daily OHLCV for 25 fixed US stocks.

2, collect_fundamentals.py      
Download SEC companyfacts JSON into data/sec_raw/.

3, build_fundamental_features.py 

Produce financial features using filing availability.

4, build_ranking_dataset.py     
Make weekly technical features and historical targets.

5, merge_fundamentals.py      
Join available financial data to weekly stock rows.

6, .env     
Local credentials

7, main.py

Runs the entire pipeline

8, predict.py

Loads the model and generates the latest rankings

9, buld_features.py

Creates technical features and combines fundamentals

How to use

1, Install Python 3.11 or 3.12 from python.org.

2, Extract the project and set up its environment

Open PowerShell in the extracted folder and run this script to create the environment:

then install all the dependencies:

py -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

3, Configure the APIs

Open the .env file, then enter your own Alpaca API credentials and real SEC contact email.
API keys should never be shared with anyone as part of your personal .env.

4, Collect and prepare historical data

Run:

.\.venv\Scripts\python.exe collect_stocks.py

.\.venv\Scripts\python.exe collect_fundamentals.py

.\.venv\Scripts\python.exe build_fundamental_features.py

.\.venv\Scripts\python.exe build_ranking_dataset.py

.\.venv\Scripts\python.exe merge_fundamentals.py

5, Run the bot

In powershell:

.\.venv\Scripts\python.exe main.py


What the bot generates:

your folder/

├── data/

│   ├── stocks_daily_adjusted.csv

│   ├── fundamental_features.csv

│   └── weekly_features.csv

├── models/

│   ├── ranker.joblib

│   └── model_info.json

└── output/

   └── rankings_YYYYMMDD.csv
    

