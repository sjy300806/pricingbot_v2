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

How to use on macOS (Mac mini)

1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ .
   Open Terminal (or VS Code > Terminal > New Terminal) in this project folder.

2. Create and activate a virtual environment:

   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt

   If your installed Python command is python3 rather than python3.12,
   verify `python3 --version` shows 3.11 or 3.12, then use
   `python3 -m venv .venv` instead.

3. Edit the included .env file to enter your own Alpaca API key,
   Alpaca secret key, and real SEC contact email. Do not share .env.
   Finder hides filenames beginning with a dot by default; in VS Code
   it should appear in the Explorer. On Mac Finder, press Command+Shift+.
   to toggle hidden files.

4. To run the entire active pipeline (downloads, feature generation,
   training, and latest ranking):

   python main.py

   Or run its individual stages in this exact order:

   python collect_stocks.py
   python collect_fundamentals.py --refresh
   python build_fundamental_features.py
   python build_features.py
   python train_model.py
   python predict.py

   Note: build_ranking_dataset.py and merge_fundamentals.py are alternate
   standalone dataset-building scripts and are not called by main.py.
   Do not substitute them for build_features.py in the active pipeline.

5. Outputs:
   data/stocks_daily_adjusted.csv
   data/fundamental_features.csv
   data/weekly_features.csv
   models/ranker.joblib
   models/model_info.json
   output/rankings_YYYYMMDD.csv

Notes:
- The first run needs internet access and valid Alpaca credentials.
- The IEX feed is the default; availability and historical coverage depend
  on the Alpaca account and data plan.
- The program creates research rankings; it does not place trades.
- The code is otherwise unchanged from the uploaded project.
