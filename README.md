# TradeVision (live NSE/BSE analyzer)

One service: FastAPI backend + the web page (served at `/`).

## Run on your computer
    pip install -r requirements.txt
    uvicorn main:app --reload
    # open http://localhost:8000

## Full 7000+ stock search
Download EQUITY_L.csv from nseindia.com (Market Data > Securities available for trading),
rename it to `stocks.csv`, put it next to main.py, restart. Without it, a small built-in list is used
(you can still type any NSE/BSE symbol and press Enter).

## Deploy (Render / Railway)
- Push this folder to a GitHub repo.
- Render: New > Web Service > Build `pip install -r requirements.txt`,
  Start `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Railway: it picks up the Procfile automatically.

Notes: yfinance is unofficial and delayed; if Yahoo blocks your host, retry later or move fetch() in main.py to a broker API.
Educational use only, not investment advice.
