import csv
import time
from pathlib import Path

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from indicators import analyze

BASE = Path(__file__).parent
CACHE_TTL = 300  # seconds

app = FastAPI(title="TradeVision API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

FALLBACK = {
    "CDSL": "Central Depository Services (India)", "JIOFIN": "Jio Financial Services",
    "TCS": "Tata Consultancy Services", "RELIANCE": "Reliance Industries", "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank", "SBIN": "State Bank of India",
    "TATAMOTORS": "Tata Motors", "TATASTEEL": "Tata Steel", "ITC": "ITC", "LT": "Larsen & Toubro",
    "BAJFINANCE": "Bajaj Finance", "ADANIENT": "Adani Enterprises", "WIPRO": "Wipro",
    "STYLEBAAZA": "Baazar Style Retail", "BSE": "BSE Ltd", "ZOMATO": "Zomato",
    "HAL": "Hindustan Aeronautics", "IRFC": "Indian Railway Finance Corporation",
}


def load_stocks():
    f = BASE / "stocks.csv"  # NSE's EQUITY_L.csv, renamed
    if not f.exists():
        return dict(FALLBACK)
    out = {}
    with open(f, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip().upper(): (v or "").strip() for k, v in row.items() if k}
            if row.get("SYMBOL"):
                out[row["SYMBOL"]] = row.get("NAME OF COMPANY", "")
    return out or dict(FALLBACK)


STOCKS = load_stocks()
_cache = {}


@app.get("/api/search")
def search(q: str = Query("", max_length=40)):
    q = q.strip().upper()
    if not q:
        return []
    ranked = []
    for sym, name in STOCKS.items():
        n = name.upper()
        if sym == q: r = 0
        elif sym.startswith(q): r = 1
        elif q in sym: r = 2
        elif q in n: r = 3
        else: continue
        ranked.append((r, sym, name))
    ranked.sort()
    return [{"symbol": s, "name": n} for _, s, n in ranked[:8]]


def fetch(symbol):
    for suffix in (".NS", ".BO"):
        df = yf.Ticker(symbol + suffix).history(period="2y", interval="1d", auto_adjust=True)
        df = df.dropna(subset=["Close"])
        if len(df) >= 60:
            return df.tz_localize(None), suffix
    return None, None


@app.get("/api/analyze")
def analyze_symbol(symbol: str = Query(..., max_length=20)):
    s = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
    hit = _cache.get(s)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    try:
        df, suffix = fetch(s)
    except Exception:
        raise HTTPException(502, "Data source did not respond. Try again in a minute.")
    if df is None:
        raise HTTPException(404, f"No price data found for {s}.")
    res = analyze(df)
    res.update(symbol=s, name=STOCKS.get(s, s), exchange="NSE" if suffix == ".NS" else "BSE")
    _cache[s] = (time.time(), res)
    return res


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")
