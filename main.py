import csv
import time
from pathlib import Path

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from indicators import analyze, analyze_intraday, fundamentals_view

BASE = Path(__file__).parent

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


def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    if len(_cache) > 400:
        _cache.clear()
    _cache[key] = (time.time(), val)
    return val


def clean(symbol):
    return symbol.strip().upper().replace(".NS", "").replace(".BO", "")


def hist(symbol, period, interval="1d", min_rows=60):
    for suffix in (".NS", ".BO"):
        df = yf.Ticker(symbol + suffix).history(period=period, interval=interval, auto_adjust=True)
        df = df.dropna(subset=["Close"])
        if len(df) >= min_rows:
            if interval == "1d" and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            elif df.index.tz is not None:
                df.index = df.index.tz_convert("Asia/Kolkata")
            return df, suffix
    return None, None


def guarded(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(502, "Data source did not respond. Try again in a minute.")


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


@app.get("/api/analyze")
def analyze_symbol(symbol: str = Query(..., max_length=20)):
    s = clean(symbol)

    def work():
        df, suffix = hist(s, "2y")
        if df is None:
            raise HTTPException(404, f"No price data found for {s}.")
        res = analyze(df)
        res.update(symbol=s, name=STOCKS.get(s, s), exchange="NSE" if suffix == ".NS" else "BSE")
        return res

    return guarded(lambda: cached(("d", s), 300, work))


@app.get("/api/intraday")
def intraday(symbol: str = Query(..., max_length=20)):
    s = clean(symbol)

    def work():
        df, _ = hist(s, "5d", "15m", min_rows=30)
        if df is None:
            raise HTTPException(404, f"No 15-minute data found for {s}.")
        return analyze_intraday(df)

    return guarded(lambda: cached(("i", s), 120, work))


@app.get("/api/fundamentals")
def fundamentals(symbol: str = Query(..., max_length=20)):
    s = clean(symbol)

    def work():
        df, suffix = hist(s, "5y")
        if df is None:
            raise HTTPException(404, f"No price data found for {s}.")
        try:
            info = yf.Ticker(s + suffix).info or {}
        except Exception:
            info = {}
        try:
            b = yf.Ticker("^NSEI").history(period="5y", interval="1d")["Close"].dropna()
        except Exception:
            b = None
        res = fundamentals_view(info, df, b)
        res.update(symbol=s, name=STOCKS.get(s) or info.get("longName") or s)
        return res

    return guarded(lambda: cached(("f", s), 3600, work))


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")
