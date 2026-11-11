import numpy as np
import pandas as pd


# ---------- basic indicators ----------
def _rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    atr = _atr(h, l, c, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def _supertrend(h, l, c, n=10, m=3.0):
    atr = _atr(h, l, c, n)
    hl2 = (h + l) / 2
    ub, lb, cc = (hl2 + m * atr).to_numpy(), (hl2 - m * atr).to_numpy(), c.to_numpy()
    N = len(cc)
    fub, flb, dr = ub.copy(), lb.copy(), np.ones(N, dtype=int)
    for i in range(1, N):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or cc[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or cc[i - 1] < flb[i - 1]) else flb[i - 1]
        if dr[i - 1] == 1:
            dr[i] = -1 if cc[i] < flb[i] else 1
        else:
            dr[i] = 1 if cc[i] > fub[i] else -1
    line = np.where(dr == 1, flb, fub)
    return pd.Series(line, index=c.index), pd.Series(dr, index=c.index)


def _f(x, d=2):
    try:
        x = float(x)
        return None if np.isnan(x) else round(x, d)
    except Exception:
        return None


def _lst(s, d=2):
    return [_f(x, d) for x in s]


def _label(score):
    return ("STRONG BULLISH" if score >= 0.5 else "BULLISH" if score >= 0.2 else
            "NEUTRAL" if score > -0.2 else "BEARISH" if score > -0.5 else "STRONG BEARISH")


def _count(sig):
    nb = sum(s["verdict"] == "bullish" for s in sig)
    nr = sum(s["verdict"] == "bearish" for s in sig)
    return {"bullish": nb, "neutral": len(sig) - nb - nr, "bearish": nr}, (nb - nr) / len(sig)


# ---------- trade plan ----------
def _plan(price, atr, score, stop_mult=1.5):
    if atr is None or np.isnan(atr) or atr <= 0:
        return {"side": "WAIT"}
    risk = stop_mult * atr
    if score >= 0.2:
        side, stop, t1, t2 = "LONG", price - risk, price + 2 * risk, price + 3 * risk
    elif score <= -0.2:
        side, stop, t1, t2 = "SHORT", price + risk, price - 2 * risk, price - 3 * risk
    else:
        return {"side": "WAIT", "atr": _f(atr)}
    return {"side": side, "entry": _f(price), "stop": _f(stop), "t1": _f(t1), "t2": _f(t2),
            "risk": _f(risk), "atr": _f(atr)}


# ---------- candlestick patterns ----------
def _patterns(df):
    if len(df) < 8:
        return []
    o, h, l, c = (df[x].to_numpy() for x in ("Open", "High", "Low", "Close"))
    body, rng = abs(c[-1] - o[-1]), h[-1] - l[-1]
    out = []
    if rng > 0:
        upper, lower = h[-1] - max(c[-1], o[-1]), min(c[-1], o[-1]) - l[-1]
        down, up = c[-2] < c[-7], c[-2] > c[-7]
        if body <= 0.1 * rng:
            out.append(("Doji (indecision)", "neutral"))
        elif body >= 0.9 * rng:
            out.append(("Strong green candle" if c[-1] > o[-1] else "Strong red candle",
                        "bullish" if c[-1] > o[-1] else "bearish"))
        if body > 0 and lower >= 2 * body and upper <= 0.5 * body and down:
            out.append(("Hammer (possible bottom)", "bullish"))
        if body > 0 and upper >= 2 * body and lower <= 0.5 * body and up:
            out.append(("Shooting star (possible top)", "bearish"))
    if c[-2] < o[-2] and c[-1] > o[-1] and c[-1] >= o[-2] and o[-1] <= c[-2]:
        out.append(("Bullish engulfing", "bullish"))
    if c[-2] > o[-2] and c[-1] < o[-1] and c[-1] <= o[-2] and o[-1] >= c[-2]:
        out.append(("Bearish engulfing", "bearish"))
    if h[-1] < h[-2] and l[-1] > l[-2]:
        out.append(("Inside bar (breakout pending)", "neutral"))
    if not out:
        out.append(("No clear pattern", "neutral"))
    return [{"name": n, "verdict": v} for n, v in out]


# ---------- weekly trend ----------
def _weekly(df):
    w = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min",
                              "Close": "last", "Volume": "sum"}).dropna()
    if len(w) < 30:
        return None
    c = w["Close"]
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    checks = [
        ("Price above 10-week average", c.iloc[-1] > c.rolling(10).mean().iloc[-1]),
        ("Weekly RSI above 50", _rsi(c).iloc[-1] > 50),
        ("Weekly MACD above signal", macd.iloc[-1] > sig.iloc[-1]),
    ]
    n = sum(bool(x) for _, x in checks)
    trend = ["BEARISH", "LEANING BEARISH", "LEANING BULLISH", "BULLISH"][n]
    return {"n": n, "trend": trend,
            "checks": [{"name": a, "verdict": "bullish" if b else "bearish"} for a, b in checks]}


def _swing_action(score, weekly, rsi):
    wk = weekly["n"] if weekly else None
    reasons = []
    if wk is not None:
        reasons.append("Weekly trend supports it" if (score > 0 and wk >= 2) or (score < 0 and wk <= 1)
                       else "Weekly trend does not support it")
    if score >= 0.2:
        if wk is not None and wk < 2:
            return "WAIT", reasons + ["Daily is bullish but the weekly trend is weak"]
        if rsi > 72:
            return "WAIT FOR DIP", reasons + [f"RSI {rsi:.0f} is overheated, let it cool off"]
        return "BUY SETUP", reasons + ["Most indicators agree on the upside"]
    if score <= -0.2:
        return "AVOID", reasons + ["Most indicators are bearish"]
    return "WAIT", reasons + ["Indicators are mixed, no clear edge"]


# ---------- swing / daily analysis ----------
def analyze(df: pd.DataFrame) -> dict:
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]

    sma20, sma50, sma200 = (c.rolling(n).mean() for n in (20, 50, 200))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - macd_sig
    rsi = _rsi(c)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(3).mean()
    wr = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    tp = (h + l + c) / 3
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))
    adx, pdi, mdi = _adx(h, l, c)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    roc = c.pct_change(20) * 100
    atr = _atr(h, l, c)
    st, st_dir = _supertrend(h, l, c)

    last = lambda s: s.iloc[-1]
    price = last(c)
    sig = []

    def add(name, value, verdict):
        sig.append({"name": name, "value": value, "verdict": verdict})

    def vs(name, ma):
        m = last(ma)
        if not np.isnan(m):
            add(name, f"{m:,.2f}", "bullish" if price > m else "bearish")

    vs("Price vs SMA 20", sma20)
    vs("Price vs SMA 50", sma50)
    vs("Price vs SMA 200", sma200)
    if not np.isnan(last(sma200)):
        gc = last(sma50) > last(sma200)
        add("SMA 50 vs SMA 200", "Golden cross" if gc else "Death cross", "bullish" if gc else "bearish")

    r = last(rsi)
    rv = ("bearish" if r > 70 else "bullish" if r >= 55 else "neutral" if r >= 45
          else "bearish" if r > 30 else "bullish")
    add("RSI (14)", f"{r:.1f}", rv)

    add("MACD (12,26,9)", f"{last(macd):.2f}", "bullish" if last(macd) > last(macd_sig) else "bearish")
    hv, hp = last(hist), hist.iloc[-2]
    add("MACD histogram", f"{hv:.2f}",
        "bullish" if hv > 0 and hv >= hp else "bearish" if hv < 0 and hv <= hp else "neutral")

    kv, dv = last(k), last(d)
    add("Stochastic (14,3)", f"{kv:.1f}",
        "bearish" if kv > 80 else "bullish" if kv < 20 else ("bullish" if kv > dv else "bearish"))
    cv = last(cci)
    add("CCI (20)", f"{cv:.1f}", "bullish" if cv > 100 else "bearish" if cv < -100 else "neutral")
    wv = last(wr)
    add("Williams %R", f"{wv:.1f}", "bearish" if wv > -20 else "bullish" if wv < -80 else "neutral")
    add("Bollinger mid-band", f"{last(sma20):,.2f}", "bullish" if price > last(sma20) else "bearish")

    a, p, m = last(adx), last(pdi), last(mdi)
    add("ADX / DI (14)", f"{a:.1f}", "bullish" if a > 20 and p > m else "bearish" if a > 20 and m > p else "neutral")
    up_obv = last(obv) > obv.rolling(20).mean().iloc[-1]
    add("OBV trend", "Rising" if up_obv else "Falling", "bullish" if up_obv else "bearish")
    add("Supertrend (10,3)", f"{last(st):,.2f}", "bullish" if last(st_dir) == 1 else "bearish")
    r20 = last(roc)
    add("Momentum (20d)", f"{r20:+.1f}%", "bullish" if r20 > 0 else "bearish")

    counts, score = _count(sig)
    s50, s200 = last(sma50), last(sma200)
    if not np.isnan(s200):
        trend = ("STRONG UPTREND" if price > s50 > s200 else "STRONG DOWNTREND" if price < s50 < s200
                 else "UPTREND" if price > s50 else "DOWNTREND")
    else:
        trend = "UPTREND" if price > s50 else "DOWNTREND"

    weekly = _weekly(df)
    action, reasons = _swing_action(score, weekly, r)
    prev = c.iloc[-2]
    y = df.tail(252)
    t = df.tail(120)
    return {
        "price": _f(price), "change": _f(price - prev), "change_pct": _f((price / prev - 1) * 100),
        "overall": _label(score), "trend": trend, "score": _f(score),
        "counts": counts, "signals": sig,
        "action": action, "reasons": reasons,
        "plan": _plan(price, last(atr), score, 1.5),
        "weekly": weekly, "patterns": _patterns(df),
        "levels": {"support_20d": _f(l.tail(20).min()), "resistance_20d": _f(h.tail(20).max()),
                   "low_52w": _f(y["Low"].min()), "high_52w": _f(y["High"].max())},
        "chart": {"t": [str(x)[:10] for x in t.index], "o": _lst(t["Open"]), "h": _lst(t["High"]),
                  "l": _lst(t["Low"]), "c": _lst(t["Close"]),
                  "lines": {"SMA 20": _lst(sma20.tail(120)), "SMA 50": _lst(sma50.tail(120))}},
        "asof": str(df.index[-1])[:10],
    }


# ---------- intraday ----------
def analyze_intraday(df: pd.DataFrame) -> dict:
    h, l, c, v, o = df["High"], df["Low"], df["Close"], df["Volume"], df["Open"]
    day = pd.Series(df.index.date, index=df.index)
    tp = (h + l + c) / 3
    vwap = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum().replace(0, np.nan)
    st, st_dir = _supertrend(h, l, c, 10, 3.0)
    ema9, ema21 = c.ewm(span=9, adjust=False).mean(), c.ewm(span=21, adjust=False).mean()
    rsi = _rsi(c)
    atr = _atr(h, l, c)
    vavg = v.rolling(20).mean()

    price = c.iloc[-1]
    today = day.iloc[-1]
    tdf = df[day == today]
    prev = df[day < today]
    or_hi, or_lo = tdf["High"].iloc[0], tdf["Low"].iloc[0]
    sig = []

    def add(name, value, verdict):
        sig.append({"name": name, "value": value, "verdict": verdict})

    vw = vwap.iloc[-1]
    add("Price vs VWAP", f"{vw:,.2f}", "bullish" if price > vw else "bearish")
    add("Supertrend (10,3)", f"{st.iloc[-1]:,.2f}", "bullish" if st_dir.iloc[-1] == 1 else "bearish")
    add("EMA 9 vs EMA 21", f"{ema9.iloc[-1]:,.2f}", "bullish" if ema9.iloc[-1] > ema21.iloc[-1] else "bearish")
    r = rsi.iloc[-1]
    add("RSI (14)", f"{r:.1f}", "bullish" if r >= 55 else "bearish" if r <= 45 else "neutral")
    add("Opening range breakout", f"{or_lo:,.2f} - {or_hi:,.2f}",
        "bullish" if price > or_hi else "bearish" if price < or_lo else "neutral")
    spike = pd.notna(vavg.iloc[-1]) and v.iloc[-1] > 1.5 * vavg.iloc[-1]
    green = c.iloc[-1] >= o.iloc[-1]
    add("Volume spike", "Yes" if spike else "Normal",
        ("bullish" if green else "bearish") if spike else "neutral")

    piv = None
    if len(prev):
        H, L, C = prev["High"].max(), prev["Low"].min(), prev["Close"].iloc[-1]
        P = (H + L + C) / 3
        piv = {"prev_close": _f(C), "pivot": _f(P), "r1": _f(2 * P - L), "r2": _f(P + H - L),
               "s1": _f(2 * P - H), "s2": _f(P - (H - L))}
        add("Price vs previous close", f"{C:,.2f}", "bullish" if price > C else "bearish")

    counts, score = _count(sig)
    t = df.tail(80)
    return {
        "price": _f(price), "overall": _label(score), "score": _f(score),
        "counts": counts, "signals": sig,
        "plan": _plan(price, atr.iloc[-1], score, 1.0),
        "levels": {"day_high": _f(tdf["High"].max()), "day_low": _f(tdf["Low"].min()),
                   "or_high": _f(or_hi), "or_low": _f(or_lo)},
        "pivots": piv,
        "chart": {"t": [str(x)[5:16] for x in t.index], "o": _lst(t["Open"]), "h": _lst(t["High"]),
                  "l": _lst(t["Low"]), "c": _lst(t["Close"]),
                  "lines": {"VWAP": _lst(vwap.tail(80)), "Supertrend": _lst(st.tail(80))}},
        "asof": str(df.index[-1])[:16],
    }


# ---------- long-term / fundamentals ----------
def fundamentals_view(info: dict, df: pd.DataFrame, bench=None) -> dict:
    def g(k):
        x = info.get(k)
        return float(x) if isinstance(x, (int, float)) and not np.isnan(x) else None

    c = df["Close"]
    price = float(c.iloc[-1])
    sma200 = c.rolling(200).mean().iloc[-1]

    def ret(series, years):
        days = int(252 * years)
        if len(series) < 0.95 * days:
            return None
        base = series.iloc[max(-1 - days, -len(series))]
        return _f((series.iloc[-1] / base - 1) * 100, 1)

    r1, r3, r5 = ret(c, 1), ret(c, 3), ret(c, 5)
    nifty1 = ret(bench, 1) if bench is not None and len(bench) else None

    pe, pb, roe, de = g("trailingPE"), g("priceToBook"), g("returnOnEquity"), g("debtToEquity")
    eg, rg, pm, dr, mcap = g("earningsGrowth"), g("revenueGrowth"), g("profitMargins"), g("dividendRate"), g("marketCap")
    divy = dr / price * 100 if dr else None
    de = de / 100 if de is not None else None  # yfinance gives percent
    fin = "financial" in str(info.get("sector", "")).lower()

    checks = []

    def chk(name, value, ok, rule):
        checks.append({"name": name, "value": value, "rule": rule,
                       "status": "na" if ok is None else "pass" if ok else "fail"})

    chk("Valuation (P/E)", None if pe is None else f"{pe:.1f}", None if pe is None else 0 < pe <= 40, "P/E 40 or below")
    chk("Return on equity", None if roe is None else f"{roe*100:.1f}%", None if roe is None else roe >= 0.15, "ROE 15% or more")
    chk("Debt / equity", None if de is None or fin else f"{de:.2f}", None if de is None or fin else de <= 1,
        "1 or below (not used for banks/NBFCs)")
    chk("Profit growth", None if eg is None else f"{eg*100:+.1f}%", None if eg is None else eg > 0, "Growing year on year")
    chk("Sales growth", None if rg is None else f"{rg*100:+.1f}%", None if rg is None else rg > 0, "Growing year on year")
    chk("Above 200-day average", f"{sma200:,.2f}" if not np.isnan(sma200) else None,
        None if np.isnan(sma200) else price > sma200, "Long-term trend is up")
    chk("Beating Nifty (1 year)", None if r1 is None or nifty1 is None else f"{r1:+.1f}% vs {nifty1:+.1f}%",
        None if r1 is None or nifty1 is None else r1 > nifty1, "Stock return above Nifty 50")

    p = sum(x["status"] == "pass" for x in checks)
    f_ = sum(x["status"] == "fail" for x in checks)
    ratio = p / (p + f_) if p + f_ else None
    if ratio is None:
        verdict, action = "NOT ENOUGH DATA", "CHECK MANUALLY"
    elif ratio >= 0.7:
        verdict = "STRONG"
        action = "ACCUMULATE" if price > sma200 else "ACCUMULATE GRADUALLY"
    elif ratio >= 0.45:
        verdict, action = "AVERAGE", "HOLD / WATCH"
    else:
        verdict, action = "WEAK", "AVOID"

    hi52 = float(c.tail(252).max())
    return {
        "price": _f(price), "verdict": verdict, "action": action, "passed": p, "total": p + f_,
        "checks": checks,
        "returns": {"1y": r1, "3y": r3, "5y": r5, "nifty_1y": nifty1},
        "from_52w_high_pct": _f((price / hi52 - 1) * 100, 1),
        "profile": {"sector": info.get("sector"), "industry": info.get("industry"),
                    "market_cap_cr": _f(mcap / 1e7, 0) if mcap else None},
        "ratios": {"pe": _f(pe, 1), "pb": _f(pb, 1), "roe_pct": _f(roe * 100, 1) if roe is not None else None,
                   "debt_equity": _f(de), "profit_margin_pct": _f(pm * 100, 1) if pm is not None else None,
                   "dividend_yield_pct": _f(divy)},
        "closes": _lst(c.iloc[::-21][::-1].tail(60)),
    }
