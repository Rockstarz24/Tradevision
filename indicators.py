import numpy as np
import pandas as pd


def _rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def _f(x, d=2):
    try:
        x = float(x)
        return None if np.isnan(x) else round(x, d)
    except Exception:
        return None


def analyze(df: pd.DataFrame) -> dict:
    """df columns: Open, High, Low, Close, Volume (daily, oldest first)."""
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]

    sma20, sma50, sma200 = (c.rolling(n).mean() for n in (20, 50, 200))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
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
        add("SMA 50 vs SMA 200", "Golden cross" if last(sma50) > last(sma200) else "Death cross",
            "bullish" if last(sma50) > last(sma200) else "bearish")

    r = last(rsi)
    if r > 70: rv = "bearish"      # overbought
    elif r >= 55: rv = "bullish"
    elif r >= 45: rv = "neutral"
    elif r > 30: rv = "bearish"
    else: rv = "bullish"           # oversold
    add("RSI (14)", f"{r:.1f}", rv)

    add("MACD (12,26,9)", f"{last(macd):.2f}", "bullish" if last(macd) > last(macd_sig) else "bearish")
    hv, hp = last(hist), hist.iloc[-2]
    add("MACD histogram", f"{hv:.2f}",
        "bullish" if hv > 0 and hv >= hp else "bearish" if hv < 0 and hv <= hp else "neutral")

    kv, dv = last(k), last(d)
    sv = "bearish" if kv > 80 else "bullish" if kv < 20 else ("bullish" if kv > dv else "bearish")
    add("Stochastic (14,3)", f"{kv:.1f}", sv)

    cv = last(cci)
    add("CCI (20)", f"{cv:.1f}", "bullish" if cv > 100 else "bearish" if cv < -100 else "neutral")
    wv = last(wr)
    add("Williams %R", f"{wv:.1f}", "bearish" if wv > -20 else "bullish" if wv < -80 else "neutral")

    add("Bollinger mid-band", f"{last(sma20):,.2f}", "bullish" if price > last(sma20) else "bearish")

    a, p, m = last(adx), last(pdi), last(mdi)
    add("ADX / DI (14)", f"{a:.1f}", "bullish" if a > 20 and p > m else "bearish" if a > 20 and m > p else "neutral")
    add("OBV trend", "Rising" if last(obv) > obv.rolling(20).mean().iloc[-1] else "Falling",
        "bullish" if last(obv) > obv.rolling(20).mean().iloc[-1] else "bearish")
    rv20 = last(roc)
    add("Momentum (20d)", f"{rv20:+.1f}%", "bullish" if rv20 > 0 else "bearish")

    nb = sum(s["verdict"] == "bullish" for s in sig)
    nn = sum(s["verdict"] == "neutral" for s in sig)
    nr = sum(s["verdict"] == "bearish" for s in sig)
    score = (nb - nr) / len(sig)
    overall = ("STRONG BULLISH" if score >= 0.5 else "BULLISH" if score >= 0.2 else
               "NEUTRAL" if score > -0.2 else "BEARISH" if score > -0.5 else "STRONG BEARISH")

    s50, s200 = last(sma50), last(sma200)
    if not np.isnan(s200):
        trend = ("STRONG UPTREND" if price > s50 > s200 else "STRONG DOWNTREND" if price < s50 < s200
                 else "UPTREND" if price > s50 else "DOWNTREND")
    else:
        trend = "UPTREND" if price > s50 else "DOWNTREND"

    prev = c.iloc[-2]
    y = df.tail(252)
    return {
        "price": _f(price), "change": _f(price - prev), "change_pct": _f((price / prev - 1) * 100),
        "overall": overall, "trend": trend, "score": _f(score, 2),
        "counts": {"bullish": nb, "neutral": nn, "bearish": nr},
        "signals": sig,
        "levels": {
            "support_20d": _f(l.tail(20).min()), "resistance_20d": _f(h.tail(20).max()),
            "low_52w": _f(y["Low"].min()), "high_52w": _f(y["High"].max()),
        },
        "closes": [_f(x) for x in c.tail(90)],
        "asof": str(df.index[-1])[:10],
    }
