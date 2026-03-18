#!/usr/bin/env python3
"""
미국 주식 종가매수 스크리너
Minervini SEPA + O'Neil CANSLIM 기반
데이터: yfinance (해외서버에서도 정상작동)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# =============================================
#  S&P 500 + NASDAQ 주요 종목 유니버스 (~200개)
# =============================================
US_UNIVERSE = {
    # 테크
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "META": "Meta", "AMZN": "Amazon", "TSLA": "Tesla", "AMD": "AMD",
    "AVGO": "Broadcom", "ORCL": "Oracle", "CRM": "Salesforce", "ADBE": "Adobe",
    "NFLX": "Netflix", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
    "NOW": "ServiceNow", "INTU": "Intuit", "AMAT": "Applied Materials",
    "MU": "Micron", "LRCX": "Lam Research", "KLAC": "KLA Corp",
    "MRVL": "Marvell", "SNPS": "Synopsys", "CDNS": "Cadence",
    "PANW": "Palo Alto", "CRWD": "CrowdStrike", "ZS": "Zscaler",
    "DDOG": "Datadog", "FTNT": "Fortinet", "SNOW": "Snowflake",
    "PLTR": "Palantir", "APP": "Applovin", "COIN": "Coinbase",
    "UBER": "Uber", "ABNB": "Airbnb", "BKNG": "Booking",
    # 금융
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "V": "Visa", "MA": "Mastercard",
    "AXP": "Amex", "BLK": "BlackRock", "SCHW": "Schwab",
    "SPGI": "S&P Global", "MCO": "Moody's", "ICE": "ICE",
    # 헬스케어
    "JNJ": "Johnson&Johnson", "UNH": "UnitedHealth", "LLY": "Eli Lilly",
    "ABBV": "AbbVie", "PFE": "Pfizer", "MRK": "Merck",
    "TMO": "Thermo Fisher", "ABT": "Abbott", "DHR": "Danaher",
    "ISRG": "Intuitive Surgical", "REGN": "Regeneron", "VRTX": "Vertex",
    "MRNA": "Moderna", "GILD": "Gilead", "BMY": "Bristol-Myers",
    # 소비재
    "HD": "Home Depot", "WMT": "Walmart", "COST": "Costco",
    "TGT": "Target", "LOW": "Lowe's", "NKE": "Nike",
    "SBUX": "Starbucks", "MCD": "McDonald's", "CMG": "Chipotle",
    "AMGN": "Amgen", "LULU": "Lululemon", "DECK": "Deckers",
    # 산업
    "CAT": "Caterpillar", "DE": "Deere", "GE": "GE",
    "HON": "Honeywell", "RTX": "RTX", "BA": "Boeing",
    "LMT": "Lockheed", "NOC": "Northrop", "GD": "General Dynamics",
    "ETN": "Eaton", "EMR": "Emerson", "PWR": "Quanta",
    # 에너지
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "EOG": "EOG Resources", "SLB": "Schlumberger", "OXY": "Occidental",
    # 통신/미디어
    "DIS": "Disney", "CMCSA": "Comcast", "T": "AT&T", "VZ": "Verizon",
    # 필수소비재
    "PG": "P&G", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "MDLZ": "Mondelez", "CL": "Colgate",
    # 기타 성장주
    "MSTR": "MicroStrategy", "HOOD": "Robinhood",
    "ENPH": "Enphase", "FSLR": "First Solar", "NEE": "NextEra",
    "SPOT": "Spotify", "RBLX": "Roblox", "DUOL": "Duolingo",
    "CELH": "Celsius", "ONON": "On Running", "HIMS": "Hims",
    "AXON": "Axon", "PODD": "Insulet", "TMDX": "TransMedics",
}

_cache = {}
_cache_lock = threading.Lock()


# =============================================
#  기술적 지표
# =============================================
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean().replace(0, np.nan)
    return 100 - (100 / (1 + ag / al))


def calc_macd(s, f=12, sl=26, sg=9):
    ef = s.ewm(span=f, adjust=False).mean()
    es = s.ewm(span=sl, adjust=False).mean()
    m = ef - es
    sig = m.ewm(span=sg, adjust=False).mean()
    return m, sig, m - sig


def calc_stoch(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(dp).mean()


def calc_atr(h, l, c, period=14):
    tr_list = []
    for j in range(1, len(c)):
        tr = max(h[j] - l[j], abs(h[j] - c[j-1]), abs(l[j] - c[j-1]))
        tr_list.append(tr)
    if len(tr_list) < period:
        return 0
    return float(np.mean(tr_list[-period:]))


def calc_obv_trend(c, v, lookback=20):
    i = len(c) - 1
    if i < lookback:
        return 0
    obv = 0
    obv_start = None
    for j in range(max(0, i - lookback), i + 1):
        if j == 0:
            continue
        if c[j] > c[j-1]:
            obv += v[j]
        elif c[j] < c[j-1]:
            obv -= v[j]
        if obv_start is None:
            obv_start = obv
    return obv - (obv_start or 0)


# =============================================
#  데이터 가져오기 (yfinance)
# =============================================
def fetch_us(ticker, target_date=None):
    cache_key = f"{ticker}_{target_date}"
    with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    try:
        end = datetime.strptime(target_date, "%Y-%m-%d") if target_date else datetime.now()
        start = end - timedelta(days=500)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            timeout=15
        )
        if df is None or len(df) < 10:
            return None

        # MultiIndex 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if target_date:
            df = df[df.index <= pd.Timestamp(target_date)]
        if len(df) < 201:
            return None

        with _cache_lock:
            _cache[cache_key] = df
        return df
    except Exception as e:
        print(f"  [yf] {ticker} fetch error: {e}")
        return None


# =============================================
#  매매가 계산 (USD 단위)
# =============================================
def calc_price_us(cl, lo, atr):
    if cl <= 0 or atr <= 0:
        return None
    buy = round(cl * 1.003, 2)
    sl_atr = round(buy - 2.0 * atr, 2)
    sl_low = round(lo * 0.995, 2)
    sl_max = round(buy * 0.92, 2)
    sl = max(sl_atr, sl_low, sl_max)
    risk = buy - sl
    if risk <= 0:
        return None
    t1 = round(buy + 2.0 * risk, 2)
    t2 = round(buy + 3.0 * risk, 2)
    rr = round((t1 - buy) / risk, 2)
    risk_pct = round((buy - sl) / buy * 100, 1)
    return {"buy": buy, "t1": t1, "t2": t2, "sl": sl,
            "rr": rr, "atr": round(atr, 2), "risk_pct": risk_pct}


# =============================================
#  조건 검색 (16개 조건)
# =============================================
def screen_us(df, name="", ticker=""):
    if len(df) < 201:
        return None

    c = df["Close"].values.astype(float)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    i = len(df) - 1

    if c[i] <= o[i]:
        return None
    if c[i] <= c[i-1]:
        return None
    chg = (c[i] - c[i-1]) / c[i-1] * 100
    if chg >= 15:
        return None

    av50 = np.mean(v[max(0, i-49):i]) if i >= 50 else np.mean(v[:i])
    vr = v[i] / av50 if av50 > 0 else 0
    if vr < 1.2:
        return None

    P, F = [], []
    score = 0

    sma50 = np.mean(c[i-49:i+1])
    sma150 = np.mean(c[i-149:i+1])
    sma200 = np.mean(c[i-199:i+1])
    sma200_1m = np.mean(c[i-221:i-22+1]) if i >= 221 else sma200

    ema10 = pd.Series(c).ewm(span=10, adjust=False).mean().values
    ema21 = pd.Series(c).ewm(span=21, adjust=False).mean().values
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values

    lookback_52w = min(i + 1, 250)
    high_52w = max(h[i - lookback_52w + 1:i + 1])
    low_52w = min(l[i - lookback_52w + 1:i + 1])

    if c[i] > sma150 and c[i] > sma200:
        P.append("Trend↑"); score += 7
    else:
        F.append("Trend✗")

    if sma150 > sma200:
        P.append("MA150>200"); score += 5
    else:
        F.append("MA150<200")

    if sma200 > sma200_1m:
        P.append("MA200↑"); score += 5
    else:
        F.append("MA200↓")

    if sma50 > sma150 > sma200:
        P.append("FullAlign"); score += 8
    else:
        F.append("NoAlign")

    if ema10[i] > ema21[i] > sma50:
        P.append("TripleStack"); score += 7
    else:
        F.append("NoStack")

    if low_52w > 0 and c[i] >= low_52w * 1.25:
        P.append("52wLow+25%"); score += 5
    else:
        F.append("52wLow✗")

    if high_52w > 0 and c[i] >= high_52w * 0.75:
        P.append("Near52wHi"); score += 5
    else:
        F.append("Far52wHi")

    if vr >= 3.0:
        P.append(f"Vol{vr:.1f}x↑↑"); score += 10
    elif vr >= 2.0:
        P.append(f"Vol{vr:.1f}x↑"); score += 8
    elif vr >= 1.5:
        P.append(f"Vol{vr:.1f}x"); score += 6
    else:
        P.append(f"Vol{vr:.1f}x"); score += 2

    obv_delta = calc_obv_trend(c, v, 20)
    if obv_delta > 0:
        P.append("OBV↑"); score += 5
    else:
        F.append("OBV↓")

    rv = calc_rsi(pd.Series(c), 14).iloc[-1]
    if not np.isnan(rv):
        if 45 <= rv <= 75:
            P.append(f"RSI{rv:.0f}"); score += 7
        elif rv > 75:
            F.append(f"RSI{rv:.0f}OB")
        elif rv > 40:
            P.append(f"RSI{rv:.0f}"); score += 3
        else:
            F.append(f"RSI{rv:.0f}✗")
    else:
        F.append("RSI-")

    macd_line, signal_line, macd_hist = calc_macd(pd.Series(c), 12, 26, 9)
    mh_now = macd_hist.iloc[-1]
    mh_prev = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
    ml_now = macd_line.iloc[-1]
    if not np.isnan(mh_now):
        if mh_now > 0 and mh_now > mh_prev:
            P.append("MACD↑↑"); score += 8
        elif mh_now > 0:
            P.append("MACD+"); score += 4
        elif ml_now > 0:
            P.append("MACD±"); score += 2
        else:
            F.append("MACD↓")
    else:
        F.append("MACD-")

    sk, sd = calc_stoch(pd.Series(h), pd.Series(l), pd.Series(c), 14, 3)
    sk_v = sk.iloc[-1]
    sd_v = sd.iloc[-1]
    if not np.isnan(sk_v) and not np.isnan(sd_v):
        if 20 <= sk_v <= 80 and sk_v > sd_v:
            P.append(f"Stoch{sk_v:.0f}↑"); score += 7
        elif sk_v > sd_v:
            P.append(f"Stoch{sk_v:.0f}"); score += 3
        else:
            F.append("Stoch↓")
    else:
        F.append("Stoch-")

    if ema20[i] > ema20[i-1] > ema20[i-2]:
        P.append("EMA20↑"); score += 5
    else:
        F.append("EMA20↓")

    bb_mid = np.mean(c[i-19:i+1])
    bb_std = np.std(c[i-19:i+1], ddof=1)
    bb_upper = bb_mid + 2 * bb_std
    if bb_mid <= c[i] <= bb_upper:
        P.append("BB↑Zone"); score += 5
    elif c[i] > bb_mid:
        P.append("BB+Mid"); score += 2
    else:
        F.append("BB↓")

    body = abs(c[i] - o[i])
    rng = h[i] - l[i]
    body_ratio = body / rng if rng > 0 else 0
    if body_ratio >= 0.65:
        P.append("Marubozu"); score += 5
    else:
        cn = 0
        for j in range(i, max(i-5, 0), -1):
            if c[j] > o[j]:
                cn += 1
            else:
                break
        if cn >= 3:
            P.append(f"{cn}ConsUp"); score += 4
        elif cn >= 2:
            P.append(f"{cn}ConsUp"); score += 2
        else:
            F.append("Candle✗")

    if sma200 > 0:
        rs_pct = (c[i] / sma200 - 1) * 100
        if rs_pct > 20:
            P.append(f"RS+{rs_pct:.0f}%↑↑"); score += 5
        elif rs_pct > 5:
            P.append(f"RS+{rs_pct:.0f}%"); score += 3
        elif rs_pct > 0:
            P.append(f"RS+{rs_pct:.0f}%"); score += 1
        else:
            F.append("RS✗")

    if len(P) < 8:
        return None

    atr = calc_atr(h, l, c)
    return {
        "passed": P, "failed": F,
        "pass_count": len(P), "total": len(P) + len(F),
        "momentum": round(min(score, 100), 1),
        "atr": round(atr, 2),
        "rsi": round(float(rv), 1) if not np.isnan(rv) else 0,
        "macd_hist": round(float(mh_now), 4) if not np.isnan(mh_now) else 0,
        "volume_ratio": round(vr, 1)
    }


# =============================================
#  스캔 실행
# =============================================
us_scan_status = {
    "running": False, "progress": 0, "total": 0,
    "found": 0, "message": "", "phase": ""
}


def run_us_scan(date_str):
    global us_scan_status
    results = []

    total = len(US_UNIVERSE)
    us_scan_status = {
        "running": True, "progress": 0, "total": total,
        "found": 0, "message": f"Scanning {total} US stocks...", "phase": "scan"
    }

    print(f"\n{'='*60}")
    print(f"  [US SCAN] {date_str} | {total} stocks")
    print(f"{'='*60}")

    scanned = 0

    def analyze_one(item):
        ticker, name = item
        df = fetch_us(ticker, date_str)
        if df is None:
            return None
        r = screen_us(df, name, ticker)
        if r is None:
            return None

        last_close = float(df["Close"].iloc[-1])
        last_open = float(df["Open"].iloc[-1])
        last_high = float(df["High"].iloc[-1])
        last_low = float(df["Low"].iloc[-1])
        last_vol = int(df["Volume"].iloc[-1])
        chg = (last_close - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2]) * 100

        p = calc_price_us(last_close, last_low, r["atr"])
        if p is None:
            return None

        dd = df.index[-1].strftime("%Y-%m-%d")
        return {
            "ticker": ticker,
            "name": name,
            "close": round(last_close, 2),
            "open": round(last_open, 2),
            "high": round(last_high, 2),
            "low": round(last_low, 2),
            "volume": last_vol,
            "buyPrice": p["buy"],
            "target1": p["t1"],
            "target2": p["t2"],
            "stoploss": p["sl"],
            "rrRatio": p["rr"],
            "riskPct": p["risk_pct"],
            "atr": p["atr"],
            "momentum": r["momentum"],
            "conditionsMet": r["passed"],
            "conditionsDetail": f'{r["pass_count"]}/{r["total"]}',
            "rsi": r["rsi"],
            "macdHist": r["macd_hist"],
            "volumeRatio": r["volume_ratio"],
            "changeRate": round(chg, 2),
            "dataDate": dd,
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_one, item): item for item in US_UNIVERSE.items()}
        for future in as_completed(futures):
            scanned += 1
            ticker, name = futures[future]
            us_scan_status["progress"] = scanned
            if scanned % 5 == 0:
                us_scan_status["message"] = f"{ticker} ({scanned}/{total})"
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    us_scan_status["found"] = len(results)
            except Exception as e:
                print(f"  [US] {ticker} error: {e}")

    results.sort(key=lambda x: x["momentum"], reverse=True)
    us_scan_status = {
        "running": False, "progress": total, "total": total,
        "found": len(results), "message": "done", "phase": "done"
    }

    print(f"\n{'='*60}")
    print(f"  [US RESULT] {date_str} | {len(results)} stocks found")
    if results:
        for idx, s in enumerate(results[:10], 1):
            print(f"  {idx:>2} {s['ticker']:<8} {s['name']:<20} ${s['close']:>8.2f} Score:{s['momentum']:>5.1f}")
    print(f"{'='*60}\n")
    return results
