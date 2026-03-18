#!/usr/bin/env python3
"""
미국 주식 종가매수 스크리너
yfinance 배치 다운로드 (빠른 버전)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import threading

# =============================================
#  종목 유니버스 (70개 - 배치 다운로드 최적화)
# =============================================
US_UNIVERSE = {
    "AAPL":"Apple", "MSFT":"Microsoft", "NVDA":"NVIDIA", "GOOGL":"Alphabet",
    "META":"Meta", "AMZN":"Amazon", "TSLA":"Tesla", "AMD":"AMD",
    "AVGO":"Broadcom", "ORCL":"Oracle", "CRM":"Salesforce", "ADBE":"Adobe",
    "NFLX":"Netflix", "QCOM":"Qualcomm", "NOW":"ServiceNow",
    "AMAT":"Applied Mat.", "MU":"Micron", "PANW":"Palo Alto",
    "CRWD":"CrowdStrike", "PLTR":"Palantir", "APP":"AppLovin",
    "COIN":"Coinbase", "UBER":"Uber", "SNOW":"Snowflake",
    "JPM":"JPMorgan", "BAC":"BofA", "GS":"Goldman", "V":"Visa",
    "MA":"Mastercard", "AXP":"Amex", "SPGI":"S&P Global",
    "JNJ":"J&J", "UNH":"UnitedHealth", "LLY":"Eli Lilly",
    "ABBV":"AbbVie", "MRK":"Merck", "TMO":"Thermo Fisher",
    "ISRG":"Intuitive Surg.", "REGN":"Regeneron", "VRTX":"Vertex",
    "HD":"Home Depot", "WMT":"Walmart", "COST":"Costco",
    "NKE":"Nike", "SBUX":"Starbucks", "MCD":"McDonald's",
    "CMG":"Chipotle", "LULU":"Lululemon", "DECK":"Deckers",
    "CAT":"Caterpillar", "DE":"Deere", "HON":"Honeywell",
    "RTX":"RTX", "LMT":"Lockheed", "GD":"General Dynamics",
    "XOM":"ExxonMobil", "CVX":"Chevron", "COP":"ConocoPhillips",
    "OXY":"Occidental",
    "DIS":"Disney", "CMCSA":"Comcast",
    "PG":"P&G", "KO":"Coca-Cola", "PEP":"PepsiCo",
    "AXON":"Axon", "CELH":"Celsius", "ONON":"On Running",
    "ENPH":"Enphase", "NEE":"NextEra", "FSLR":"First Solar",
    "SPOT":"Spotify", "DUOL":"Duolingo", "RBLX":"Roblox",
}

us_scan_status = {
    "running": False, "progress": 0, "total": 0,
    "found": 0, "message": "", "phase": ""
}


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
    return m, m.ewm(span=sg, adjust=False).mean(), m - m.ewm(span=sg, adjust=False).mean()


def calc_stoch(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(dp).mean()


def calc_atr(h, l, c, period=14):
    tr_list = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(1, len(c))]
    return float(np.mean(tr_list[-period:])) if len(tr_list) >= period else 0


def calc_obv_trend(c, v, lookback=20):
    i = len(c) - 1
    if i < lookback: return 0
    obv, obv_start = 0, None
    for j in range(max(0, i-lookback), i+1):
        if j == 0: continue
        obv += v[j] if c[j] > c[j-1] else (-v[j] if c[j] < c[j-1] else 0)
        if obv_start is None: obv_start = obv
    return obv - (obv_start or 0)


# =============================================
#  조건 검색
# =============================================
def screen_us(df, name="", ticker=""):
    if len(df) < 201: return None
    c = df["Close"].values.astype(float)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    i = len(df) - 1

    if c[i] <= o[i] or c[i] <= c[i-1]: return None
    chg = (c[i]-c[i-1])/c[i-1]*100
    if chg >= 15: return None

    av50 = np.mean(v[max(0,i-49):i]) if i >= 50 else np.mean(v[:i])
    vr = v[i]/av50 if av50 > 0 else 0
    if vr < 1.2: return None

    P, F, score = [], [], 0
    sma50 = np.mean(c[i-49:i+1])
    sma150 = np.mean(c[i-149:i+1])
    sma200 = np.mean(c[i-199:i+1])
    sma200_1m = np.mean(c[i-221:i-22+1]) if i >= 221 else sma200
    ema10 = pd.Series(c).ewm(span=10, adjust=False).mean().values
    ema21 = pd.Series(c).ewm(span=21, adjust=False).mean().values
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    high_52w = max(h[max(0,i-249):i+1])
    low_52w  = min(l[max(0,i-249):i+1])

    def chk(cond, label_p, label_f, pts):
        nonlocal score
        if cond: P.append(label_p); score += pts
        else: F.append(label_f)

    chk(c[i]>sma150 and c[i]>sma200, "Trend↑","Trend✗",7)
    chk(sma150>sma200, "MA150>200","MA150<200",5)
    chk(sma200>sma200_1m, "MA200↑","MA200↓",5)
    chk(sma50>sma150>sma200, "FullAlign","NoAlign",8)
    chk(ema10[i]>ema21[i]>sma50, "TripleStack","NoStack",7)
    chk(low_52w>0 and c[i]>=low_52w*1.25, "52wLow+25%","52wLow✗",5)
    chk(high_52w>0 and c[i]>=high_52w*0.75, "Near52wHi","Far52wHi",5)

    if vr>=3.0: P.append(f"Vol{vr:.1f}x↑↑"); score+=10
    elif vr>=2.0: P.append(f"Vol{vr:.1f}x↑"); score+=8
    elif vr>=1.5: P.append(f"Vol{vr:.1f}x"); score+=6
    else: P.append(f"Vol{vr:.1f}x"); score+=2

    chk(calc_obv_trend(c,v,20)>0, "OBV↑","OBV↓",5)

    rv = calc_rsi(pd.Series(c),14).iloc[-1]
    if not np.isnan(rv):
        if 45<=rv<=75: P.append(f"RSI{rv:.0f}"); score+=7
        elif rv>75: F.append(f"RSI{rv:.0f}OB")
        elif rv>40: P.append(f"RSI{rv:.0f}"); score+=3
        else: F.append(f"RSI{rv:.0f}✗")

    _, _, mh = calc_macd(pd.Series(c))
    mh_now = mh.iloc[-1]; mh_prev = mh.iloc[-2]
    ml_now = (pd.Series(c).ewm(span=12,adjust=False).mean()-pd.Series(c).ewm(span=26,adjust=False).mean()).iloc[-1]
    if not np.isnan(mh_now):
        if mh_now>0 and mh_now>mh_prev: P.append("MACD↑↑"); score+=8
        elif mh_now>0: P.append("MACD+"); score+=4
        elif ml_now>0: P.append("MACD±"); score+=2
        else: F.append("MACD↓")

    sk,sd = calc_stoch(pd.Series(h),pd.Series(l),pd.Series(c),14,3)
    sv,dv = sk.iloc[-1],sd.iloc[-1]
    if not np.isnan(sv) and not np.isnan(dv):
        if 20<=sv<=80 and sv>dv: P.append(f"Stoch{sv:.0f}↑"); score+=7
        elif sv>dv: P.append(f"Stoch{sv:.0f}"); score+=3
        else: F.append("Stoch↓")

    chk(ema20[i]>ema20[i-1]>ema20[i-2], "EMA20↑","EMA20↓",5)

    bb_mid = np.mean(c[i-19:i+1]); bb_std = np.std(c[i-19:i+1],ddof=1)
    if bb_mid<=c[i]<=bb_mid+2*bb_std: P.append("BB↑Zone"); score+=5
    elif c[i]>bb_mid: P.append("BB+Mid"); score+=2
    else: F.append("BB↓")

    body = abs(c[i]-o[i]); rng = h[i]-l[i]
    if rng>0 and body/rng>=0.65: P.append("Marubozu"); score+=5
    else:
        cn=0
        for j in range(i,max(i-5,0),-1):
            if c[j]>o[j]: cn+=1
            else: break
        if cn>=2: P.append(f"{cn}ConsUp"); score+=(4 if cn>=3 else 2)
        else: F.append("Candle✗")

    if sma200>0:
        rs = (c[i]/sma200-1)*100
        if rs>20: P.append(f"RS+{rs:.0f}%↑↑"); score+=5
        elif rs>5: P.append(f"RS+{rs:.0f}%"); score+=3
        elif rs>0: P.append(f"RS+{rs:.0f}%"); score+=1
        else: F.append("RS✗")

    if len(P) < 8: return None

    atr = calc_atr(h,l,c)
    return {
        "passed":P, "failed":F,
        "pass_count":len(P), "total":len(P)+len(F),
        "momentum":round(min(score,100),1),
        "atr":round(atr,2), "rsi":round(float(rv),1) if not np.isnan(rv) else 0,
        "macd_hist":round(float(mh_now),4), "volume_ratio":round(vr,1)
    }


def calc_price_us(cl, lo, atr):
    if cl<=0 or atr<=0: return None
    buy = round(cl*1.003,2)
    sl = max(round(buy-2.0*atr,2), round(lo*0.995,2), round(buy*0.92,2))
    risk = buy-sl
    if risk<=0: return None
    t1 = round(buy+2.0*risk,2); t2 = round(buy+3.0*risk,2)
    return {"buy":buy,"t1":t1,"t2":t2,"sl":sl,
            "rr":round((t1-buy)/risk,2),"risk_pct":round((buy-sl)/buy*100,1),"atr":round(atr,2)}


# =============================================
#  배치 다운로드 스캔
# =============================================
def run_us_scan(date_str):
    global us_scan_status
    results = []
    tickers = list(US_UNIVERSE.keys())
    total = len(tickers)

    us_scan_status = {"running":True,"progress":0,"total":total,
                      "found":0,"message":"Downloading data...","phase":"download"}
    print(f"\n[US SCAN] {date_str} | Batch download {total} tickers...")

    try:
        end_dt = datetime.strptime(date_str, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=500)
        # 배치 다운로드 (1번 API 호출로 전체 다운로드)
        raw = yf.download(
            tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt+timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            timeout=60,
            group_by="ticker"
        )
    except Exception as e:
        print(f"[US SCAN] Download error: {e}")
        us_scan_status["running"] = False
        us_scan_status["message"] = f"Download error: {e}"
        return []

    us_scan_status["message"] = "Analyzing..."
    print(f"[US SCAN] Download complete. Analyzing {total} stocks...")

    for idx, ticker in enumerate(tickers):
        us_scan_status["progress"] = idx+1
        if (idx+1) % 10 == 0:
            us_scan_status["message"] = f"Analyzing {ticker} ({idx+1}/{total})"
        try:
            # 배치 결과에서 개별 종목 추출
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[ticker].copy()
            else:
                df = raw.copy()

            df = df[["Open","High","Low","Close","Volume"]].dropna()
            df = df[df.index <= pd.Timestamp(date_str)]
            if len(df) < 201: continue

            name = US_UNIVERSE.get(ticker, ticker)
            r = screen_us(df, name, ticker)
            if r is None: continue

            last_close = float(df["Close"].iloc[-1])
            last_open  = float(df["Open"].iloc[-1])
            last_high  = float(df["High"].iloc[-1])
            last_low   = float(df["Low"].iloc[-1])
            last_vol   = int(df["Volume"].iloc[-1])
            chg = (last_close - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2]) * 100

            p = calc_price_us(last_close, last_low, r["atr"])
            if p is None: continue

            results.append({
                "ticker":ticker, "name":name,
                "close":round(last_close,2), "open":round(last_open,2),
                "high":round(last_high,2), "low":round(last_low,2),
                "volume":last_vol,
                "buyPrice":p["buy"],"target1":p["t1"],"target2":p["t2"],
                "stoploss":p["sl"],"rrRatio":p["rr"],"riskPct":p["risk_pct"],"atr":p["atr"],
                "momentum":r["momentum"],"conditionsMet":r["passed"],
                "conditionsDetail":f'{r["pass_count"]}/{r["total"]}',
                "rsi":r["rsi"],"macdHist":r["macd_hist"],"volumeRatio":r["volume_ratio"],
                "changeRate":round(chg,2),
                "dataDate":df.index[-1].strftime("%Y-%m-%d"),
            })
            us_scan_status["found"] = len(results)
        except Exception as e:
            print(f"  [US] {ticker} error: {e}")
            continue

    results.sort(key=lambda x: x["momentum"], reverse=True)
    us_scan_status = {"running":False,"progress":total,"total":total,
                      "found":len(results),"message":"done","phase":"done"}

    print(f"[US SCAN] Done: {len(results)} stocks found")
    if results:
        for i,s in enumerate(results[:5],1):
            print(f"  {i} {s['ticker']:<8} ${s['close']:>8.2f}  Score:{s['momentum']}")
    return results
