"""데모 데이터 생성 모듈"""
import numpy as np, pandas as pd
from datetime import datetime

DEMO_STOCKS = {
    "005930": ("삼성전자", 4000000, "KOSPI"), "000660": ("SK하이닉스", 1200000, "KOSPI"),
    "005380": ("현대차", 500000, "KOSPI"), "000270": ("기아", 380000, "KOSPI"),
    "068270": ("셀트리온", 350000, "KOSPI"), "035420": ("NAVER", 330000, "KOSPI"),
    "051910": ("LG화학", 280000, "KOSPI"), "105560": ("KB금융", 250000, "KOSPI"),
    "055550": ("신한지주", 200000, "KOSPI"), "066570": ("LG전자", 170000, "KOSPI"),
    "086790": ("하나금융지주", 140000, "KOSPI"), "015760": ("한국전력", 130000, "KOSPI"),
    "035720": ("카카오", 120000, "KOSPI"), "259960": ("크래프톤", 110000, "KOSPI"),
    "033780": ("KT&G", 90000, "KOSPI"), "009150": ("삼성전기", 95000, "KOSPI"),
    "011200": ("HMM", 85000, "KOSPI"), "024110": ("기업은행", 95000, "KOSPI"),
    "352820": ("하이브", 75000, "KOSPI"), "034020": ("두산에너빌리티", 80000, "KOSPI"),
    "047050": ("포스코인터내셔널", 65000, "KOSPI"), "000720": ("현대건설", 35000, "KOSPI"),
    "010140": ("삼성중공업", 25000, "KOSPI"), "004020": ("현대제철", 20000, "KOSPI"),
    "247540": ("에코프로비엠", 120000, "KOSDAQ"), "196170": ("알테오젠", 150000, "KOSDAQ"),
    "058470": ("리노공업", 35000, "KOSDAQ"), "036930": ("주성엔지니어링", 15000, "KOSDAQ"),
    "357780": ("솔브레인", 18000, "KOSDAQ"), "041510": ("에스엠", 15000, "KOSDAQ"),
    "095340": ("ISC", 8000, "KOSDAQ"), "067310": ("하나마이크론", 8000, "KOSDAQ"),
    "336260": ("두산퓨얼셀", 6000, "KOSDAQ"), "000990": ("DB하이텍", 10000, "KOSDAQ"),
    "383220": ("F&F", 25000, "KOSDAQ"), "035900": ("JYP Ent.", 15000, "KOSDAQ"),
}

def demo_data(name, mcap, seed, tgt):
    rng = np.random.RandomState(seed); n = 300
    if mcap > 100000: base = rng.randint(40000, 80000)
    elif mcap > 10000: base = rng.randint(10000, 60000)
    else: base = rng.randint(2000, 20000)
    bull = (seed % 5 <= 2); strong = (seed % 7 <= 1)
    mu = rng.uniform(0.0005, 0.0015) if bull else rng.uniform(-0.0005, 0.0005)
    sig = rng.uniform(0.015, 0.03); rets = rng.normal(mu, sig, n)
    for _ in range(rng.randint(3, 8)): rets[rng.randint(50, n)] += rng.choice([-1, 1]) * rng.uniform(0.02, 0.06)
    if bull: rets[-60:] += 0.0015; rets[-10:] += 0.001
    if strong: rets[-3:] = np.abs(rng.normal(0.015, 0.005, 3)); rets[-1] = rng.uniform(0.02, 0.04)
    prices = np.maximum(base * np.exp(np.cumsum(rets)), 100).astype(int)
    dates = pd.bdate_range(end=tgt, periods=n); rows = []
    for j, (dt, cl) in enumerate(zip(dates, prices)):
        iv = sig * rng.uniform(0.5, 1.5)
        if strong and j >= n - 3:
            op = int(cl * rng.uniform(0.97, 0.99)); hi = int(cl * rng.uniform(1.001, 1.01)); lo = int(op * rng.uniform(0.995, 1.0))
        else:
            hi = int(cl * (1 + abs(rng.normal(0, iv)))); lo = int(cl * (1 - abs(rng.normal(0, iv)))); op = int(lo + (hi - lo) * rng.uniform(0.2, 0.8))
        hi = max(hi, cl, op); lo = max(min(lo, cl, op), 1)
        bv = int(max(mcap, 1000) * 80 * rng.uniform(0.5, 2.0)); vol = int(bv * (1 + abs(rets[j]) * 15) * rng.uniform(0.3, 2.5))
        if strong and j >= n - 1: vol = int(vol * rng.uniform(2.5, 4.0))
        rows.append({"Date": dt, "Open": max(op, 1), "High": max(hi, 1), "Low": max(lo, 1), "Close": max(cl, 1), "Volume": max(vol, 1)})
    return pd.DataFrame(rows).set_index("Date")

def calc_rsi(s, p=14):
    d = s.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
    return 100 - (100 / (1 + g.ewm(alpha=1/p, min_periods=p, adjust=False).mean() / l.ewm(alpha=1/p, min_periods=p, adjust=False).mean().replace(0, np.nan)))

def calc_macd(s, f=12, sl=26, sg=9):
    ef = s.ewm(span=f, adjust=False).mean(); es = s.ewm(span=sl, adjust=False).mean(); m = ef - es
    return m, m.ewm(span=sg, adjust=False).mean(), m - m.ewm(span=sg, adjust=False).mean()

def calc_stoch(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp).min(); hi = h.rolling(kp).max(); k = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(dp).mean()

def screen(df, name, code, mcap):
    if len(df) < 201: return None
    c = df["Close"].values; o = df["Open"].values; h = df["High"].values; l = df["Low"].values; v = df["Volume"].values; i = len(df) - 1
    P, F = [], []
    if 500 <= c[i] <= 500000: P.append("A.주가범위")
    else: F.append("A.주가범위")
    if 500 <= mcap <= 50000: P.append("B.시총")
    elif mcap > 50000: P.append("B.대형주")
    else: F.append("B.시총미달")
    if c[i] > o[i]: P.append("C.양봉")
    else: return None
    if c[i] > c[i-1]: P.append("D.전일대비↑")
    else: return None
    m20 = np.mean(c[i-19:i+1]); m60 = np.mean(c[i-59:i+1]); m120 = np.mean(c[i-119:i+1])
    if m20 > m60 > m120: P.append("E.정배열")
    else: F.append("E.정배열")
    if c[i] > np.mean(c[i-199:i+1]): P.append("F.200MA↑")
    else: F.append("F.200MA↑")
    av = np.mean(v[i-20:i]); vr = v[i] / av if av > 0 else 0
    if vr >= 1.5: P.append(f"G.거래량{vr:.1f}x")
    else: return None
    bm = np.mean(c[i-19:i+1]); bs = np.std(c[i-19:i+1], ddof=1)
    if bm <= c[i] <= bm + 2 * bs: P.append("H.BB중심↑")
    else: F.append("H.BB범위밖")
    rv = calc_rsi(pd.Series(c), 14).iloc[-1]
    if not np.isnan(rv) and 40 <= rv <= 70: P.append(f"I.RSI{rv:.0f}")
    else: F.append(f"I.RSI{rv:.0f}" if not np.isnan(rv) else "I.RSI-")
    _, _, mh = calc_macd(pd.Series(c), 12, 26, 9); mv = mh.iloc[-1]
    if not np.isnan(mv) and mv > 0: P.append("J.MACD+")
    else: F.append("J.MACD-")
    sk, _ = calc_stoch(pd.Series(h), pd.Series(l), pd.Series(c), 14, 3); sv = sk.iloc[-1]
    if not np.isnan(sv) and 25 <= sv <= 80: P.append(f"K.Stoch{sv:.0f}")
    else: F.append("K.Stoch범위밖")
    ma = pd.Series(c).rolling(20).mean().values
    if not np.isnan(ma[i]) and not np.isnan(ma[i-2]) and ma[i] > ma[i-1] > ma[i-2]: P.append("L.20MA추세↑")
    else: F.append("L.20MA횡보")
    bd = abs(c[i] - o[i]); rn = h[i] - l[i]; br = bd / rn if rn > 0 else 0
    if br >= 0.6: P.append("M.장대양봉")
    elif c[i] > o[i] and c[i-1] > o[i-1]: P.append("M.연속양봉")
    else: F.append("M.캔들약세")
    dc = (c[i] - c[i-1]) / c[i-1] * 100 if c[i-1] > 0 else 0
    if dc < 29: P.append("N.상한가X")
    else: return None
    if len(P) < 8: return None
    sc = 0
    if rn > 0: sc += min(bd / rn * 40, 25)
    sc += min(vr * 10, 25); sc += min(max(dc * 3, 0), 20)
    if not np.isnan(rv): sc += max(0, 15 - abs(rv - 57) * 0.5)
    cn = 0
    for j in range(i, max(i - 5, 0), -1):
        if c[j] > o[j]: cn += 1
        else: break
    sc += min(cn * 5, 15)
    return {"passed": P, "failed": F, "pass_count": len(P), "total": len(P) + len(F), "momentum": round(min(sc, 100), 1)}

def tick(p, r):
    for lim, t in [(2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)]:
        if r < lim: return (p // t) * t
    return (p // 1000) * 1000

def calc_price(cl, lo):
    if cl <= 0: return None
    buy = tick(int(cl * 1.005), cl); t1 = tick(int(buy * 1.03), cl); t2 = tick(int(buy * 1.05), cl)
    sl = tick(min(int(buy * 0.97), int(lo * 0.995) if lo > 0 else int(buy * 0.97)), cl)
    risk = buy - sl; reward = t1 - buy
    if risk <= 0: return None
    return {"buy": buy, "t1": t1, "t2": t2, "sl": sl, "rr": round(reward / risk, 2)}

def run_demo_scan(date_str, status):
    tgt = datetime.strptime(date_str, "%Y-%m-%d")
    results = []; total = len(DEMO_STOCKS); scanned = 0
    status["total"] = total
    for code, (name, mcap, market) in DEMO_STOCKS.items():
        scanned += 1
        status["progress"] = scanned
        status["message"] = f"[데모] {name} 분석 중..."
        df = demo_data(name, mcap, hash(code) % 10000, tgt)
        if df is None: continue
        r = screen(df, name, code, mcap)
        if r is None: continue
        cl = int(df["Close"].iloc[-1]); lo = int(df["Low"].iloc[-1]); p = calc_price(cl, lo)
        if p is None: continue
        dd = df.index[-1]; dd = dd.strftime("%Y-%m-%d") if hasattr(dd, 'strftime') else date_str
        results.append({
            "code": code, "name": name, "market": market,
            "close": cl, "open": int(df["Open"].iloc[-1]), "high": int(df["High"].iloc[-1]), "low": lo,
            "volume": int(df["Volume"].iloc[-1]), "marketCap": mcap,
            "buyPrice": p["buy"], "target1": p["t1"], "target2": p["t2"],
            "stoploss": p["sl"], "rrRatio": p["rr"], "momentum": r["momentum"],
            "conditionsMet": r["passed"], "conditionsDetail": f'{r["pass_count"]}/{r["total"]}',
            "dataDate": dd, "changeRate": round((cl - int(df["Open"].iloc[-1])) / int(df["Open"].iloc[-1]) * 100, 2)
        })
        status["found"] = len(results)
    results.sort(key=lambda x: x["momentum"], reverse=True)
    return results
