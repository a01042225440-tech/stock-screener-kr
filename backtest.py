#!/usr/bin/env python3
"""
한국주식 종가매수 스크리너 백테스트
기간: 2025-03-01 ~ 2026-03-01
매일 거래량1위 + 모멘텀1위 2종목 종가매수 → 목표가/손절가 기준 청산
"""
import sys, os, time, json, pickle
import numpy as np, pandas as pd
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except: pass

# =============================================
#  네이버 OHLCV 조회 (app.py와 동일)
# =============================================
_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'backtest_ohlcv.pkl')

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except: pass
    return {}

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

_ohlcv_cache = load_cache()

def naver_ohlcv(code, start_date="20240301", end_date="20260320"):
    """네이버 금융 OHLCV 조회 (캐시)"""
    if code in _ohlcv_cache:
        return _ohlcv_cache[code]

    params = {
        "symbol": code, "requestType": "1",
        "startTime": start_date,
        "endTime": end_date,
        "timeframe": "day"
    }
    try:
        r = _session.get("https://fchart.stock.naver.com/siseJson.naver",
                         params=params, timeout=10)
        text = r.text.strip()
        rows = []
        for line in text.split('\n'):
            line = line.strip().rstrip(',')
            if not line: continue
            if ("'2" in line or '"2' in line) and line.startswith('['):
                line = line.strip('[]')
                parts = [p.strip().strip("'\"") for p in line.split(',')]
                if len(parts) >= 6:
                    try:
                        dt = parts[0].strip()
                        o, h, l, c, v = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                        if c > 0:
                            rows.append({"Date": dt, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
                    except: continue
        if len(rows) < 50:
            return None
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df = df.set_index("Date").sort_index()
        _ohlcv_cache[code] = df
        return df
    except:
        return None


# =============================================
#  KOSPI + KOSDAQ 전종목 리스트 (네이버)
# =============================================
def get_all_stock_codes():
    """네이버에서 KOSPI + KOSDAQ 전종목 코드 가져오기"""
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'all_codes.pkl')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            if len(data) > 500:
                print(f"  [CACHE] {len(data)}개 종목코드 로드됨")
                return data
        except: pass

    EXCLUDE_KEYWORDS = [
        "ETF", "ETN", "KODEX", "TIGER", "KOSEF", "KINDEX", "HANARO",
        "SOL ", "ACE ", "ARIRANG", "BNK", "FOCUS", "KBSTAR", "TREX",
        "파워", "인버스", "레버리지", "2X", "3X",
        "스팩", "SPAC", "기업인수", "리츠", "REIT", "선물", "옵션",
    ]

    all_stocks = {}
    for market in ["KOSPI", "KOSDAQ"]:
        page = 1
        while True:
            try:
                url = f"https://m.stock.naver.com/api/stocks/up/{market}?page={page}&pageSize=100"
                r = _session.get(url, timeout=10)
                if r.status_code != 200: break
                data = r.json()
                stocks = data.get("stocks", [])
                if not stocks: break
                for s in stocks:
                    code = s.get("itemCode", "")
                    name = s.get("stockName", "")
                    end_type = s.get("stockEndType", "")
                    if end_type not in ("stock", ""): continue
                    name_upper = name.upper()
                    skip = False
                    for kw in EXCLUDE_KEYWORDS:
                        if kw.upper() in name_upper:
                            skip = True; break
                    if skip: continue
                    if len(code) == 6 and code[-1] in ("5","7","8","9","K","L"): continue
                    if name.endswith("우") or name.endswith("우B") or "우선" in name: continue
                    all_stocks[code] = {"name": name, "market": market}
                total = data.get("totalCount", 0)
                if len(stocks) < 100 or page * 100 >= total: break
                page += 1
                time.sleep(0.05)
            except: break

    # 하락 종목도 가져오기
    for market in ["KOSPI", "KOSDAQ"]:
        page = 1
        while True:
            try:
                url = f"https://m.stock.naver.com/api/stocks/down/{market}?page={page}&pageSize=100"
                r = _session.get(url, timeout=10)
                if r.status_code != 200: break
                data = r.json()
                stocks = data.get("stocks", [])
                if not stocks: break
                for s in stocks:
                    code = s.get("itemCode", "")
                    name = s.get("stockName", "")
                    end_type = s.get("stockEndType", "")
                    if end_type not in ("stock", ""): continue
                    name_upper = name.upper()
                    skip = False
                    for kw in EXCLUDE_KEYWORDS:
                        if kw.upper() in name_upper:
                            skip = True; break
                    if skip: continue
                    if len(code) == 6 and code[-1] in ("5","7","8","9","K","L"): continue
                    if name.endswith("우") or name.endswith("우B") or "우선" in name: continue
                    if code not in all_stocks:
                        all_stocks[code] = {"name": name, "market": market}
                total = data.get("totalCount", 0)
                if len(stocks) < 100 or page * 100 >= total: break
                page += 1
                time.sleep(0.05)
            except: break

    print(f"  [NAVER] 전체 {len(all_stocks)}개 종목 수집")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(all_stocks, f)
    return all_stocks


# =============================================
#  기술적 지표 (app.py와 동일)
# =============================================
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean().replace(0, np.nan)
    return 100 - (100 / (1 + ag / al))

def calc_atr(h, l, c, period=14):
    tr_list = []
    for j in range(1, len(c)):
        tr = max(h[j] - l[j], abs(h[j] - c[j-1]), abs(l[j] - c[j-1]))
        tr_list.append(tr)
    if len(tr_list) < period: return 0
    return float(np.mean(tr_list[-period:]))

def calc_stoch(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(dp).mean()

def calc_obv_trend(c, v, lookback=20):
    i = len(c) - 1
    if i < lookback: return 0
    obv = 0; obv_start = None
    for j in range(max(0, i - lookback), i + 1):
        if j == 0: continue
        if c[j] > c[j-1]: obv += v[j]
        elif c[j] < c[j-1]: obv -= v[j]
        if obv_start is None: obv_start = obv
    return (obv - obv_start) if obv_start is not None else 0


# =============================================
#  스크리닝 (app.py screen_pro와 동일)
# =============================================
def screen_pro(df):
    """16개 조건 스크리닝 - app.py와 동일 로직"""
    if len(df) < 201: return None
    c = df["Close"].values
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    v = df["Volume"].values
    i = len(df) - 1

    # Must-pass
    if c[i] <= o[i]: return None
    if c[i] <= c[i-1]: return None
    chg = (c[i] - c[i-1]) / c[i-1] * 100
    if chg >= 29: return None

    av50 = np.mean(v[max(0, i-49):i]) if i >= 50 else np.mean(v[:i])
    vr = v[i] / av50 if av50 > 0 else 0
    if vr < 1.5: return None
    if vr > 8.0: return None  # ★ 거래량 폭증 = 단타세력, 페이드 패턴

    # ★ 갭업 4% 이상 필터 (Day-1 손절 방지)
    gap_pct = (o[i] - c[i-1]) / c[i-1] * 100 if c[i-1] > 0 else 0
    if gap_pct > 4.0: return None

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

    # ★ 필수조건: 완전정배열 (50>150>200)
    if not (sma50 > sma150 > sma200): return None

    # ★ 필수조건: RSI 40~75
    rv = calc_rsi(pd.Series(c), 14).iloc[-1]
    if np.isnan(rv) or rv < 40 or rv > 75: return None

    def chk(cond, p_label, f_label, pts):
        nonlocal score
        if cond: P.append(p_label); score += pts
        else: F.append(f_label)

    chk(c[i] > sma150 and c[i] > sma200, "A", "A", 7)
    chk(sma150 > sma200, "B", "B", 5)
    chk(sma200 > sma200_1m, "C", "C", 5)
    chk(sma50 > sma150 > sma200, "D", "D", 12)  # ★ 8→12 (최고 신뢰 조건)
    chk(ema10[i] > ema21[i] > sma50, "E", "E", 7)
    chk(low_52w > 0 and c[i] >= low_52w * 1.25, "F", "F", 5)
    chk(high_52w > 0 and c[i] >= high_52w * 0.75, "G", "G", 5)

    # ★ H(거래량) 점수: 적정 거래량이 최적, 극단치 페널티
    if 1.5 <= vr < 3.0: P.append("H"); score += 8    # 최적 구간
    elif 3.0 <= vr < 5.0: P.append("H"); score += 5  # 수용 가능
    elif 5.0 <= vr <= 8.0: P.append("H"); score += 2 # 주의
    else: P.append("H"); score += 0

    obv_delta = calc_obv_trend(c, v, 20)
    chk(obv_delta > 0, "I", "I", 5)

    rv = calc_rsi(pd.Series(c), 14).iloc[-1]
    if not np.isnan(rv):
        if 45 <= rv <= 75: P.append("J"); score += 7
        elif rv > 75: F.append("J")
        elif rv > 40: P.append("J"); score += 3
        else: F.append("J")
    else: F.append("J")

    macd_line = pd.Series(c).ewm(span=12,adjust=False).mean()-pd.Series(c).ewm(span=26,adjust=False).mean()
    signal = macd_line.ewm(span=9,adjust=False).mean()
    hist = macd_line - signal
    mh_now = hist.iloc[-1]; mh_prev = hist.iloc[-2] if len(hist)>1 else 0
    ml_now = macd_line.iloc[-1]
    if not np.isnan(mh_now):
        if mh_now > 0 and mh_now > mh_prev: P.append("K"); score += 8
        elif mh_now > 0: P.append("K"); score += 4
        elif ml_now > 0: P.append("K"); score += 2
        else: F.append("K")

    sk, sd = calc_stoch(pd.Series(h), pd.Series(l), pd.Series(c), 14, 3)
    sv, dv = sk.iloc[-1], sd.iloc[-1]
    if not np.isnan(sv) and not np.isnan(dv):
        if 20 <= sv <= 80 and sv > dv: P.append("L"); score += 7
        elif sv > dv: P.append("L"); score += 3
        else: F.append("L")
    else: F.append("L")

    chk(ema20[i] > ema20[i-1] > ema20[i-2], "M", "M", 5)

    bb_mid = np.mean(c[i-19:i+1]); bb_std = np.std(c[i-19:i+1], ddof=1)
    if bb_mid <= c[i] <= bb_mid + 2 * bb_std: P.append("N"); score += 5
    elif c[i] > bb_mid: P.append("N"); score += 2
    else: F.append("N")

    body = abs(c[i] - o[i]); rng = h[i] - l[i]
    if rng > 0 and body / rng >= 0.65: P.append("O"); score += 5
    else:
        cn = 0
        for j in range(i, max(i-5,0), -1):
            if c[j] > o[j]: cn += 1
            else: break
        if cn >= 2: P.append("O"); score += (4 if cn >= 3 else 2)
        else: F.append("O")

    if sma200 > 0:
        rs = (c[i]/sma200-1)*100
        if rs > 20: P.append("P"); score += 5
        elif rs > 5: P.append("P"); score += 3
        elif rs > 0: P.append("P"); score += 1
        else: F.append("P")

    # ★ Q. 풀백 바운스 보너스 (최근 5일내 EMA 터치 후 반등)
    ema10_touched = False; ema21_touched = False
    for j in range(max(0, i-4), i):
        if l[j] <= ema10[j] <= h[j]: ema10_touched = True
        if l[j] <= ema21[j] <= h[j]: ema21_touched = True
    if ema10_touched or ema21_touched:
        P.append("Q"); score += 5
    else:
        F.append("Q")

    if len(P) < 10: return None

    atr = calc_atr(h, l, c)
    return {
        "pass_count": len(P), "total": len(P)+len(F),
        "momentum": round(min(score, 100), 1),
        "atr": round(atr), "atr_raw": atr,
        "volume_ratio": round(vr, 1),
        "volume": int(v[i]),
    }


# =============================================
#  매매가 계산 (app.py와 동일)
# =============================================
def tick(p, ref):
    for lim, t in [(2000,1),(5000,5),(20000,10),(50000,50),(200000,100),(500000,500)]:
        if ref < lim: return (p // t) * t
    return (p // 1000) * 1000

def calc_price(cl, lo, atr):
    if cl <= 0 or atr <= 0: return None
    buy = tick(int(cl * 1.003), cl)
    sl_atr = int(buy - 1.8 * atr)      # ★ 1.5→1.8 (약간 넓혀 Day-1 손절 감소)
    sl_low = int(lo * 0.995)
    sl_max = int(buy * 0.93)
    sl = tick(max(sl_atr, sl_low, sl_max), cl)
    risk = buy - sl
    if risk <= 0: return None
    t1 = tick(int(buy + 2.5 * risk), cl)
    t2 = tick(int(buy + 4.0 * risk), cl)  # ★ 3.5→4.0 (수익 극대화)
    return {"buy": buy, "t1": t1, "t2": t2, "sl": sl, "atr": atr}


# =============================================
#  하루 스크리닝 (특정 날짜 기준)
# =============================================
def screen_one_day(all_stocks, target_date_str):
    """특정 날짜 기준으로 전종목 스크리닝"""
    target_dt = pd.Timestamp(target_date_str)
    results = []

    for code, info in all_stocks.items():
        df = _ohlcv_cache.get(code)
        if df is None: continue

        # 해당 날짜까지의 데이터만
        df_cut = df[df.index <= target_dt]
        if len(df_cut) < 201: continue

        # 해당 날짜에 거래가 있는지 확인
        last_date = df_cut.index[-1]
        if abs((last_date - target_dt).days) > 3: continue  # 3일 이상 차이나면 스킵

        r = screen_pro(df_cut)
        if r is None: continue

        cl = int(df_cut["Close"].iloc[-1])
        lo = int(df_cut["Low"].iloc[-1])
        p = calc_price(cl, lo, r["atr"])
        if p is None: continue

        results.append({
            "code": code, "name": info["name"],
            "close": cl, "low": lo,
            "buy": p["buy"], "t1": p["t1"], "t2": p["t2"], "sl": p["sl"],
            "atr": p.get("atr", 0),
            "momentum": r["momentum"],
            "volume_ratio": r["volume_ratio"],
            "volume": r["volume"],
            "data_date": last_date.strftime("%Y-%m-%d"),
        })

    return results


# =============================================
#  매수 후 결과 추적
# =============================================
def simulate_trade(code, buy_price, t1, t2, sl, entry_date_str, atr=0):
    """★ 트레일링 스톱 + 분할매도 전략 (최대 25 거래일)"""
    df = _ohlcv_cache.get(code)
    if df is None: return {"result": "no_data", "pnl_pct": 0, "days": 0}

    entry_dt = pd.Timestamp(entry_date_str)
    future = df[df.index > entry_dt].head(25)  # ★ 15→25일 (타임아웃 수익 극대화)

    if len(future) == 0:
        return {"result": "no_data", "pnl_pct": 0, "days": 0}

    if atr <= 0:
        all_data = df[df.index <= entry_dt]
        if len(all_data) > 14:
            atr = calc_atr(all_data["High"].values, all_data["Low"].values, all_data["Close"].values, 14)

    current_sl = sl
    highest = buy_price
    t1_hit = False
    t1_pnl = 0.0

    for idx, (dt, row) in enumerate(future.iterrows()):
        hi = int(row["High"])
        lo = int(row["Low"])
        cl = int(row["Close"])

        highest = max(highest, hi)
        gain_pct = (highest - buy_price) / buy_price * 100

        # ★ 트레일링 스톱: T1 도달 후에만 적용 (그 전에는 원래 SL 유지)
        if t1_hit and atr > 0:
            trailing = int(highest - 2.0 * atr)
            current_sl = max(current_sl, trailing)

        # 손절 체크
        if lo <= current_sl:
            exit_p = current_sl
            if t1_hit:
                pnl = t1_pnl * 0.333 + (exit_p - buy_price) / buy_price * 100 * 0.667
                result = "trail_stop"
            else:
                pnl = (exit_p - buy_price) / buy_price * 100
                result = "stoploss"
            return {"result": result, "pnl_pct": round(pnl, 2), "days": idx+1,
                    "exit_date": dt.strftime("%Y-%m-%d")}

        # T2 도달
        if hi >= t2:
            t2_pnl = (t2 - buy_price) / buy_price * 100
            if t1_hit:
                pnl = t1_pnl * 0.333 + t2_pnl * 0.333 + (cl - buy_price) / buy_price * 100 * 0.334
            else:
                pnl = t2_pnl
            return {"result": "target2", "pnl_pct": round(pnl, 2), "days": idx+1,
                    "exit_date": dt.strftime("%Y-%m-%d")}

        # T1 도달 (1/3 매도)
        if not t1_hit and hi >= t1:
            t1_hit = True
            t1_pnl = (t1 - buy_price) / buy_price * 100
            current_sl = max(current_sl, buy_price)  # 최소 본전

    # 25일 경과 시 종가 청산
    last_close = int(future["Close"].iloc[-1])
    if t1_hit:
        pnl = t1_pnl * 0.333 + (last_close - buy_price) / buy_price * 100 * 0.667
    else:
        pnl = (last_close - buy_price) / buy_price * 100
    return {"result": "timeout", "pnl_pct": round(pnl, 2), "days": len(future),
            "exit_date": future.index[-1].strftime("%Y-%m-%d")}


# =============================================
#  메인 백테스트
# =============================================
def main():
    print("=" * 70)
    print("  한국주식 종가매수 스크리너 백테스트")
    print("  기간: 2025-03-01 ~ 2026-03-01")
    print("  전략: 매일 종합점수1위 1종목 + 트레일링스톱")
    print("=" * 70)

    # 1. 전종목 코드 수집
    print("\n[1/4] 전종목 코드 수집...")
    all_stocks = get_all_stock_codes()
    print(f"  → {len(all_stocks)}개 종목")

    # 2. 전종목 OHLCV 다운로드 (2024-03-01 ~ 2026-03-20)
    print(f"\n[2/4] 전종목 OHLCV 다운로드 (캐시: {len(_ohlcv_cache)}개)...")
    codes_to_fetch = [c for c in all_stocks if c not in _ohlcv_cache]
    if codes_to_fetch:
        print(f"  → {len(codes_to_fetch)}개 신규 다운로드 필요")
        done = 0
        for code in codes_to_fetch:
            naver_ohlcv(code)
            done += 1
            if done % 50 == 0:
                print(f"  → {done}/{len(codes_to_fetch)} 다운로드 완료...")
                save_cache(_ohlcv_cache)
                time.sleep(0.5)
            time.sleep(0.05)
        save_cache(_ohlcv_cache)
    print(f"  → 캐시: {len(_ohlcv_cache)}개 종목")

    # 3. 거래일 목록 생성 (2025-03-01 ~ 2026-03-01)
    print("\n[3/4] 거래일별 스크리닝...")
    trading_days = pd.bdate_range(start="2025-03-03", end="2026-03-01", freq="B")

    all_trades = []
    daily_summary = []

    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d")

        results = screen_one_day(all_stocks, day_str)

        if not results:
            continue

        # ★ 종합점수 = 모멘텀×0.7 + 거래량점수×0.3 (1종목/일)
        for r in results:
            vr = r["volume_ratio"]
            if vr < 3.0:
                vol_score = vr / 3.0 * 100
            elif vr < 8.0:
                vol_score = 100 - (vr - 3.0) * 10
            else:
                vol_score = 20
            r["composite"] = r["momentum"] * 0.7 + vol_score * 0.3

        ranked = sorted(results, key=lambda x: x["composite"], reverse=True)
        pick = ranked[0]

        trade_result = simulate_trade(
            pick["code"], pick["buy"], pick["t1"], pick["t2"], pick["sl"],
            day_str, atr=pick.get("atr", 0)
        )
        trade = {
            "entry_date": day_str,
            "type": "종합1위",
            "code": pick["code"],
            "name": pick["name"],
            "buy": pick["buy"],
            "t1": pick["t1"],
            "t2": pick["t2"],
            "sl": pick["sl"],
            "momentum": pick["momentum"],
            "volume_ratio": pick["volume_ratio"],
            **trade_result
        }
        all_trades.append(trade)

        if (day_idx + 1) % 20 == 0:
            wins = sum(1 for t in all_trades if t["pnl_pct"] > 0)
            total = len(all_trades)
            wr = wins/total*100 if total > 0 else 0
            avg = np.mean([t["pnl_pct"] for t in all_trades]) if all_trades else 0
            print(f"  {day_str} | 스캔:{len(results)}종목 | 누적거래:{total}건 | 승률:{wr:.1f}% | 평균수익:{avg:.2f}%")

    # 4. 결과 출력
    print_results(all_trades)

    # 거래 내역 CSV 저장
    df_trades = pd.DataFrame(all_trades)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_result.csv")
    df_trades.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  거래내역 저장: {csv_path}")


def print_results(trades):
    if not trades:
        print("\n  거래 내역 없음!")
        return

    print("\n" + "=" * 70)
    print("  백테스트 결과 (2025-03-01 ~ 2026-03-01)")
    print("=" * 70)

    total = len(trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    pnls = [t["pnl_pct"] for t in trades]
    avg_pnl = np.mean(pnls)
    med_pnl = np.median(pnls)
    total_pnl = np.sum(pnls)
    max_win = max(pnls) if pnls else 0
    max_loss = min(pnls) if pnls else 0

    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

    # 유형별 분석
    t2_trades = [t for t in trades if t["result"] == "target2"]
    sl_trades = [t for t in trades if t["result"] == "stoploss"]
    to_trades = [t for t in trades if t["result"] == "timeout"]

    ts_trades = [t for t in trades if t["result"] == "trail_stop"]

    # 복리 수익률 (종목당 동일 금액 투자 가정)
    compound = 1.0
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        compound *= (1 + t["pnl_pct"] / 100)  # ★ 1종목/일 전액 투자
    compound_pct = (compound - 1) * 100

    print(f"""
  ┌────────────────────────────────────────────────┐
  │  총 거래 횟수:  {total:>6}건                      │
  │  승리:          {len(wins):>6}건 ({len(wins)/total*100:.1f}%)                │
  │  패배:          {len(losses):>6}건 ({len(losses)/total*100:.1f}%)                │
  ├────────────────────────────────────────────────┤
  │  평균 수익률:    {avg_pnl:>+8.2f}%                    │
  │  중간값 수익률:  {med_pnl:>+8.2f}%                    │
  │  누적 수익률합:  {total_pnl:>+8.2f}%                    │
  │  복리 수익률:    {compound_pct:>+8.2f}%                    │
  │  최대 수익:      {max_win:>+8.2f}%                    │
  │  최대 손실:      {max_loss:>+8.2f}%                    │
  ├────────────────────────────────────────────────┤
  │  평균 수익 (승): {avg_win:>+8.2f}%                    │
  │  평균 손실 (패): {avg_loss:>+8.2f}%                    │
  │  손익비(RR):     {abs(avg_win/avg_loss) if avg_loss!=0 else 0:>8.2f}                    │
  ├────────────────────────────────────────────────┤
  │  T2 도달:  {len(t2_trades):>5}건 | 손절: {len(sl_trades):>5}건 | 트레일: {len(ts_trades):>5}건 | 타임아웃: {len(to_trades):>5}건 │
  ├────────────────────────────────────────────────┤
  │  전략: 1종목/일 종합점수1위 + 트레일링스톱 25일  │
  └────────────────────────────────────────────────┘
""")

    # 상위/하위 5개 거래
    sorted_trades = sorted(trades, key=lambda x: x["pnl_pct"], reverse=True)
    print("  ── 최고 수익 TOP 5 ──")
    for t in sorted_trades[:5]:
        print(f"    {t['entry_date']} {t['name']:<10} {t['type']} | 매수:{t['buy']:>8,} → {t['result']} | {t['pnl_pct']:+.2f}%")

    print("\n  ── 최대 손실 TOP 5 ──")
    for t in sorted_trades[-5:]:
        print(f"    {t['entry_date']} {t['name']:<10} {t['type']} | 매수:{t['buy']:>8,} → {t['result']} | {t['pnl_pct']:+.2f}%")


if __name__ == "__main__":
    main()
