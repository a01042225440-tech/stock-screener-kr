"""
1년 백테스트 v2 (캐시 우회 + 직접 fetch)
2025-05-27 ~ 2026-05-26 영업일 × 모든 종목 시뮬레이션
"""
import sys, os, json, time, requests, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app import (screen_pro, fetch_investor_data,
                  naver_all_rising_parallel, parse_num,
                  is_excluded_by_name, calc_price_pro, _session)
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

T_START = time.time()
END_DATE = "2026-05-26"
START_DATE = "2025-05-26"

# ─── 백테스트 전용 OHLCV fetch (캐시 우회, 더 긴 기간) ───
_bt_cache = {}
_bt_lock = threading.Lock()

def fetch_ohlcv_direct(code, end_date="2026-05-26", days_back=520):
    """캐시 우회하고 직접 네이버 차트 API에서 fetch.
    end_date 기준 days_back일 전부터 end_date까지."""
    with _bt_lock:
        if code in _bt_cache:
            return _bt_cache[code]

    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = end - timedelta(days=days_back)
    params = {
        "symbol": code, "requestType": "1",
        "startTime": start.strftime("%Y%m%d"),
        "endTime": end.strftime("%Y%m%d"),
        "timeframe": "day"
    }
    try:
        r = _session.get("https://fchart.stock.naver.com/siseJson.naver",
                         params=params, timeout=20)
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
        if len(rows) < 250:
            return None
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df = df.set_index("Date").sort_index()
        with _bt_lock:
            _bt_cache[code] = df
        return df
    except Exception:
        return None

# ─── 1. 종목 리스트 ───
print("\n" + "="*72)
print("  1년 백테스트 v2 — 캐시 우회 + 직접 fetch")
print("="*72)
print("\n[1/4] 종목 리스트...")
all_stocks = naver_all_rising_parallel()
candidates = []
for s in all_stocks:
    code = s.get("itemCode", ""); name = s.get("stockName", "")
    if s.get("stockEndType", "") not in ("stock", ""): continue
    if is_excluded_by_name(name, code): continue
    cl = parse_num(s.get("closePrice", "0"))
    trdval = parse_num(s.get("accumulatedTradingValue", "0"))
    mcap = parse_num(s.get("marketValue", "0"))
    if cl < 2000 or trdval < 1000 or mcap < 1000: continue
    candidates.append({"code": code, "name": name, "mcap": mcap})
print(f"  대상: {len(candidates)}개")

# ─── 2. 1년치 OHLCV 병렬 fetch (캐시 무시) ───
print(f"\n[2/4] 1년치 OHLCV 직접 fetch (병렬 20)...")
ohlcv_map = {}; fund_map = {}; done = [0]; fail = [0]

def fetch_one(cand):
    code = cand["code"]
    df = fetch_ohlcv_direct(code, END_DATE, days_back=520)
    if df is not None:
        ohlcv_map[code] = df
        inv = fetch_investor_data(code)
        if inv:
            fund_map[code] = {
                "per": inv.get("per", 0), "pbr": inv.get("pbr", 0),
                "eps": inv.get("eps", 0),
                "target_price": inv.get("target_price", 0),
                "sector_ratio": inv.get("sector_ratio", 50),
            }
    else:
        fail[0] += 1
    done[0] += 1
    if done[0] % 100 == 0:
        print(f"  {done[0]}/{len(candidates)}  OK={len(ohlcv_map)} 실패={fail[0]}  T+{time.time()-T_START:.0f}s")

with ThreadPoolExecutor(max_workers=20) as ex:
    list(ex.map(fetch_one, candidates))
print(f"  확보: {len(ohlcv_map)}개  실패: {fail[0]}개  T+{time.time()-T_START:.0f}s")

if len(ohlcv_map) < 100:
    print("ERROR: 데이터 너무 적음. 종료.")
    sys.exit(1)

# ─── 3. 영업일 ───
all_dates = sorted(set().union(*[set(df.index) for df in ohlcv_map.values()]))
test_dates = [d for d in all_dates if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]
test_dates = test_dates[:-5]
print(f"\n[3/4] 기간: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)}영업일)")

# ─── 4. 시뮬레이션 ───
print(f"\n[4/4] 시뮬레이션 ({len(ohlcv_map)}종목 × {len(test_dates)}일)...")
trades = []

CONDS = ["a1_ma200","a2_ma60","a3_align","a4_ma200up",
         "b1_eps","b2_per","b3_target",
         "k1_up_close","k2_bullish","k3_short_wick",
         "d1_pullback","d2_above_ma5","d3_bullish","d4_vol_pickup","d5_rsi_ok",
         "e1_new_high","e2_vol_burst","e3_price_up","e4_rsi_ok","e5_ma20_near",
         "inLeadingSector","huntTrigger","breakoutTrigger","bbLowerBreak"]

def sim_one_day(td):
    local = []
    for code, df_full in ohlcv_map.items():
        df_at = df_full[df_full.index <= td]
        if len(df_at) < 201: continue
        name = next((c["name"] for c in candidates if c["code"] == code), "")
        mcap = next((c["mcap"] for c in candidates if c["code"] == code), 0)
        try:
            r = screen_pro(df_at, name, code, mcap, fundamental=fund_map.get(code))
        except Exception:
            continue
        if r is None: continue
        grade = r.get("grade")
        if grade not in ("HUNT", "BREAKOUT", "BB_BREAK"): continue
        if not r.get("k2_bullish"): continue

        atr_v = r.get("atr", 0)
        if atr_v <= 0: continue
        last_close = int(df_at["Close"].iloc[-1])
        last_low = int(df_at["Low"].iloc[-1])
        p = calc_price_pro(last_close, last_low, atr_v)
        if p is None: continue
        buy, target, stop = p["buy"], p["t1"], p["sl"]

        df_after = df_full[df_full.index > td].head(5)
        if len(df_after) < 1: continue
        exit_day = None; exit_reason = None; exit_price = None
        for i, (idx, row) in enumerate(df_after.iterrows(), 1):
            h, l, c_ = int(row["High"]), int(row["Low"]), int(row["Close"])
            if exit_day is None:
                if h >= target:
                    exit_day, exit_reason, exit_price = i, "target", target
                elif l <= stop:
                    exit_day, exit_reason, exit_price = i, "stop", stop
                elif i == len(df_after):
                    exit_day, exit_reason, exit_price = i, "expire", c_
        if exit_day is None: continue
        profit_pct = (exit_price - buy) / buy * 100

        rec = {
            "date": str(td.date()), "code": code, "name": name,
            "grade": grade, "buy": buy, "target": target, "stop": stop,
            "exit_day": exit_day, "exit_reason": exit_reason,
            "profit_pct": round(profit_pct, 2),
            "rsi": float(r.get("rsi", 0)), "ma20Dist": float(r.get("ma20Dist", 0)),
            "volMult5": float(r.get("volMult5", 0)), "volMult20": float(r.get("volMult20", 0)),
            "finalScore": float(r.get("finalScore", 0)),
            "aCount": int(r.get("aCount", 0)), "bCount": int(r.get("bCount", 0)),
        }
        for k in CONDS:
            rec[k] = bool(r.get(k, False))
        local.append(rec)
    return local

# 영업일을 병렬 처리하지 말고 (메모리 문제), 순차 처리하되 종목 루프만
for di, td in enumerate(test_dates):
    trades.extend(sim_one_day(td))
    if di % 20 == 0 or di == len(test_dates) - 1:
        print(f"  {di+1}/{len(test_dates)}일 누적={len(trades)}  T+{time.time()-T_START:.0f}s")

print(f"\n총 trades: {len(trades)}  소요: {time.time()-T_START:.0f}s")
if not trades:
    sys.exit(0)

# ─── 5. 통계 ───
print("\n" + "="*72)
print("  📊 결과 통계")
print("="*72)

def stats(rows, label):
    if not rows: return
    n = len(rows)
    wins   = sum(1 for r in rows if r["exit_reason"] == "target")
    stops_ = sum(1 for r in rows if r["exit_reason"] == "stop")
    expire = sum(1 for r in rows if r["exit_reason"] == "expire")
    pl     = [r["profit_pct"] for r in rows]
    avg, med = np.mean(pl), np.median(pl)
    print(f"\n[{label}] n={n}")
    print(f"  🎯 청산도달: {wins} ({wins/n*100:.1f}%)")
    print(f"  🛑 손절:     {stops_} ({stops_/n*100:.1f}%)")
    print(f"  ⏰ 만기:     {expire} ({expire/n*100:.1f}%)")
    print(f"  평균수익: {avg:+.2f}%  중간값 {med:+.2f}%")

stats(trades, "전체")
for g in ["HUNT", "BREAKOUT", "BB_BREAK"]:
    stats([r for r in trades if r["grade"] == g], f"등급: {g}")

print("\n" + "="*72)
print("  🔬 각 조건이 +10% 청산도달률에 미치는 영향")
print("="*72)
print(f"{'조건':<22} {'통과 n':>7} {'미통과 n':>9} {'통과승률':>9} {'미통과':>8} {'차이':>8}")
print("-"*72)
for c in CONDS:
    p = [r for r in trades if r[c]]
    f = [r for r in trades if not r[c]]
    if not p or not f: continue
    pw = sum(1 for r in p if r["profit_pct"] >= 10)/len(p)*100
    fw = sum(1 for r in f if r["profit_pct"] >= 10)/len(f)*100
    diff = pw - fw
    mark = "↑↑" if diff > 10 else "↑" if diff > 5 else "↓↓" if diff < -10 else "↓" if diff < -5 else " "
    print(f"{c:<22} {len(p):>7d} {len(f):>9d} {pw:>8.1f}% {fw:>7.1f}% {diff:>+7.1f}%{mark}")

print("\n" + "="*72)
print("  📈 수치별 구간 분석")
print("="*72)

def bucket(field, ranges, label):
    print(f"\n[{label}]")
    for lo, hi in ranges:
        rows = [r for r in trades if lo <= r[field] < hi]
        if not rows: continue
        wins = sum(1 for r in rows if r["profit_pct"] >= 10)
        avg = np.mean([r["profit_pct"] for r in rows])
        med = np.median([r["profit_pct"] for r in rows])
        print(f"  {lo:>5.0f}~{hi:>5.0f}: n={len(rows):>4} 청산 {wins/len(rows)*100:>5.1f}% 평균 {avg:+5.2f}% 중간 {med:+5.2f}%")

bucket("rsi", [(0,40),(40,50),(50,60),(60,70),(70,80),(80,100)], "RSI")
bucket("ma20Dist", [(-30,-10),(-10,-5),(-5,0),(0,5),(5,10),(10,20),(20,50)], "20MA 거리")
bucket("volMult20", [(0,1.5),(1.5,2),(2,3),(3,5),(5,10),(10,50)], "거래량 20일 배수")
bucket("finalScore", [(0,80),(80,100),(100,120),(120,150),(150,250)], "최종점수")

# 저장
with open("backtest_1year_result.json", "w", encoding="utf-8") as f:
    json.dump({"period": f"{test_dates[0].date()}~{test_dates[-1].date()}",
               "total_trades": len(trades),
               "ohlcv_count": len(ohlcv_map),
               "elapsed_sec": int(time.time()-T_START),
               "trades": trades}, f, ensure_ascii=False, default=str)
print(f"\n저장 완료. 총 소요: {time.time()-T_START:.0f}s")
