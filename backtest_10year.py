"""
10년 백테스트 v2 (2016-05-30 ~ 2026-05-26) - 엄격 필터 + 안전장치
- 시총 3000억+로 종목 약 500개로 축소
- 모든 print flush=True (실시간 로그)
- try/except 종목당 격리
- 500 trades마다 중간 저장 (실패해도 부분 결과 보존)
"""
import sys, os, json, time, threading, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app import (screen_pro, fetch_investor_data,
                  naver_all_rising_parallel, parse_num,
                  is_excluded_by_name, calc_price_pro, _session)
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def log(msg):
    print(msg, flush=True)

T_START = time.time()
END_DATE = "2026-05-26"
START_DATE = "2016-05-30"
DAYS_BACK = 3650 + 250
MIN_MCAP = 3000  # 시총 3000억+ (이전 1000억 → 3000억)

CYCLES = [
    ("2016-05~2017-12", "2016-05-30", "2017-12-31", "회복/강세"),
    ("2018-01~2018-12", "2018-01-01", "2018-12-31", "약세 (미중무역)"),
    ("2019-01~2019-12", "2019-01-01", "2019-12-31", "횡보"),
    ("2020-01~2020-03", "2020-01-01", "2020-03-31", "코로나 폭락"),
    ("2020-04~2021-06", "2020-04-01", "2021-06-30", "코로나 회복"),
    ("2021-07~2022-12", "2021-07-01", "2022-12-31", "약세 (금리)"),
    ("2023-01~2024-07", "2023-01-01", "2024-07-31", "회복/강세"),
    ("2024-08~2025-04", "2024-08-01", "2025-04-30", "조정"),
    ("2025-05~2026-05", "2025-05-01", "2026-05-26", "최근 1년"),
]

# ─── 1. 종목 리스트 ───
log("\n" + "="*72)
log(f"  10년 백테스트 v2 — 시총 {MIN_MCAP}억+ 필터, 안전장치 강화")
log("="*72)
log("\n[1/4] 종목 리스트...")
all_stocks = naver_all_rising_parallel()
candidates = []
for s in all_stocks:
    code = s.get("itemCode", ""); name = s.get("stockName", "")
    if s.get("stockEndType", "") not in ("stock", ""): continue
    if is_excluded_by_name(name, code): continue
    cl = parse_num(s.get("closePrice", "0"))
    trdval = parse_num(s.get("accumulatedTradingValue", "0"))
    mcap = parse_num(s.get("marketValue", "0"))
    if cl < 2000 or trdval < 1000 or mcap < MIN_MCAP: continue
    candidates.append({"code": code, "name": name, "mcap": mcap})
log(f"  대상: {len(candidates)}개")

# ─── 2. OHLCV 병렬 fetch ───
log(f"\n[2/4] 10년치 OHLCV fetch...")
ohlcv_map = {}; fund_map = {}; done = [0]; fail = [0]
_lock = threading.Lock()

def fetch_one(cand):
    try:
        code = cand["code"]
        end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=DAYS_BACK)
        params = {"symbol": code, "requestType": "1",
                  "startTime": start_dt.strftime("%Y%m%d"),
                  "endTime": end_dt.strftime("%Y%m%d"),
                  "timeframe": "day"}
        r = _session.get("https://fchart.stock.naver.com/siseJson.naver",
                         params=params, timeout=15)  # 15초 강제 타임아웃
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
            with _lock: fail[0] += 1
        else:
            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
            df = df.set_index("Date").sort_index()
            with _lock: ohlcv_map[code] = df
            try:
                inv = fetch_investor_data(code)
                if inv:
                    with _lock:
                        fund_map[code] = {k: inv.get(k,0) for k in
                                          ['per','pbr','eps','target_price','sector_ratio']}
            except: pass
    except Exception:
        with _lock: fail[0] += 1
    with _lock:
        done[0] += 1
        if done[0] % 50 == 0:
            log(f"  {done[0]}/{len(candidates)}  OK={len(ohlcv_map)} 실패={fail[0]}  T+{time.time()-T_START:.0f}s")

with ThreadPoolExecutor(max_workers=20) as ex:
    list(ex.map(fetch_one, candidates))
log(f"  확보: {len(ohlcv_map)}개  실패: {fail[0]}개  T+{time.time()-T_START:.0f}s")

if len(ohlcv_map) < 50:
    log("ERROR: 데이터 부족")
    sys.exit(1)

# 종목 이름/시총 빠른 조회용
code_meta = {c["code"]: c for c in candidates}

# ─── 3. 영업일 ───
all_dates = sorted(set().union(*[set(df.index) for df in ohlcv_map.values()]))
test_dates = [d for d in all_dates if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]
test_dates = test_dates[:-5]
log(f"\n[3/4] 기간: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)}영업일)")

# ─── 4. 시뮬레이션 ───
log(f"\n[4/4] 시뮬레이션 ({len(ohlcv_map)}종목 × {len(test_dates)}일)...")
trades = []
CONDS = ["a1_ma200","a2_ma60","a3_align","a4_ma200up",
         "b1_eps","b2_per","b3_target",
         "k1_up_close","k2_bullish","k3_short_wick",
         "d1_pullback","d2_above_ma5","d3_bullish","d4_vol_pickup","d5_rsi_ok",
         "e1_new_high","e2_vol_burst","e3_price_up","e4_rsi_ok","e5_ma20_near",
         "f1_first_pullback",
         "inLeadingSector","huntTrigger","breakoutTrigger"]

def save_partial(label):
    """중간 저장 - 실패 대비"""
    try:
        with open("backtest_10year_partial.json", "w", encoding="utf-8") as f:
            json.dump({"label": label, "trades_so_far": len(trades),
                       "trades": trades}, f, ensure_ascii=False, default=str)
    except: pass

err_count = 0
for di, td in enumerate(test_dates):
    for code, df_full in ohlcv_map.items():
        try:
            df_at = df_full[df_full.index <= td]
            if len(df_at) < 201: continue
            meta = code_meta.get(code, {})
            name = meta.get("name", "")
            mcap = meta.get("mcap", 0)
            r = screen_pro(df_at, name, code, mcap, fundamental=fund_map.get(code))
            if r is None: continue
            grade = r.get("grade")
            if grade not in ("HUNT", "BREAKOUT"): continue
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
            exit_day = exit_reason = exit_price = None
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

            rec = {"date": str(td.date()), "code": code, "name": name,
                   "grade": grade, "buy": buy, "target": target, "stop": stop,
                   "exit_day": exit_day, "exit_reason": exit_reason,
                   "profit_pct": round(profit_pct, 2),
                   "rsi": float(r.get("rsi", 0)),
                   "ma20Dist": float(r.get("ma20Dist", 0)),
                   "volMult20": float(r.get("volMult20", 0)),
                   "finalScore": float(r.get("finalScore", 0)),
                   "aCount": int(r.get("aCount", 0)),
                   "pullbackCount": int(r.get("pullbackCount", 0))}
            for k in CONDS:
                rec[k] = bool(r.get(k, False))
            trades.append(rec)
            # 500 trades마다 중간 저장
            if len(trades) % 500 == 0:
                save_partial(f"day{di+1}")
        except Exception as e:
            err_count += 1
            if err_count <= 5:
                log(f"  ⚠️ 예외 {code}@{td.date()}: {e}")
    if di % 50 == 0 or di == len(test_dates) - 1:
        log(f"  {di+1}/{len(test_dates)}일 trades={len(trades)} errs={err_count}  T+{time.time()-T_START:.0f}s")

log(f"\n총 trades: {len(trades)}  예외: {err_count}  소요: {time.time()-T_START:.0f}s")
if not trades:
    log("거래 없음 - 종료")
    sys.exit(0)

# ─── 통계 ───
log("\n" + "="*72)
log("  📊 전체 통계 (10년)")
log("="*72)

def stats(rows, label):
    if not rows: return
    n = len(rows)
    wins   = sum(1 for r in rows if r["exit_reason"] == "target")
    stops_ = sum(1 for r in rows if r["exit_reason"] == "stop")
    expire = sum(1 for r in rows if r["exit_reason"] == "expire")
    pl     = [r["profit_pct"] for r in rows]
    avg, med = np.mean(pl), np.median(pl)
    log(f"\n[{label}] n={n}")
    log(f"  🎯 청산도달: {wins:>5} ({wins/n*100:>5.1f}%)  🛑 손절: {stops_:>5} ({stops_/n*100:>5.1f}%)  ⏰ 만기: {expire:>5} ({expire/n*100:>5.1f}%)")
    log(f"  평균수익: {avg:+.2f}%  중간값 {med:+.2f}%")

stats(trades, "전체 10년")
for g in ["HUNT", "BREAKOUT"]:
    stats([r for r in trades if r["grade"] == g], f"등급: {g}")

# 시장 사이클별
log("\n" + "="*72)
log("  📅 시장 사이클별 성과")
log("="*72)
log(f"{'사이클':<24} {'상태':<14} {'n':>5} {'청산':>6} {'손절':>6} {'평균':>7}")
log("-"*72)
for label, s_dt, e_dt, market in CYCLES:
    rows = [r for r in trades if s_dt <= r["date"] <= e_dt]
    if not rows: continue
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")/n*100
    s_ = sum(1 for r in rows if r["exit_reason"] == "stop")/n*100
    avg = np.mean([r["profit_pct"] for r in rows])
    log(f"{label:<24} {market:<14} {n:>5} {w:>5.1f}% {s_:>5.1f}% {avg:>+6.2f}%")

# 연도별
log("\n" + "="*72)
log("  📅 연도별 성과")
log("="*72)
years = sorted(set(r["date"][:4] for r in trades))
log(f"{'연도':<8} {'n':>5} {'청산':>7} {'손절':>7} {'평균':>8}")
log("-"*72)
for y in years:
    rows = [r for r in trades if r["date"].startswith(y)]
    if not rows: continue
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")/n*100
    s_ = sum(1 for r in rows if r["exit_reason"] == "stop")/n*100
    avg = np.mean([r["profit_pct"] for r in rows])
    log(f"{y:<8} {n:>5} {w:>6.1f}% {s_:>6.1f}% {avg:>+7.2f}%")

# 조건별 영향
log("\n" + "="*72)
log("  🔬 조건별 영향")
log("="*72)
log(f"{'조건':<22} {'통과 n':>7} {'미통과':>8} {'통과승률':>9} {'미통과':>8} {'차이':>8}")
log("-"*72)
for c in CONDS:
    p = [r for r in trades if r[c]]
    f = [r for r in trades if not r[c]]
    if not p or not f: continue
    pw = sum(1 for r in p if r["profit_pct"] >= 10)/len(p)*100
    fw = sum(1 for r in f if r["profit_pct"] >= 10)/len(f)*100
    diff = pw - fw
    mark = "↑↑" if diff > 10 else "↑" if diff > 5 else "↓↓" if diff < -10 else "↓" if diff < -5 else " "
    log(f"{c:<22} {len(p):>7d} {len(f):>8d} {pw:>8.1f}% {fw:>7.1f}% {diff:>+7.1f}%{mark}")

# F1 풀백 카운트별 (영상 검증)
log("\n" + "="*72)
log("  ⭐ F1 풀백 카운트별 (영상 가이드 검증)")
log("="*72)
log(f"{'풀백차수':<10} {'n':>5} {'청산':>7} {'손절':>7} {'평균':>8}")
log("-"*72)
for cnt in range(7):
    rows = [r for r in trades if r["pullbackCount"] == cnt]
    if not rows: continue
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")/n*100
    s_ = sum(1 for r in rows if r["exit_reason"] == "stop")/n*100
    avg = np.mean([r["profit_pct"] for r in rows])
    log(f"{cnt}차       {n:>5} {w:>6.1f}% {s_:>6.1f}% {avg:>+7.2f}%")

# 수치 구간
log("\n" + "="*72)
log("  📈 수치별 구간 (10년)")
log("="*72)

def bucket(field, ranges, label):
    log(f"\n[{label}]")
    for lo, hi in ranges:
        rows = [r for r in trades if lo <= r[field] < hi]
        if not rows: continue
        wins = sum(1 for r in rows if r["profit_pct"] >= 10)
        avg = np.mean([r["profit_pct"] for r in rows])
        log(f"  {lo:>5.0f}~{hi:>5.0f}: n={len(rows):>5} 청산 {wins/len(rows)*100:>5.1f}% 평균 {avg:+5.2f}%")

bucket("rsi", [(0,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,100)], "RSI")
bucket("ma20Dist", [(-30,-10),(-10,-5),(-5,0),(0,5),(5,10),(10,20),(20,50)], "20MA 거리")
bucket("volMult20", [(0,1.5),(1.5,2),(2,3),(3,5),(5,10),(10,50)], "거래량")
bucket("finalScore", [(0,80),(80,100),(100,120),(120,150),(150,250)], "최종점수")

# 최종 저장
with open("backtest_10year_result.json", "w", encoding="utf-8") as f:
    json.dump({"period": f"{test_dates[0].date()}~{test_dates[-1].date()}",
               "total_trades": len(trades), "ohlcv_count": len(ohlcv_map),
               "elapsed_sec": int(time.time()-T_START),
               "trades": trades}, f, ensure_ascii=False, default=str)
log(f"\n저장: backtest_10year_result.json  총 소요: {time.time()-T_START:.0f}s")
