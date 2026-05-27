"""
변경된 조건 (C 섹터 보너스화 + 안전장치 + -10% 손절 + RSI 30-70)으로
4/1 ~ 5/27 백테스트 + 수익률 분석
"""
import sys, os, json, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app import (screen_pro, fetch_investor_data,
                  naver_all_rising_parallel, parse_num,
                  is_excluded_by_name, calc_price_pro, _session)
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def log(m): print(m, flush=True)

T0 = time.time()
END_DATE = "2026-06-05"     # +5일 추적 위해 여유분
START_DATE = "2026-04-01"
LAST_BUY_DATE = "2026-05-27"

# 1. 종목 리스트
log("\n" + "="*72)
log("  4/1 ~ 5/27 백테스트 (변경된 조건 적용)")
log("="*72)
log("\n[1] 종목 리스트...")
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
log(f"  대상: {len(candidates)}개")

# 2. OHLCV fetch
log(f"\n[2] OHLCV fetch (캐시 활용)...")
ohlcv_map = {}; fund_map = {}; done = [0]
_lock = threading.Lock()

def fetch(cand):
    try:
        code = cand["code"]
        end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=520)
        params = {"symbol": code, "requestType": "1",
                  "startTime": start_dt.strftime("%Y%m%d"),
                  "endTime": end_dt.strftime("%Y%m%d"),
                  "timeframe": "day"}
        r = _session.get("https://fchart.stock.naver.com/siseJson.naver",
                         params=params, timeout=15)
        rows = []
        for line in r.text.strip().split('\n'):
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
        if len(rows) >= 250:
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
    except: pass
    with _lock:
        done[0] += 1
        if done[0] % 100 == 0:
            log(f"  {done[0]}/{len(candidates)}  OK={len(ohlcv_map)}  T+{time.time()-T0:.0f}s")

with ThreadPoolExecutor(max_workers=20) as ex:
    list(ex.map(fetch, candidates))
log(f"  확보: {len(ohlcv_map)}  T+{time.time()-T0:.0f}s")

code_meta = {c["code"]: c for c in candidates}

# 3. 영업일 (4/1 ~ 5/27)
all_dates = sorted(set().union(*[set(df.index) for df in ohlcv_map.values()]))
test_dates = [d for d in all_dates
              if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(LAST_BUY_DATE)]
log(f"\n[3] 영업일: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)}일)")

# 4. 시뮬레이션 (변경된 등급 판정 + 안전장치)
log(f"\n[4] 시뮬레이션...")
all_trades = []

BUY_GRADES = {"HUNT", "BREAKOUT"}

for di, td in enumerate(test_dates):
    # 1단계: 모든 종목 screen_pro
    day_results = []
    for code, df_full in ohlcv_map.items():
        try:
            df_at = df_full[df_full.index <= td]
            if len(df_at) < 201: continue
            meta = code_meta.get(code, {})
            r = screen_pro(df_at, meta.get("name",""), code, meta.get("mcap",0),
                          fundamental=fund_map.get(code))
            if r is None: continue
            r["code"] = code; r["name"] = meta.get("name","")
            day_results.append(r)
        except: continue

    # 2단계: 안전장치 적용 (app.py와 동일 로직)
    primary = [r for r in day_results
               if r.get("grade") in BUY_GRADES
               and r.get("k2_bullish")
               and r.get("finalScore", 0) >= 100]
    if not primary:
        primary = [r for r in day_results
                   if r.get("grade") in BUY_GRADES
                   and r.get("k2_bullish")
                   and r.get("finalScore", 0) >= 80][:3]
    if not primary:
        primary = [r for r in day_results
                   if r.get("grade") == "TREND"
                   and r.get("k2_bullish")
                   and r.get("finalScore", 0) >= 100][:3]
        for r in primary: r["isAlt"] = True
    if not primary:
        trend_bullish = [r for r in day_results
                         if r.get("grade") == "TREND" and r.get("k2_bullish")]
        trend_bullish.sort(key=lambda x: -x.get("finalScore", 0))
        primary = trend_bullish[:1]
        for r in primary: r["isAlt"] = True
    # 등급 우선순위 정렬
    GO = {"HUNT":0,"BREAKOUT":1,"TREND":2}
    primary.sort(key=lambda x: (GO.get(x.get("grade","T"),3), -x.get("finalScore",0)))
    primary = primary[:9]

    # 3단계: 각 매수 후보 → 5영업일 추적
    for r in primary:
        code = r["code"]
        df_full = ohlcv_map.get(code)
        if df_full is None: continue
        last_close = int(df_full[df_full.index <= td]["Close"].iloc[-1])
        last_low = int(df_full[df_full.index <= td]["Low"].iloc[-1])
        atr_v = r.get("atr", 0)
        if atr_v <= 0: continue
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

        all_trades.append({
            "date": str(td.date()), "code": code, "name": r["name"],
            "grade": r["grade"], "isAlt": r.get("isAlt", False),
            "buy": buy, "target": target, "stop": stop,
            "exit_day": exit_day, "exit_reason": exit_reason,
            "exit_price": exit_price, "profit_pct": round(profit_pct, 2),
            "finalScore": r.get("finalScore", 0),
            "rsi": float(r.get("rsi", 0)),
            "sector": r.get("sector", "-"),
            "sectorBonus": r.get("sectorBonus", 0),
        })

log(f"\n총 trades: {len(all_trades)}  T+{time.time()-T0:.0f}s")

# 5. 통계
log("\n" + "="*72)
log("  📊 통계 분석")
log("="*72)

def stats(rows, label):
    if not rows: return
    n = len(rows)
    wins = sum(1 for r in rows if r["exit_reason"] == "target")
    stops_ = sum(1 for r in rows if r["exit_reason"] == "stop")
    expire = sum(1 for r in rows if r["exit_reason"] == "expire")
    pl = [r["profit_pct"] for r in rows]
    avg, med = np.mean(pl), np.median(pl)
    total = sum(pl)
    win_only_avg = np.mean([r["profit_pct"] for r in rows if r["exit_reason"] == "target"]) if wins > 0 else 0
    log(f"\n[{label}] n={n}")
    log(f"  🎯 청산도달(+10%): {wins} ({wins/n*100:.1f}%)")
    log(f"  🛑 손절:           {stops_} ({stops_/n*100:.1f}%)")
    log(f"  ⏰ 만기:           {expire} ({expire/n*100:.1f}%)")
    log(f"  평균수익: {avg:+.2f}%  중간값 {med:+.2f}%")
    log(f"  누적수익: {total:+.2f}% (단일종목 비중 무시 단순합)")

stats(all_trades, "전체")
for g in ["HUNT", "BREAKOUT", "TREND"]:
    stats([t for t in all_trades if t["grade"] == g], g)
stats([t for t in all_trades if not t.get("isAlt")], "메인 (HUNT/BREAKOUT)")
stats([t for t in all_trades if t.get("isAlt")], "차선 (TREND)")

# 6. 일별 성과
log("\n" + "="*72)
log("  📅 일별 성과")
log("="*72)
log(f"\n{'날짜':<12} {'후보수':>5} {'청산':>4} {'손절':>4} {'만기':>4} {'평균수익':>9}")
log("-"*60)
days = sorted(set(t["date"] for t in all_trades))
for d in days:
    rows = [t for t in all_trades if t["date"] == d]
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")
    s = sum(1 for r in rows if r["exit_reason"] == "stop")
    e = sum(1 for r in rows if r["exit_reason"] == "expire")
    avg = np.mean([r["profit_pct"] for r in rows])
    log(f"{d:<12} {n:>5} {w:>4} {s:>4} {e:>4} {avg:>+8.2f}%")

# 7. 가장 큰 수익/손실
log("\n" + "="*72)
log("  🏆 TOP 5 (수익) / WORST 5 (손실)")
log("="*72)
sorted_t = sorted(all_trades, key=lambda x: -x["profit_pct"])
log("\n[TOP 5 수익]")
log(f"{'날짜':<12} {'종목':<14} {'등급':<10} {'매수':>9} {'청산일':>4} {'수익':>7}")
for t in sorted_t[:5]:
    log(f"{t['date']:<12} {t['name']:<14} {t['grade']:<10} {t['buy']:>9,} {t['exit_day']:>4} {t['profit_pct']:>+6.2f}%")
log("\n[WORST 5 손실]")
for t in sorted_t[-5:]:
    log(f"{t['date']:<12} {t['name']:<14} {t['grade']:<10} {t['buy']:>9,} {t['exit_day']:>4} {t['profit_pct']:>+6.2f}%")

# 8. 50만원 시작 복리 시뮬레이션
log("\n" + "="*72)
log("  💰 50만원 시드머니 복리 시뮬레이션")
log("="*72)
log("[가정: 매일 매수 후보 중 1순위 1종목만 매수, 5일 보유 후 차익만 적립]")

# 날짜별 1순위만 추출
date_top = {}
for t in all_trades:
    if t["date"] not in date_top:
        date_top[t["date"]] = t

balance = 500000.0
trades_done = 0
total_profit_won = 0
for d in sorted(date_top.keys()):
    t = date_top[d]
    profit_won = balance * t["profit_pct"] / 100
    balance += profit_won
    total_profit_won += profit_won
    trades_done += 1

log(f"\n시작: 500,000원")
log(f"총 거래: {trades_done}회 (영업일별 1순위만)")
log(f"최종 잔고: {balance:,.0f}원")
log(f"누적 수익: {total_profit_won:+,.0f}원 ({(balance-500000)/500000*100:+.2f}%)")

# 9. 섹터별
log("\n" + "="*72)
log("  🏭 섹터별 성과")
log("="*72)
log(f"\n{'섹터':<14} {'n':>4} {'청산':>5} {'평균':>7}")
log("-"*40)
sec_stats = {}
for t in all_trades:
    sec = t["sector"]
    sec_stats.setdefault(sec, []).append(t)
for sec in sorted(sec_stats.keys(), key=lambda x: -len(sec_stats[x])):
    rows = sec_stats[sec]
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")/n*100
    avg = np.mean([r["profit_pct"] for r in rows])
    log(f"{sec:<14} {n:>4} {w:>4.1f}% {avg:>+6.2f}%")

# 저장
with open("backtest_april_may_result.json", "w", encoding="utf-8") as f:
    json.dump({"period": f"{test_dates[0].date()}~{test_dates[-1].date()}",
               "total_trades": len(all_trades),
               "elapsed_sec": int(time.time()-T0),
               "trades": all_trades}, f, ensure_ascii=False, default=str)
log(f"\n저장: backtest_april_may_result.json  소요: {time.time()-T0:.0f}s")
