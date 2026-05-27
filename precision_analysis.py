"""
정밀 분석 — 1년 백테스트 데이터로 수익률 최대화 방법 찾기

1. 각 trade의 매수일 이후 5일치 OHLCV 다시 가져오기
2. 모든 (청산%, 손절%, 보유일) 조합 시뮬레이션 → 최적 매매룰
3. 조건 조합 효과 분석 (단일이 아닌 AND 조합)
4. 수익률 분포 정밀 분석
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app import naver_ohlcv_fast
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations

T0 = time.time()

# === 1년 백테스트 결과 로드 ===
with open("backtest_1year_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)
trades = data["trades"]
print(f"\n로드: {len(trades)} trades")

# === 1. 각 trade에 매수 후 5일치 OHLCV 부착 ===
print("\n[1단계] 매수 후 5일치 OHLCV 부착...")

# unique (code, buy_date) 페어 추출
unique_keys = sorted(set((t["code"], t["date"]) for t in trades))
print(f"  unique (code, date): {len(unique_keys)}")

ohlcv_cache = {}
def fetch_for_trade(code):
    """종목당 1번만 OHLCV fetch"""
    if code in ohlcv_cache: return
    df = naver_ohlcv_fast(code, days=400, target_date="2026-05-26")
    if df is not None:
        ohlcv_cache[code] = df

unique_codes = sorted(set(c for c, _ in unique_keys))
print(f"  unique codes: {len(unique_codes)}")
with ThreadPoolExecutor(max_workers=20) as ex:
    list(ex.map(fetch_for_trade, unique_codes))
print(f"  OHLCV 확보: {len(ohlcv_cache)}  T+{time.time()-T0:.0f}s")

# 각 trade에 매수+1 ~ +5일 OHLCV 부착
enriched = []
for t in trades:
    df = ohlcv_cache.get(t["code"])
    if df is None: continue
    buy_dt = pd.Timestamp(t["date"])
    df_after = df[df.index > buy_dt].head(5)
    if len(df_after) < 1: continue
    buy = t["buy"]
    bars = []
    for i, (idx, row) in enumerate(df_after.iterrows(), 1):
        bars.append({"day": i, "open": int(row["Open"]), "high": int(row["High"]),
                     "low": int(row["Low"]), "close": int(row["Close"])})
    # 매수 후 일별 수익률 (고가, 저가, 종가)
    max_high = max(b["high"] for b in bars)
    min_low  = min(b["low"] for b in bars)
    max_high_pct = (max_high - buy) / buy * 100
    min_low_pct  = (min_low - buy) / buy * 100
    close_5d_pct = (bars[-1]["close"] - buy) / buy * 100
    t2 = {**t, "bars": bars,
          "max_high_pct": round(max_high_pct, 2),
          "min_low_pct": round(min_low_pct, 2),
          "close_5d_pct": round(close_5d_pct, 2),
          "days_held": len(bars)}
    enriched.append(t2)

print(f"  enriched trades: {len(enriched)}  T+{time.time()-T0:.0f}s")

# === 2. 매매룰 그리드 서치 (청산% × 손절% × 보유일) ===
print("\n[2단계] 매매룰 그리드 서치...")

def simulate(rows, target_pct, stop_pct, max_hold):
    """청산%, 손절%, 보유일 조합으로 시뮬레이션"""
    results = []
    for t in rows:
        buy = t["buy"]
        target = buy * (1 + target_pct / 100)
        stop = buy * (1 + stop_pct / 100)
        bars = t["bars"][:max_hold]
        exit_p = None; exit_day = None; reason = None
        for i, b in enumerate(bars, 1):
            if b["high"] >= target:
                exit_p = target; exit_day = i; reason = "target"; break
            elif b["low"] <= stop:
                exit_p = stop; exit_day = i; reason = "stop"; break
        if exit_p is None:
            exit_p = bars[-1]["close"]; exit_day = len(bars); reason = "expire"
        profit_pct = (exit_p - buy) / buy * 100
        results.append({"profit": profit_pct, "day": exit_day, "reason": reason})
    n = len(results)
    if n == 0: return None
    avg = np.mean([r["profit"] for r in results])
    med = np.median([r["profit"] for r in results])
    wins = sum(1 for r in results if r["profit"] >= target_pct - 0.5)
    stops = sum(1 for r in results if r["reason"] == "stop")
    expire = sum(1 for r in results if r["reason"] == "expire")
    avg_day = np.mean([r["day"] for r in results])
    return {"avg": avg, "med": med, "wins": wins, "win_rate": wins/n*100,
            "stops": stops, "stop_rate": stops/n*100,
            "expire_rate": expire/n*100, "avg_day": avg_day, "n": n}

# 그리드: 청산 5,7,10,15,20% × 손절 -3,-5,-7,-10% × 보유 3,5,7일
print("\n=== 매매룰 그리드 서치 (전체 trades) ===")
print(f"{'청산%':>6} {'손절%':>6} {'보유일':>6} {'평균수익':>10} {'중간값':>8} {'승률':>7} {'손절률':>7}")
print("-"*72)

best = None
for tp in [5, 7, 10, 12, 15, 20]:
    for sp in [-3, -5, -7, -10]:
        for mh in [3, 5, 7]:
            r = simulate(enriched, tp, sp, mh)
            if r is None: continue
            print(f"+{tp:>3}%   {sp:>+4}%   {mh:>3}일  {r['avg']:>+8.2f}%  {r['med']:>+6.2f}%  {r['win_rate']:>5.1f}%  {r['stop_rate']:>5.1f}%")
            if best is None or r["avg"] > best["avg"]:
                best = {**r, "tp": tp, "sp": sp, "mh": mh}

print(f"\n🏆 최적 (평균수익 기준): 청산 +{best['tp']}% / 손절 {best['sp']}% / 보유 {best['mh']}일")
print(f"   평균수익 {best['avg']:+.2f}% / 승률 {best['win_rate']:.1f}% / 손절률 {best['stop_rate']:.1f}%")

# === 3. HUNT/BREAKOUT 별 그리드 ===
for grade in ["HUNT", "BREAKOUT"]:
    g_trades = [t for t in enriched if t["grade"] == grade]
    if not g_trades: continue
    print(f"\n=== {grade} ({len(g_trades)} trades) ===")
    print(f"{'청산%':>6} {'손절%':>6} {'보유일':>6} {'평균수익':>10} {'승률':>7} {'손절률':>7}")
    print("-"*60)
    g_best = None
    for tp in [5, 7, 10, 12, 15]:
        for sp in [-5, -7, -10]:
            for mh in [3, 5, 7]:
                r = simulate(g_trades, tp, sp, mh)
                if r is None: continue
                if g_best is None or r["avg"] > g_best["avg"]:
                    g_best = {**r, "tp": tp, "sp": sp, "mh": mh}
    # Top 5만 출력
    top_rs = []
    for tp in [5, 7, 10, 12, 15]:
        for sp in [-5, -7, -10]:
            for mh in [3, 5, 7]:
                r = simulate(g_trades, tp, sp, mh)
                if r: top_rs.append({**r, "tp": tp, "sp": sp, "mh": mh})
    top_rs.sort(key=lambda x: -x["avg"])
    for r in top_rs[:5]:
        print(f"+{r['tp']:>3}%   {r['sp']:>+4}%   {r['mh']:>3}일  {r['avg']:>+8.2f}%  {r['win_rate']:>5.1f}%  {r['stop_rate']:>5.1f}%")
    print(f"  🏆 {grade} 최적: +{g_best['tp']}% / {g_best['sp']}% / {g_best['mh']}일 → 평균 {g_best['avg']:+.2f}%")

# === 4. 조건 조합 분석 ===
print("\n[3단계] 조건 조합 효과 분석...")
print("\n=== 가장 강력한 조건 2개 조합 (HUNT 등급) ===")
hunt_trades = [t for t in enriched if t["grade"] == "HUNT"]
KEY_CONDS = ["a4_ma200up", "d4_vol_pickup", "d2_above_ma5", "e3_price_up",
             "e2_vol_burst", "inLeadingSector", "k3_short_wick"]

combo_results = []
for c1, c2 in combinations(KEY_CONDS, 2):
    pass_rows = [t for t in hunt_trades if t.get(c1) and t.get(c2)]
    if len(pass_rows) < 10: continue
    avg = np.mean([t["close_5d_pct"] for t in pass_rows])
    max_high_avg = np.mean([t["max_high_pct"] for t in pass_rows])
    win_pct10 = sum(1 for t in pass_rows if t["max_high_pct"] >= 10) / len(pass_rows) * 100
    combo_results.append({"combo": f"{c1} + {c2}", "n": len(pass_rows),
                          "avg_close": avg, "avg_high": max_high_avg, "win10": win_pct10})
combo_results.sort(key=lambda x: -x["win10"])

print(f"{'조합':<42} {'n':>4} {'평균종가':>9} {'평균고점':>9} {'+10%도달':>9}")
print("-"*80)
for r in combo_results[:10]:
    print(f"{r['combo']:<42} {r['n']:>4} {r['avg_close']:>+8.2f}% {r['avg_high']:>+8.2f}% {r['win10']:>7.1f}%")

# === 5. 매수 후 일자별 평균 수익 추세 ===
print("\n=== 매수 후 일자별 평균 고점/저점 수익률 ===")
print(f"{'일자':<6} {'평균종가':>9} {'평균고점':>9} {'평균저점':>9}")
print("-"*48)
for day in range(1, 6):
    day_rows = [t for t in enriched if len(t["bars"]) >= day]
    if not day_rows: continue
    avg_close = np.mean([(t["bars"][day-1]["close"] - t["buy"]) / t["buy"] * 100 for t in day_rows])
    avg_high  = np.mean([(t["bars"][day-1]["high"] - t["buy"]) / t["buy"] * 100 for t in day_rows])
    avg_low   = np.mean([(t["bars"][day-1]["low"] - t["buy"]) / t["buy"] * 100 for t in day_rows])
    print(f"+{day}일   {avg_close:>+8.2f}% {avg_high:>+8.2f}% {avg_low:>+8.2f}%")

# === 6. 매수 후 max_high 분포 ===
print("\n=== 매수 후 5일 내 최대 상승률 분포 ===")
buckets = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 50)]
for lo, hi in buckets:
    rows = [t for t in enriched if lo <= t["max_high_pct"] < hi]
    pct = len(rows) / len(enriched) * 100
    print(f"  최대고점 {lo:>+3}~{hi:>+3}%: {len(rows):>4}건 ({pct:>4.1f}%)")

# === 7. min_low 분포 (손절폭 검토) ===
print("\n=== 매수 후 5일 내 최저 하락률 분포 ===")
buckets = [(-50, -10), (-10, -7), (-7, -5), (-5, -3), (-3, 0), (0, 50)]
for lo, hi in buckets:
    rows = [t for t in enriched if lo <= t["min_low_pct"] < hi]
    pct = len(rows) / len(enriched) * 100
    print(f"  최저저가 {lo:>+4}~{hi:>+4}%: {len(rows):>4}건 ({pct:>4.1f}%)")

# === 8. 핵심 결론 ===
print("\n" + "="*72)
print("  🎯 정밀 분석 핵심 결론")
print("="*72)
print(f"""
[최적 매매룰] {best['tp']}% 청산 / {best['sp']}% 손절 / {best['mh']}일 보유
  → 평균수익 {best['avg']:+.2f}%, 승률 {best['win_rate']:.1f}%, 손절률 {best['stop_rate']:.1f}%
  (현재 시스템: +10%/-7%/5일 = 평균 {simulate(enriched, 10, -7, 5)['avg']:+.2f}%)

[현재 vs 최적 차이]
  현재:  평균 {simulate(enriched, 10, -7, 5)['avg']:+.2f}%
  최적:  평균 {best['avg']:+.2f}%
  개선:  {best['avg'] - simulate(enriched, 10, -7, 5)['avg']:+.2f}%p
""")

# 저장
with open("precision_analysis.json", "w", encoding="utf-8") as f:
    json.dump({"best": best, "enriched_count": len(enriched),
               "combos_top5": combo_results[:5],
               "elapsed": time.time()-T0}, f, ensure_ascii=False, default=str)
print(f"\n저장: precision_analysis.json  T={time.time()-T0:.0f}s")
