"""청산 % 그리드 시뮬레이션 — 새 조건 4/1~5/27 trades로
청산 5~15% × 분산 1~5종목 × 손절 -5% 고정"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app import naver_ohlcv_fast
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

with open("backtest_april_may_result.json","r",encoding="utf-8") as f:
    data = json.load(f)
trades = data["trades"]
print(f"\n로드: {len(trades)} trades (새 조건)")

# OHLCV 부착
codes = sorted(set(t["code"] for t in trades))
ohlcv = {}
def fetch(c):
    if c in ohlcv: return
    df = naver_ohlcv_fast(c, days=300, target_date="2026-06-05")
    if df is not None: ohlcv[c] = df
with ThreadPoolExecutor(max_workers=20) as ex:
    list(ex.map(fetch, codes))
print(f"OHLCV: {len(ohlcv)}/{len(codes)}")

for t in trades:
    df = ohlcv.get(t["code"])
    if df is None: continue
    buy_dt = pd.Timestamp(t["date"])
    df_after = df[df.index > buy_dt].head(5)
    t["bars"] = [{"high":int(r["High"]),"low":int(r["Low"]),"close":int(r["Close"])}
                 for _,r in df_after.iterrows()]

# 시뮬레이션 함수
def sim(rows, target_pct, stop_pct=-5):
    out = []
    for t in rows:
        if not t.get("bars"): continue
        buy = t["buy"]
        target = buy * (1 + target_pct/100)
        stop = buy * (1 + stop_pct/100)
        exit_p = None; reason = None
        for b in t["bars"]:
            if exit_p is None:
                if b["high"] >= target:
                    exit_p, reason = target, "target"
                elif b["low"] <= stop:
                    exit_p, reason = stop, "stop"
        if exit_p is None and t["bars"]:
            exit_p, reason = t["bars"][-1]["close"], "expire"
        if exit_p is None: continue
        out.append({"date":t["date"],"profit":(exit_p-buy)/buy*100,"reason":reason})
    return out

# 분산 복리
def compound(rows, n_split):
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    bal = 500000.0
    for d in sorted(by_date.keys()):
        group = by_date[d][:n_split]
        if not group: continue
        avg = np.mean([r["profit"] for r in group])
        bal *= (1 + avg/100)
    return bal

# 메인: 청산 % × 분산 매트릭스
print("\n" + "="*78)
print("  🎯 청산 % 변경 매트릭스 (손절 -5% 고정, 50만원 시드)")
print("="*78)
print(f"\n{'청산':<6} {'청산률':>7} {'평균':>7} {'1종목복리':>10} {'2종목':>10} {'3종목':>10} {'5종목':>10}")
print("-"*78)

best_compound = 0
best_combo = ""
for tp in [5, 6, 7, 8, 9, 10, 11, 12, 15]:
    res = sim(trades, tp)
    n = len(res)
    if not n: continue
    w = sum(1 for r in res if r["reason"]=="target")/n*100
    avg = np.mean([r["profit"] for r in res])
    b1 = compound(res, 1)
    b2 = compound(res, 2)
    b3 = compound(res, 3)
    b5 = compound(res, 5)
    print(f"+{tp:>2}%   {w:>6.1f}% {avg:>+6.2f}% {(b1-500000)/500000*100:>+9.2f}% {(b2-500000)/500000*100:>+9.2f}% {(b3-500000)/500000*100:>+9.2f}% {(b5-500000)/500000*100:>+9.2f}%")
    for n_, b in [(1,b1),(2,b2),(3,b3),(5,b5)]:
        if b > best_compound:
            best_compound = b
            best_combo = f"청산+{tp}%/손절-5%/{n_}종목분산"

print(f"\n🏆 최적: {best_combo} → 최종 {best_compound:,.0f}원 ({(best_compound-500000)/500000*100:+.2f}%)")

# 추가: 다양한 손절 × 청산 매트릭스
print("\n" + "="*78)
print("  💎 청산 % × 손절 % 매트릭스 (3종목 분산, 복리 수익률)")
print("="*78)
print(f"\n{'손절\\청산':<10}", end="")
for tp in [5, 6, 7, 8, 10, 12, 15]:
    print(f"{f'+{tp}%':>9}", end="")
print()
print("-"*70)
for sp in [-3, -4, -5, -6, -7, -10]:
    print(f"{sp:>+3}%      ", end="")
    for tp in [5, 6, 7, 8, 10, 12, 15]:
        res = sim(trades, tp, sp)
        b3 = compound(res, 3)
        print(f"{(b3-500000)/500000*100:>+8.2f}%", end="")
    print()
