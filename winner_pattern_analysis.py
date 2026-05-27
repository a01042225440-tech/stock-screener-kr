"""
🎯 청산도달(+10%) vs 손절 종목 정밀 비교 분석
4/1~5/27 백테스트 145 trades + 1년 백테스트 550 trades 통합

목표: 진짜 상승할 종목의 특징 발견 + 새 가중치 시스템 제안
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from collections import defaultdict
from itertools import combinations

print("\n" + "="*72)
print("  🎯 청산 vs 손절 종목 정밀 비교 분석")
print("="*72)

# 두 백테스트 결과 통합
all_trades = []
for path in ["backtest_april_may_result.json", "backtest_1year_result.json"]:
    if not os.path.exists(path): continue
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    for t in d["trades"]:
        all_trades.append(t)

# 중복 제거 (date+code 기준)
seen = set()
trades = []
for t in all_trades:
    key = (t.get("date"), t.get("code"))
    if key in seen: continue
    seen.add(key)
    trades.append(t)

print(f"\n총 통합 trades: {len(trades)}")
wins = [t for t in trades if t["exit_reason"] == "target"]
stops = [t for t in trades if t["exit_reason"] == "stop"]
print(f"  🎯 청산도달: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
print(f"  🛑 손절:    {len(stops)} ({len(stops)/len(trades)*100:.1f}%)")

# === 1. 수치 조건 비교 ===
print("\n" + "="*72)
print("  📊 수치 조건: 청산 vs 손절 그룹 평균")
print("="*72)

NUM_FIELDS = [
    ("finalScore", "최종점수"),
    ("rsi", "RSI"),
    ("ma20Dist", "20MA거리%"),
    ("volMult20", "거래량20일배수"),
    ("aCount", "A정배열개수"),
    ("pullbackCount", "풀백카운트"),
]

print(f"\n{'필드':<14} {'청산평균':>10} {'손절평균':>10} {'차이':>10} {'의미':<30}")
print("-"*80)
for field, label in NUM_FIELDS:
    win_vals = [t.get(field, 0) for t in wins if t.get(field) is not None]
    stop_vals = [t.get(field, 0) for t in stops if t.get(field) is not None]
    if not win_vals or not stop_vals: continue
    w_avg = np.mean([float(v) for v in win_vals])
    s_avg = np.mean([float(v) for v in stop_vals])
    diff = w_avg - s_avg
    note = ""
    if abs(diff) > w_avg * 0.05:
        note = "✓ 의미있는 차이" if diff > 0 else "→ 손절이 더 큼"
    print(f"{label:<14} {w_avg:>9.2f}  {s_avg:>9.2f}  {diff:>+9.2f}  {note}")

# === 2. 불린 조건 통과율 비교 ===
print("\n" + "="*72)
print("  🔬 불린 조건: 청산 그룹 vs 손절 그룹 통과율 차이")
print("="*72)

BOOL_FIELDS = [
    "a1_ma200", "a2_ma60", "a3_align", "a4_ma200up",
    "b1_eps", "b2_per", "b3_target",
    "k1_up_close", "k2_bullish", "k3_short_wick",
    "d1_pullback", "d2_above_ma5", "d3_bullish", "d4_vol_pickup", "d5_rsi_ok",
    "e1_new_high", "e2_vol_burst", "e3_price_up", "e4_rsi_ok", "e5_ma20_near",
    "inLeadingSector", "huntTrigger", "breakoutTrigger",
]

results = []
for field in BOOL_FIELDS:
    win_pass = sum(1 for t in wins if t.get(field) is True) / len(wins) * 100 if wins else 0
    stop_pass = sum(1 for t in stops if t.get(field) is True) / len(stops) * 100 if stops else 0
    diff = win_pass - stop_pass
    results.append({"field": field, "win": win_pass, "stop": stop_pass, "diff": diff})

# 차이가 큰 순서로
results.sort(key=lambda x: -x["diff"])
print(f"\n{'조건':<22} {'청산 통과율':>10} {'손절 통과율':>10} {'차이':>10}")
print("-"*70)
print("[청산에서 더 많이 통과 = 좋은 신호]")
for r in results[:10]:
    mark = "🚀" if r["diff"] > 10 else "↑" if r["diff"] > 5 else " "
    print(f"{r['field']:<22} {r['win']:>9.1f}% {r['stop']:>9.1f}% {r['diff']:>+9.1f}% {mark}")
print("\n[손절에서 더 많이 통과 = 함정 신호]")
for r in results[-5:]:
    mark = "⚠️" if r["diff"] < -5 else " "
    print(f"{r['field']:<22} {r['win']:>9.1f}% {r['stop']:>9.1f}% {r['diff']:>+9.1f}% {mark}")

# === 3. 조건 조합 정밀 분석 ===
print("\n" + "="*72)
print("  💎 조건 조합 정밀 분석 (n≥15, 청산률 높은 순)")
print("="*72)

# 가장 영향력 큰 조건 8개로 조합
TOP_CONDS = [r["field"] for r in results[:8]]
combo_stats = []
for size in [2, 3]:
    for combo in combinations(TOP_CONDS, size):
        match = [t for t in trades
                 if all(t.get(c) is True for c in combo)
                 and t["exit_reason"] in ("target", "stop")]
        if len(match) < 15: continue
        w = sum(1 for t in match if t["exit_reason"] == "target")
        win_rate = w / len(match) * 100
        avg = np.mean([t["profit_pct"] for t in match])
        combo_stats.append({"combo": combo, "n": len(match),
                            "win_rate": win_rate, "avg": avg})

combo_stats.sort(key=lambda x: -x["win_rate"])
print(f"\n{'조합':<55} {'n':>4} {'청산률':>8} {'평균':>8}")
print("-"*85)
for c in combo_stats[:15]:
    label = " + ".join(c["combo"])[:55]
    print(f"{label:<55} {c['n']:>4} {c['win_rate']:>7.1f}% {c['avg']:>+7.2f}%")

# === 4. RSI 구간별 정밀 분석 ===
print("\n" + "="*72)
print("  📈 RSI 구간별 청산률 (모든 trades)")
print("="*72)
print(f"\n{'RSI':<10} {'n':>5} {'청산':>5} {'손절':>5} {'청산률':>8} {'평균수익':>9}")
print("-"*60)
for lo, hi in [(0,30),(30,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,100)]:
    rows = [t for t in trades if lo <= float(t.get("rsi",0)) < hi]
    if not rows: continue
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")
    s = sum(1 for r in rows if r["exit_reason"] == "stop")
    avg = np.mean([r["profit_pct"] for r in rows])
    print(f"{lo:>3}~{hi:>3}    {n:>5} {w:>5} {s:>5} {w/n*100:>7.1f}% {avg:>+8.2f}%")

# === 5. 20MA 거리 구간 ===
print("\n" + "="*72)
print("  📐 20MA 거리 구간별 청산률")
print("="*72)
print(f"\n{'거리':<10} {'n':>5} {'청산':>5} {'손절':>5} {'청산률':>8} {'평균수익':>9}")
print("-"*60)
for lo, hi in [(-30,-10),(-10,-5),(-5,-2),(-2,0),(0,2),(2,5),(5,10),(10,15),(15,30)]:
    rows = [t for t in trades if lo <= float(t.get("ma20Dist",0)) < hi]
    if not rows: continue
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")
    s = sum(1 for r in rows if r["exit_reason"] == "stop")
    avg = np.mean([r["profit_pct"] for r in rows])
    print(f"{lo:>+3}~{hi:>+3}%   {n:>5} {w:>5} {s:>5} {w/n*100:>7.1f}% {avg:>+8.2f}%")

# === 6. 거래량 배수 구간 ===
print("\n" + "="*72)
print("  📊 거래량(20일평균 배수) 구간")
print("="*72)
print(f"\n{'배수':<10} {'n':>5} {'청산':>5} {'손절':>5} {'청산률':>8} {'평균수익':>9}")
print("-"*60)
for lo, hi in [(0,1),(1,1.5),(1.5,2),(2,3),(3,5),(5,10),(10,50)]:
    rows = [t for t in trades if lo <= float(t.get("volMult20",0)) < hi]
    if not rows: continue
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")
    s = sum(1 for r in rows if r["exit_reason"] == "stop")
    avg = np.mean([r["profit_pct"] for r in rows])
    print(f"{lo:.1f}~{hi:.1f}x  {n:>5} {w:>5} {s:>5} {w/n*100:>7.1f}% {avg:>+8.2f}%")

# === 7. finalScore 구간 ===
print("\n" + "="*72)
print("  💯 finalScore 구간별 청산률")
print("="*72)
print(f"\n{'점수':<10} {'n':>5} {'청산':>5} {'손절':>5} {'청산률':>8} {'평균수익':>9}")
print("-"*60)
for lo, hi in [(0,80),(80,100),(100,110),(110,120),(120,130),(130,150),(150,200)]:
    rows = [t for t in trades if lo <= float(t.get("finalScore",0)) < hi]
    if not rows: continue
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")
    s = sum(1 for r in rows if r["exit_reason"] == "stop")
    avg = np.mean([r["profit_pct"] for r in rows])
    print(f"{lo:>3}~{hi:>3}   {n:>5} {w:>5} {s:>5} {w/n*100:>7.1f}% {avg:>+8.2f}%")

# === 8. 등급별 ===
print("\n" + "="*72)
print("  🏷️ 등급별 청산률")
print("="*72)
for g in ["HUNT", "BREAKOUT", "TREND", "BB_BREAK"]:
    rows = [t for t in trades if t.get("grade") == g]
    if not rows: continue
    n = len(rows)
    w = sum(1 for r in rows if r["exit_reason"] == "target")
    s = sum(1 for r in rows if r["exit_reason"] == "stop")
    avg = np.mean([r["profit_pct"] for r in rows])
    print(f"  {g:<10} n={n:>4} 청산률 {w/n*100:>5.1f}% 손절률 {s/n*100:>5.1f}% 평균 {avg:>+6.2f}%")

# === 9. 최적 진입 조건 발견 ===
print("\n" + "="*72)
print("  🏆 최적 진입 조건 발견 (청산률 60%+ 조합)")
print("="*72)
golden = [c for c in combo_stats if c["win_rate"] >= 60]
if golden:
    for c in golden[:10]:
        label = " + ".join(c["combo"])
        print(f"  ✨ {label}")
        print(f"     n={c['n']}, 청산률={c['win_rate']:.1f}%, 평균수익={c['avg']:+.2f}%")
else:
    print("\n  60% 이상 조합 없음. 50%+ 조합 표시:")
    for c in [x for x in combo_stats if x["win_rate"] >= 50][:5]:
        label = " + ".join(c["combo"])
        print(f"  ★ {label} - n={c['n']}, 청산률={c['win_rate']:.1f}%, 평균={c['avg']:+.2f}%")

# === 10. 종합 권고 ===
print("\n" + "="*72)
print("  🎯 종합 결론 - 상승 종목 선별 핵심 발견")
print("="*72)

# 가장 강력한 신호 TOP 3
top3 = sorted(results, key=lambda x: -x["diff"])[:3]
print("\n[가장 강력한 단일 신호 TOP 3 (청산 vs 손절 통과율 차이)]")
for r in top3:
    print(f"  → {r['field']}: 청산 {r['win']:.1f}% vs 손절 {r['stop']:.1f}% (차이 +{r['diff']:.1f}%)")

# 가장 약한 신호 (역효과 가능)
worst = sorted(results, key=lambda x: x["diff"])[:3]
print("\n[역효과 의심 신호 (손절에서 더 많이 통과)]")
for r in worst:
    print(f"  → {r['field']}: 청산 {r['win']:.1f}% vs 손절 {r['stop']:.1f}% (차이 {r['diff']:+.1f}%)")

# 황금 조합
if combo_stats:
    best = combo_stats[0]
    print(f"\n[황금 조합]: {' + '.join(best['combo'])}")
    print(f"  → n={best['n']}, 청산률={best['win_rate']:.1f}%, 평균수익={best['avg']:+.2f}%")

# 저장
with open("winner_pattern.json", "w", encoding="utf-8") as f:
    json.dump({"single_signals": results, "combos_top10": combo_stats[:10]},
              f, ensure_ascii=False, default=str)
print(f"\n저장: winner_pattern.json")
