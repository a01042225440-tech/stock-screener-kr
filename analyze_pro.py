"""PRO/BUY/BB_BREAK 등급 상세 분석"""
import json

with open("latest_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]
print(f"\n=== 날짜: {data['date']} | 총 {len(results)}종목 매칭 ===\n")

# 등급별
from collections import Counter
g_counts = Counter(r.get("grade") for r in results)
print("등급별 개수:", dict(g_counts))
print()

def show_stocks(label, lst, limit=10):
    print(f"\n=== {label} ({len(lst)}개) ===")
    for r in lst[:limit]:
        bb = ""
        if r.get("bbBreakClose"): bb = " 🟣BB돌파(종가)"
        elif r.get("bbBreakIntraday"): bb = " 🟣BB돌파(인트라)"
        miss = []
        for k, lbl in [("a1_52w","A1"),("a2_mom6m","A2"),("a3_vcp","A3"),("a4_volprice","A4"),
                       ("a5_pead","A5"),("a6_obv","A6"),("a7_trend","A7"),("a8_lowvol","A8"),
                       ("a9_rsi","A9"),("a10_base","A10")]:
            if not r.get(k): miss.append(lbl)
        print(f"  {r['name']:<12s} ({r['code']}) 종가={r['close']:>7,}  P1={r.get('p1Score',0)}/10  PRO={r.get('proCount',0)}/10  "
              f"Sec={r.get('sector','-'):<8s}  Final={r.get('finalScore',0):>5.0f}{bb}")
        if miss:
            print(f"    미달인자: {miss}")
        if r.get('a2_mom6m') is not None:
            print(f"    6M수익률={r.get('ret6m',0):+.1f}%  60일변동={r.get('annualVol60',0):.0f}%  거래량={r.get('volMult',0):.1f}x  당일={r.get('chgToday',0):+.1f}%")

show_stocks("🟢 PRO 등급", [r for r in results if r.get("grade")=="PRO"])
show_stocks("🔴 BUY 등급", [r for r in results if r.get("grade")=="BUY"])
show_stocks("🟣 BB_BREAK 단독 등급 (P1<3이지만 BB하단 돌파)", [r for r in results if r.get("grade")=="BB_BREAK"], 15)

# BB 돌파 통계
bb_close = [r for r in results if r.get("bbBreakClose")]
bb_intra = [r for r in results if r.get("bbBreakIntraday") and not r.get("bbBreakClose")]
print(f"\n=== BB 하단 돌파 신호 통계 ===")
print(f"  종가 돌파 (전일종가<BB하단, 당일종가>BB하단): {len(bb_close)}개")
print(f"  인트라데이 돌파 (저가<BB하단, 종가>BB하단): {len(bb_intra)}개")
print(f"  총 BB하단 돌파: {len(bb_close)+len(bb_intra)}개")

# 등급×BB돌파 교차
print(f"\n=== BB하단돌파 종목의 등급 분포 ===")
bb_results = [r for r in results if r.get("bbLowerBreak")]
bb_g = Counter(r.get("grade") for r in bb_results)
for g, n in sorted(bb_g.items(), key=lambda x:["PRO","BUY","WATCH","COILING","BB_BREAK"].index(x[0]) if x[0] in ["PRO","BUY","WATCH","COILING","BB_BREAK"] else 99):
    print(f"  {g}: {n}개")
