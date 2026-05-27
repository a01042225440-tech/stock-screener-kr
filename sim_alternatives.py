"""
4/1~5/27 백테스트 결과로 4가지 대안 시뮬레이션:
1) 손절 -5% / -7% / -10% / -15% 시뮬레이션
2) 분산 매수 (매일 N종목 균등) vs 1순위 매수
3) 시장 필터 (KOSPI 하락일 회피)
4) 종합 권고
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app import naver_ohlcv_fast
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 데이터 로드
with open("backtest_april_may_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)
trades = data["trades"]
print(f"\n로드: {len(trades)} trades")

# 각 trade에 5일치 OHLCV 부착 (캐시 활용)
ohlcv_cache = {}
def fetch(code):
    if code in ohlcv_cache: return
    df = naver_ohlcv_fast(code, days=300, target_date="2026-06-05")
    if df is not None:
        ohlcv_cache[code] = df

unique_codes = sorted(set(t["code"] for t in trades))
with ThreadPoolExecutor(max_workers=20) as ex:
    list(ex.map(fetch, unique_codes))
print(f"OHLCV 확보: {len(ohlcv_cache)}/{len(unique_codes)}")

# 각 trade에 5일치 OHLCV 부착
for t in trades:
    df = ohlcv_cache.get(t["code"])
    if df is None: continue
    buy_dt = pd.Timestamp(t["date"])
    df_after = df[df.index > buy_dt].head(5)
    bars = []
    for i, (idx, row) in enumerate(df_after.iterrows(), 1):
        bars.append({"day": i, "open": int(row["Open"]), "high": int(row["High"]),
                     "low": int(row["Low"]), "close": int(row["Close"])})
    t["bars"] = bars

# === 1. 손절 임계 시뮬레이션 ===
print("\n" + "="*72)
print("  🛑 손절 임계별 시뮬레이션 (청산 +10% 고정, 5일 보유)")
print("="*72)
print(f"\n{'손절':<8} {'청산':>4} {'손절':>4} {'만기':>4} {'평균수익':>9} {'중간값':>8} {'복리(1순위)':>12}")
print("-"*72)

def simulate_with_stop(rows, stop_pct):
    """손절 % 변경해서 재시뮬레이션"""
    out = []
    for t in rows:
        if not t.get("bars"): continue
        buy = t["buy"]
        target = buy * 1.10
        stop = buy * (1 + stop_pct / 100)
        exit_p = None; exit_day = None; reason = None
        for i, b in enumerate(t["bars"], 1):
            if exit_p is None:
                if b["high"] >= target:
                    exit_p, exit_day, reason = target, i, "target"
                elif b["low"] <= stop:
                    exit_p, exit_day, reason = stop, i, "stop"
                elif i == len(t["bars"]):
                    exit_p, exit_day, reason = b["close"], i, "expire"
        if exit_p is None: continue
        out.append({"date": t["date"], "code": t["code"], "name": t["name"],
                    "profit_pct": (exit_p - buy) / buy * 100, "reason": reason})
    return out

for sp in [-3, -5, -7, -10, -15]:
    res = simulate_with_stop(trades, sp)
    if not res: continue
    n = len(res)
    w = sum(1 for r in res if r["reason"] == "target")
    s = sum(1 for r in res if r["reason"] == "stop")
    e = sum(1 for r in res if r["reason"] == "expire")
    pl = [r["profit_pct"] for r in res]
    avg = np.mean(pl)
    med = np.median(pl)
    # 복리 시뮬레이션 (매일 1순위)
    date_top = {}
    for r in res:
        if r["date"] not in date_top:
            date_top[r["date"]] = r
    balance = 500000
    for d in sorted(date_top.keys()):
        balance *= (1 + date_top[d]["profit_pct"] / 100)
    print(f"{sp:>+4}%   {w:>4} {s:>4} {e:>4} {avg:>+8.2f}% {med:>+7.2f}% {balance/500000*100-100:>+11.2f}%")

# === 2. 분산 매수 시뮬레이션 ===
print("\n" + "="*72)
print("  📊 분산 매수 시뮬레이션 (손절 -10% 고정)")
print("="*72)

def compound_distributed(rows, n_per_day):
    """매일 상위 n_per_day 종목 균등 매수, 자본 100% 활용"""
    date_groups = {}
    for r in rows:
        date_groups.setdefault(r["date"], []).append(r)
    balance = 500000
    daily_log = []
    for d in sorted(date_groups.keys()):
        day_rows = date_groups[d][:n_per_day]
        if not day_rows: continue
        # 균등 분산: 자본 N등분
        avg_pct = np.mean([r["profit_pct"] for r in day_rows])
        balance *= (1 + avg_pct / 100)
        daily_log.append((d, len(day_rows), avg_pct, balance))
    return balance, daily_log

print(f"\n{'전략':<25} {'최종잔고':>12} {'수익률':>10}")
print("-"*52)
for n in [1, 2, 3, 5]:
    b, _ = compound_distributed(trades, n)
    print(f"매일 상위 {n}종목 균등{'':<8} {b:>11,.0f}원 {(b-500000)/500000*100:>+9.2f}%")

# === 3. 손절 + 분산 조합 ===
print("\n" + "="*72)
print("  🔬 손절 % × 분산 종목수 매트릭스 (복리 수익률)")
print("="*72)
print(f"\n{'손절':<8}", end="")
for n in [1, 2, 3, 5]:
    print(f"{'1순위' if n == 1 else f'{n}종목분산':>10}", end="")
print()
print("-"*50)

for sp in [-3, -5, -7, -10, -15]:
    sim = simulate_with_stop(trades, sp)
    print(f"{sp:>+4}%   ", end="")
    for n in [1, 2, 3, 5]:
        b, _ = compound_distributed(sim, n)
        print(f"{(b-500000)/500000*100:>+9.2f}%", end="")
    print()

# === 4. 시장 필터 (KOSPI 하락일 회피) ===
print("\n" + "="*72)
print("  🌐 KOSPI 시장 필터 효과")
print("="*72)
print("\nKOSPI 일별 데이터 가져오는 중...")
kospi_df = naver_ohlcv_fast("KOSPI", days=200, target_date="2026-06-05")
if kospi_df is not None:
    kospi_df["chg"] = kospi_df["Close"].pct_change() * 100
    # 각 trade의 buy 전일 KOSPI 변동률 확인
    skip_count = 0
    filtered = []
    for t in trades:
        buy_dt = pd.Timestamp(t["date"])
        prev_kospi = kospi_df[kospi_df.index < buy_dt].tail(1)
        if prev_kospi.empty:
            filtered.append(t)
            continue
        prev_chg = prev_kospi["chg"].iloc[0]
        if prev_chg < -1.5:  # 전일 KOSPI -1.5% 이상 하락
            skip_count += 1
            continue
        filtered.append(t)
    print(f"  전일 KOSPI -1.5%↓ 이라 스킵: {skip_count}건")
    print(f"  필터 적용 후 trades: {len(filtered)}")

    # 비교
    for sp in [-10, -7, -5]:
        orig = simulate_with_stop(trades, sp)
        filt = simulate_with_stop(filtered, sp)
        b_orig, _ = compound_distributed(orig, 1)
        b_filt, _ = compound_distributed(filt, 1)
        print(f"  손절{sp:+}% 필터X→1순위복리: {(b_orig-500000)/500000*100:+.2f}%  필터O→{(b_filt-500000)/500000*100:+.2f}%")
else:
    print("  KOSPI 데이터 없음 (건너뜀)")

print("\n저장 완료")
