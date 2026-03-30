#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BB Lower Bounce 검증 백테스트
- 2026-01 ~ 2026-03-25 매 거래일 스캔
- 1위 종목 종가매수 (50만원)
- 5%+ 상승 성공률 + 실매매 시뮬레이션
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app import run_scan, naver_ohlcv_fast, save_cache_to_disk
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

KR_HOLIDAYS = {"2026-01-01","2026-01-27","2026-01-28","2026-01-29",
               "2026-02-17","2026-02-18","2026-02-19","2026-03-01"}

def trading_days(start, end):
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in KR_HOLIDAYS:
            days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days

def check_stock_performance(code, name, buy_date, buy_price, target1, target2, stoploss, atr):
    """매수 후 실제 주가 추적: 5%/10% 도달 여부 + 매매 시뮬레이션"""
    df = naver_ohlcv_fast(code, 300, target_date=None)
    if df is None or len(df) < 10:
        return None

    buy_ts = pd.Timestamp(buy_date)
    future = df[df.index > buy_ts].iloc[:25]
    if len(future) < 1:
        return None

    # ── 단순 성과 체크 ──
    max_high = 0
    min_low = 999999999
    max_pct = -999
    close_5d = None
    close_10d = None

    for j, (dt, row) in enumerate(future.iterrows()):
        pct = (row["High"] - buy_price) / buy_price * 100
        if pct > max_pct:
            max_pct = pct
        if row["High"] > max_high:
            max_high = row["High"]
        if row["Low"] < min_low:
            min_low = row["Low"]
        if j == 4:
            close_5d = row["Close"]
        if j == 9:
            close_10d = row["Close"]

    hit_5pct = bool(max_pct >= 5.0)
    hit_10pct = bool(max_pct >= 10.0)
    max_drawdown = (min_low - buy_price) / buy_price * 100

    # ── 실매매 시뮬레이션 ──
    shares = int(500000 // buy_price)
    if shares <= 0:
        return None
    cost = shares * buy_price
    remaining = shares
    realized = 0.0
    current_sl = stoploss
    t1_hit = False
    t2_hit = False
    highest = buy_price
    exit_reason = ""
    exit_date = ""
    exit_pct = 0

    for j, (dt, row) in enumerate(future.iterrows()):
        if remaining <= 0:
            break
        day_str = dt.strftime("%Y-%m-%d")

        # 손절
        if row["Low"] <= current_sl:
            realized += remaining * current_sl
            exit_reason = "손절"
            exit_date = day_str
            remaining = 0
            break

        # T1
        if not t1_hit and row["High"] >= target1:
            t1_hit = True
            sell_qty = max(1, shares // 2)  # 1/2 매도
            if sell_qty > remaining: sell_qty = remaining
            realized += sell_qty * target1
            remaining -= sell_qty
            current_sl = buy_price
            highest = row["High"]
            if remaining <= 0:
                exit_reason = "T1전량"; exit_date = day_str; break

        # T2
        if t1_hit and not t2_hit and row["High"] >= target2:
            t2_hit = True
            sell_qty = remaining
            realized += sell_qty * target2
            remaining = 0
            exit_reason = "T2전량"; exit_date = day_str; break

        # 트레일링
        if t1_hit:
            if row["High"] > highest: highest = row["High"]
            trail = highest - 1.5 * atr
            if trail > current_sl: current_sl = trail

        # 마지막날
        if j == len(future) - 1 and remaining > 0:
            realized += remaining * row["Close"]
            exit_reason = "만기"
            exit_date = day_str
            remaining = 0

    if remaining > 0:
        last_c = future.iloc[-1]["Close"]
        realized += remaining * last_c
        exit_reason = "보유중"
        exit_date = future.index[-1].strftime("%Y-%m-%d")

    profit = realized - cost
    profit_pct = (profit / cost) * 100

    return {
        "code": code, "name": name, "buy_date": buy_date,
        "buy_price": buy_price, "shares": shares, "cost": cost,
        "max_pct": round(max_pct, 1), "max_drawdown": round(max_drawdown, 1),
        "hit_5pct": hit_5pct, "hit_10pct": hit_10pct,
        "close_5d": close_5d, "close_10d": close_10d,
        "t1_hit": t1_hit, "t2_hit": t2_hit,
        "exit_reason": exit_reason, "exit_date": exit_date,
        "profit": round(profit), "profit_pct": round(profit_pct, 1),
    }


def main():
    start = datetime(2026, 1, 2)
    end = datetime(2026, 3, 25)
    days = trading_days(start, end)

    print("=" * 80)
    print(f"  BB Lower Bounce 정확도 검증 백테스트")
    print(f"  기간: {days[0]} ~ {days[-1]} ({len(days)} 거래일)")
    print(f"  기준: 매일 1위 종목 종가매수 50만원")
    print(f"  검증: 25일 내 5%+ 상승 여부")
    print("=" * 80)

    trades = []
    no_signal = 0
    errors = 0

    for idx, d in enumerate(days):
        print(f"\n[{idx+1}/{len(days)}] {d} 스캔...")

        try:
            r = run_scan(d)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            continue

        if not r:
            no_signal += 1
            continue

        pick = r[0]
        code = pick["code"]
        name = pick["name"]
        buy_price = pick["buyPrice"]
        t1 = pick["target1"]
        t2 = pick["target2"]
        sl = pick["stoploss"]
        atr = pick.get("atr", 0)
        score = pick.get("rankScore", pick.get("momentum", 0))

        result = check_stock_performance(code, name, d, buy_price, t1, t2, sl, atr)
        if result is None:
            no_signal += 1
            continue

        result["rankScore"] = score
        result["supply"] = pick.get("supplyJudge", "-")
        trades.append(result)

        tag_5 = "O" if result["hit_5pct"] else "X"
        tag_10 = "O" if result["hit_10pct"] else "X"
        s = "+" if result["profit_pct"] >= 0 else ""
        print(f"  ★ {name}({code}) 점수:{score} | 5%:{tag_5} 10%:{tag_10} "
              f"최고:{result['max_pct']:+.1f}% 최저:{result['max_drawdown']:.1f}% "
              f"| 실현:{s}{result['profit_pct']:.1f}% ({result['exit_reason']})")

        if (idx + 1) % 15 == 0:
            save_cache_to_disk()

    save_cache_to_disk()

    # ═══════════════════════════════════════════
    #  최종 결과
    # ═══════════════════════════════════════════
    print("\n\n")
    print("=" * 80)
    print(f"  {'█'*20} 검증 결과 {'█'*20}")
    print("=" * 80)

    if not trades:
        print("  매매 내역 없음")
        return

    n = len(trades)
    cnt_5 = sum(1 for t in trades if t["hit_5pct"])
    cnt_10 = sum(1 for t in trades if t["hit_10pct"])
    wins = [t for t in trades if t["profit"] > 0]
    losses = [t for t in trades if t["profit"] <= 0]

    total_cost = sum(t["cost"] for t in trades)
    total_profit = sum(t["profit"] for t in trades)
    total_pct = (total_profit / total_cost) * 100 if total_cost > 0 else 0

    avg_max = np.mean([t["max_pct"] for t in trades])
    avg_dd = np.mean([t["max_drawdown"] for t in trades])
    avg_win_pct = np.mean([t["profit_pct"] for t in wins]) if wins else 0
    avg_loss_pct = np.mean([t["profit_pct"] for t in losses]) if losses else 0

    gross_win = sum(t["profit"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["profit"] for t in losses)) if losses else 1
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    print(f"\n  검증 기간: {days[0]} ~ {days[-1]}")
    print(f"  거래일: {len(days)}일 | 신호발생: {n}회 | 무신호: {no_signal}일 | 에러: {errors}")

    print(f"\n  {'━'*50}")
    print(f"  ┃  5%+ 상승 성공률:  {cnt_5}/{n} = {cnt_5/n*100:.1f}%")
    print(f"  ┃ 10%+ 상승 성공률:  {cnt_10}/{n} = {cnt_10/n*100:.1f}%")
    print(f"  {'━'*50}")
    print(f"  ┃  실매매 승률:      {len(wins)}/{n} = {len(wins)/n*100:.1f}%")
    print(f"  ┃  실매매 총수익률:  {'+' if total_pct>=0 else ''}{total_pct:.2f}%")
    print(f"  ┃  총 손익:          {'+' if total_profit>=0 else ''}{total_profit:,.0f}원")
    print(f"  ┃  손익비(PF):       {pf:.2f}")
    print(f"  {'━'*50}")
    print(f"  ┃  평균 최고상승:    +{avg_max:.1f}%")
    print(f"  ┃  평균 최대낙폭:    {avg_dd:.1f}%")
    print(f"  ┃  평균 수익(승):    +{avg_win_pct:.1f}%")
    print(f"  ┃  평균 손실(패):    {avg_loss_pct:.1f}%")
    print(f"  {'━'*50}")

    # T1/T2 도달률
    t1_cnt = sum(1 for t in trades if t["t1_hit"])
    t2_cnt = sum(1 for t in trades if t["t2_hit"])
    sl_cnt = sum(1 for t in trades if t["exit_reason"] == "손절")
    print(f"  ┃  T1 도달률:        {t1_cnt}/{n} = {t1_cnt/n*100:.1f}%")
    print(f"  ┃  T2 도달률:        {t2_cnt}/{n} = {t2_cnt/n*100:.1f}%")
    print(f"  ┃  손절 비율:        {sl_cnt}/{n} = {sl_cnt/n*100:.1f}%")
    print(f"  {'━'*50}")

    # 수급별 성공률
    supply_groups = {}
    for t in trades:
        s = t.get("supply", "-")
        if s not in supply_groups:
            supply_groups[s] = {"total": 0, "hit5": 0, "profit": 0}
        supply_groups[s]["total"] += 1
        if t["hit_5pct"]: supply_groups[s]["hit5"] += 1
        supply_groups[s]["profit"] += t["profit"]

    print(f"\n  수급별 5%+ 성공률:")
    for s, g in sorted(supply_groups.items(), key=lambda x: -x[1]["hit5"]):
        rate = g["hit5"]/g["total"]*100 if g["total"] > 0 else 0
        print(f"    {s:<10}: {g['hit5']}/{g['total']} = {rate:.0f}% | 손익:{g['profit']:+,.0f}원")

    # 월별
    print(f"\n  월별 성과:")
    monthly = {}
    for t in trades:
        m = t["buy_date"][:7]
        if m not in monthly:
            monthly[m] = {"n": 0, "hit5": 0, "profit": 0}
        monthly[m]["n"] += 1
        if t["hit_5pct"]: monthly[m]["hit5"] += 1
        monthly[m]["profit"] += t["profit"]
    for m in sorted(monthly):
        g = monthly[m]
        rate = g["hit5"]/g["n"]*100
        print(f"    {m}: {g['n']:>3}회 | 5%+:{g['hit5']}/{g['n']}({rate:.0f}%) | {g['profit']:+,.0f}원")

    # 개별 내역
    print(f"\n  {'─'*75}")
    print(f"  {'No':>3} {'매수일':>10} {'종목':<10} {'매수가':>7} {'최고':>6} {'5%':>3} {'실현':>7} {'결과':<6} {'수급'}")
    print(f"  {'─'*75}")
    for j, t in enumerate(trades, 1):
        tag5 = "O" if t["hit_5pct"] else "X"
        s = "+" if t["profit_pct"] >= 0 else ""
        print(f"  {j:>3} {t['buy_date']:>10} {t['name']:<10} {t['buy_price']:>7,} "
              f"{t['max_pct']:>+5.1f}% {tag5:>3} {s}{t['profit_pct']:>5.1f}% {t['exit_reason']:<6} {t.get('supply','-')}")

    # 누적
    print(f"\n  누적 손익:")
    cum = 0
    for t in trades:
        cum += t["profit"]
        bar = "█" * min(int(abs(cum)/3000), 40)
        s = "+" if cum >= 0 else ""
        print(f"    {t['buy_date'][:5]} {t['name'][:6]:<6} |{bar} {s}{cum:,.0f}")

    print(f"\n  최종 누적: {'+' if cum>=0 else ''}{cum:,.0f}원")
    print("=" * 80)

    # JSON 저장
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open("backtest_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "period": f"{days[0]}~{days[-1]}",
            "total_trades": n,
            "hit_5pct_rate": round(cnt_5/n*100, 1),
            "hit_10pct_rate": round(cnt_10/n*100, 1),
            "win_rate": round(len(wins)/n*100, 1),
            "total_profit": total_profit,
            "total_pct": round(total_pct, 2),
            "profit_factor": round(pf, 2),
            "trades": trades,
        }, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"  저장: backtest_result.json")


if __name__ == "__main__":
    main()
