"""
매일 15:20 통합 매수 알림 (GitHub Actions).
순서: 🌊 스윙(1순위) → 💰 모멘텀 3종목(2순위). 스윙 없으면 '없음' 표시.
스윙·모멘텀 포지션 모두 기록(장중 매도/손절 추적용).
"""
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from app import run_scan, send_telegram, get_kospi_regime, pick_with_sector_limit
from swing_tracker import run_swing
from momentum_tracker import record_buys

_GO = {"BREAKOUT": 0, "HUNT": 1, "TREND": 2, "WATCH": 3}

def build_message(date, dow, momentum, swing_buys, regime):
    bear = regime.get("ok") and not regime.get("above_ma60", True)
    # 통합 매수 후보(돈 순서): 🌊스윙 우선 → 💰모멘텀(BREAKOUT>HUNT>TREND·점수순)
    buy_all = []
    for b in (swing_buys or []):
        buy_all.append({"strat": "swing", "name": b["name"], "code": b["code"],
                        "buy": b["close"], "target": b.get("upper"), "stop": b.get("lower"),
                        "chg": b.get("chgToday"), "industryCode": str(b.get("industryCode", "0"))})
    for s in sorted(momentum or [], key=lambda x: (_GO.get(x.get("grade"), 9), -(x.get("finalScore") or 0))):
        buy_all.append({"strat": "mom", "name": s.get("name"), "code": s.get("code"),
                        "buy": s.get("buyPrice"), "target": s.get("target1"), "stop": s.get("stoploss"),
                        "grade": s.get("grade"), "targetPct": s.get("targetPct", 10),
                        "chg": s.get("chgToday"), "industryCode": str(s.get("industryCode", "0"))})
    # 🚦 약세장 방어전환: 추세추종 OFF, 스윙만 (검증: 약세장 최악기간 +4%→+13%, 평균유지)
    if bear:
        buy_all = [c for c in buy_all if c.get("strat") == "swing"]
    # 파랑(어제대비 하락) 제외(검증: 손해없음) → 한 업종 최대 2개 → 상위 3
    buy_all = [c for c in buy_all if not (c.get("chg") is not None and c["chg"] < 0)]
    top3 = pick_with_sector_limit(buy_all, n=3, max_per_sector=2)

    L = [f"📊 <b>오늘 살 {'스윙 종목' if bear else '3종목'} ({date} {dow}) · 자본 33%씩</b>"]
    if bear:
        L.append("🔴 <b>약세장(코스피&lt;60일선) — 방어모드: 🌊스윙만, 비중축소</b>")
    L.append("⏱ 15:20~15:30 동시호가 · 🌊스윙 우선 · 한 업종 최대 2개")
    L.append("")
    if not top3:
        L.append("오늘 매수 신호 <b>없음</b> — 쉬는 것도 전략 (현금 보유)")
    else:
        for i, s in enumerate(top3, 1):
            if s["strat"] == "swing":
                tag = "🌊스윙"; tgt = f"🎯목표 {s['target']:,}(BB상한)"
            else:
                tag = f"💰{s.get('grade','')}"; tgt = f"🎯+{s.get('targetPct',10)}% {s['target']:,}"
            L.append(f"<b>{i}. {tag} {s['name']}</b>({s['code']})")
            L.append(f"   💰매수 {s['buy']:,} · {tgt} · 🛑손절 {s['stop']:,}")
    L.append("")
    L.append("📌 위 3종목 33%씩 · 🎯목표 익절 · 🛑손절 반드시")
    return "\n".join(L)


def run_buy_alert():
    """15:20 통합 매수 알림 (모멘텀+스윙). 발송 성공여부 반환. (intraday 모니터/단독 둘 다 호출)"""
    date = datetime.now().strftime("%Y-%m-%d")
    dow = ["월","화","수","목","금","토","일"][datetime.strptime(date, "%Y-%m-%d").weekday()]
    print(f"[NOTIFY] 통합 스캔 시작: {date}")

    # 모멘텀
    results = run_scan(date)
    actual = date
    for r in (results or []):
        if r.get("dataDate"):
            actual = r["dataDate"]; break
    regime = get_kospi_regime(actual)
    bear = regime.get("ok") and not regime.get("above_ma60", True)

    # 스윙 (장중 분봉 종가근사, 포지션 기록 포함)
    try:
        pl = run_swing(date_str=date, intraday=True, do_buys=True, auto_open=not bear)
        swing_buys = pl.get("buys", [])
    except Exception as e:
        print(f"[NOTIFY] 스윙 스캔 오류: {e}"); swing_buys = []

    msg = build_message(date, dow, results, swing_buys, regime)
    ok, info = send_telegram(msg)
    print(f"[NOTIFY] 발송: ok={ok} {info} | 스윙{len(swing_buys)} 모멘텀{len(results)}")

    # 모멘텀 포지션 기록 (약세장이면 보류)
    try:
        if not bear:
            added = record_buys(results, actual, max_n=3)
            print(f"[NOTIFY] 모멘텀 포지션 기록: {added}")
        else:
            print("[NOTIFY] 약세장 → 신규 포지션 미기록")
    except Exception as e:
        print(f"[NOTIFY] 포지션 기록 오류: {e}")

    return ok


def main():
    ok = run_buy_alert()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
