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
from app import run_scan, send_telegram, get_kospi_regime
from swing_tracker import run_swing
from momentum_tracker import record_buys


def build_message(date, dow, momentum, swing_buys, regime):
    bear = regime.get("ok") and not regime.get("above_ma60", True)
    L = [f"📊 <b>오늘의 매수 ({date} {dow})</b>"]
    if bear:
        L.append(f"🔴 <b>약세장(코스피&lt;60일선) — 신규매수 보류 권고</b>")
    L.append("⏱ 15:20~15:30 동시호가 매수 · 상위 3종목 균등")
    L.append("")

    # 🌊 1순위 스윙
    L.append("🌊 <b>1순위 — 스윙 (저점반등, 수익률 최고)</b>")
    if swing_buys:
        for b in swing_buys[:3]:
            L.append(f"  • <b>{b['name']}</b>({b['code']}) {b['close']:,}원")
            L.append(f"     🎯목표 BB상한 {b.get('upper','-'):,} · 🛑손절 BB하한 {b.get('lower','-'):,}")
    else:
        L.append("  └ 오늘 스윙 신호 <b>없음</b>")
    L.append("")

    # 💰 2순위 모멘텀
    L.append("💰 <b>2순위 — 모멘텀 3종목 (BREAKOUT 우선)</b>")
    if momentum:
        emoji = {"HUNT": "🟢", "BREAKOUT": "🔴", "TREND": "🟡"}
        for i, s in enumerate(momentum[:3], 1):
            g = s.get("grade", "")
            tp = s.get("targetPct", 10)
            L.append(f"  <b>{i}. {emoji.get(g,'⚪')}{g} {s.get('name')}</b>({s.get('code')})")
            L.append(f"     💰매수 {s.get('buyPrice'):,} · 🎯+{tp}% {s.get('target1'):,} · 🛑손절 {s.get('stoploss'):,}")
    else:
        L.append("  └ 오늘 모멘텀 신호 없음")
    L.append("")
    L.append("📌 <b>스윙 먼저 채우고, 남는 슬롯을 모멘텀으로</b> (합쳐 3종목)")
    L.append("🛑 손절 반드시 지키기")
    return "\n".join(L)


def main():
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

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
