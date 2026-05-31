"""
매일 자동 발송 스크립트 (GitHub Actions에서 실행)
스캔 → 매수 후보 3종목 → 텔레그램 발송
토큰은 환경변수(GitHub Secrets)에서 읽음
"""
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except: pass

from app import run_scan, format_telegram_message, send_telegram, get_kospi_regime

def main():
    date = datetime.now().strftime("%Y-%m-%d")
    print(f"[NOTIFY] 스캔 시작: {date}")
    results = run_scan(date)

    dow_names = ["월","화","수","목","금","토","일"]
    dow = datetime.strptime(date, "%Y-%m-%d").weekday()
    actual = date
    if results:
        for r in results:
            if r.get("dataDate"):
                actual = r["dataDate"]; break
    if results and results[0].get("isIntradayProxy"):
        status = "intraday_proxy"   # 장중 분봉 종가근사 신호
    elif actual != date:
        status = "intraday" if date >= datetime.now().strftime("%Y-%m-%d") else "holiday"
    else:
        status = "confirmed"

    regime = get_kospi_regime(actual)
    payload = {
        "date": date, "actualDataDate": actual, "dataStatus": status,
        "dayOfWeek": dow_names[dow], "results": results, "count": len(results),
        "marketRegime": regime,
    }
    msg = format_telegram_message(payload)
    ok, info = send_telegram(msg)
    print(f"[NOTIFY] 발송 결과: ok={ok} info={info} count={len(results)}")

    # 모멘텀 매수 포지션 기록 (장중 매도/손절 실시간 추적용) — 약세장이면 기록 안 함
    try:
        from momentum_tracker import record_buys
        bear = regime.get("ok") and not regime.get("above_ma60", True)
        if not bear:
            added = record_buys(results, actual, max_n=3)
            print(f"[NOTIFY] 모멘텀 포지션 기록: {added}")
        else:
            print("[NOTIFY] 약세장 → 모멘텀 매수 보류(포지션 미기록)")
    except Exception as e:
        print(f"[NOTIFY] 포지션 기록 오류: {e}")

    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
