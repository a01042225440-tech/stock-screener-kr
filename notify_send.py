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

from app import run_scan, format_telegram_message, send_telegram

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
    if actual != date:
        status = "intraday" if date >= datetime.now().strftime("%Y-%m-%d") else "holiday"
    else:
        status = "confirmed"

    payload = {
        "date": date, "actualDataDate": actual, "dataStatus": status,
        "dayOfWeek": dow_names[dow], "results": results, "count": len(results)
    }
    msg = format_telegram_message(payload)
    ok, info = send_telegram(msg)
    print(f"[NOTIFY] 발송 결과: ok={ok} info={info} count={len(results)}")
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
