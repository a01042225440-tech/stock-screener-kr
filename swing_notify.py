"""스윙 추적 매일 발송 (GitHub Actions). 매수신호 스캔 + 보유 매도점검 → 텔레그램."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from swing_tracker import run_swing, format_swing_telegram
from app import send_telegram

def main():
    pl = run_swing()
    msg = format_swing_telegram(pl)
    ok, info = send_telegram(msg)
    print(f"[SWING] sells={len(pl['sells'])} buys={len(pl['buys'])} open={pl['count_open']} send={ok} {info}")
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
