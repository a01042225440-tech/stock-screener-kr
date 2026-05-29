"""스윙 장중 모니터 (GitHub Actions, 세션 중 ~20분마다 실행).
- 매 실행: 보유 포지션 적응형 매도 점검(장중 분봉 종가근사) → 걸리면 '🔴 즉시 매도' 알림
- 15:05~15:22 KST 창: 저점매수 신호 스캔 → '🟢 매수신호' 알림 + 포지션 기록
- 이벤트(매도/매수신호) 있을 때만 텔레그램 발송(스팸 방지)
"""
import os, sys
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from swing_tracker import run_swing, format_swing_telegram
from app import send_telegram

def main():
    kst = datetime.now(timezone.utc) + timedelta(hours=9)
    hm = kst.hour * 60 + kst.minute
    in_buy = (14 * 60 + 35) <= hm <= (14 * 60 + 58)   # 14:40 틱(3시 전) 매수 스캔 창
    pl = run_swing(intraday=True, do_buys=in_buy)
    sells = pl.get("sells", []); buys = pl.get("buys", [])
    sent = False

    if sells:   # 장중 즉시 매도 알림
        L = [f"🔴 <b>스윙 매도 신호!</b> ({kst.strftime('%m/%d %H:%M')} 장중)", ""]
        for s in sells:
            L.append(f"  • <b>{s['name']}</b>({s['code']}) {s['close']:,}원 "
                     f"<b>{s['profit']:+.1f}%</b> — {s['reason']}")
        L.append("")
        L.append("⚡ 지금(장중) 매도 권장")
        ok, info = send_telegram("\n".join(L)); sent = True
        print(f"[SWING-SELL] {len(sells)} alerts send={ok} {info}")

    if in_buy and buys:   # 매수 신호 알림(15:13 창)
        payload = {"date": kst.strftime("%Y-%m-%d"), "sells": [],
                   "buys": buys, "holdings": pl.get("holdings", [])}
        ok, info = send_telegram(format_swing_telegram(payload)); sent = True
        print(f"[SWING-BUY] {len(buys)} signals send={ok} {info}")

    if not sent:
        print(f"[SWING] no event @ KST {kst.strftime('%H:%M')} (보유 {pl.get('count_open', 0)})")

if __name__ == "__main__":
    main()
