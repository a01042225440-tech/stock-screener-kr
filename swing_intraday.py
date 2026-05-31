"""장중 모니터 (GitHub Actions, ~20분마다 실행) — 모멘텀 + 스윙 통합.
- 매 실행: 모멘텀·스윙 보유 포지션 매도/손절 점검(장중 분봉) → 걸리면 '🔴 즉시 매도/손절' 알림
- 14:40 KST 창: 스윙 저점매수 신호 스캔 → '🟢 매수신호' 알림 + 포지션 기록
- (모멘텀 매수는 15:13 daily_notify에서 발송·기록)
- 이벤트 있을 때만 텔레그램 발송(스팸 방지)
"""
import os, sys
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from swing_tracker import run_swing, format_swing_telegram
from momentum_tracker import check_sells as mom_check_sells
from app import send_telegram

def main():
    kst = datetime.now(timezone.utc) + timedelta(hours=9)
    date_str = kst.strftime("%Y-%m-%d")
    hm = kst.hour * 60 + kst.minute
    in_buy = (14 * 60 + 35) <= hm <= (14 * 60 + 58)   # 14:40 틱(3시 전) 스윙 매수 스캔 창
    sent = False

    # ── 1) 모멘텀 보유 매도/손절 점검 (실시간) ──
    try:
        mom_sells = mom_check_sells(date_str, intraday=True)
    except Exception as e:
        print(f"[MOM] check error: {e}"); mom_sells = []
    if mom_sells:
        L = [f"⚡ <b>모멘텀 청산 신호!</b> ({kst.strftime('%m/%d %H:%M')} 장중)", ""]
        for s in mom_sells:
            L.append(f"  • <b>{s['name']}</b>({s['code']}) {s['close']:,}원 "
                     f"<b>{s['profit']:+.1f}%</b> — {s['reason']}")
        ok, info = send_telegram("\n".join(L)); sent = True
        print(f"[MOM-SELL] {len(mom_sells)} alerts send={ok} {info}")

    # ── 2) 스윙 보유 매도 점검 + 14:40 매수 스캔 ──
    pl = run_swing(intraday=True, do_buys=in_buy, date_str=date_str)
    sw_sells = pl.get("sells", []); buys = pl.get("buys", [])
    if sw_sells:
        L = [f"🔴 <b>스윙 매도 신호!</b> ({kst.strftime('%m/%d %H:%M')} 장중)", ""]
        for s in sw_sells:
            L.append(f"  • <b>{s['name']}</b>({s['code']}) {s['close']:,}원 "
                     f"<b>{s['profit']:+.1f}%</b> — {s['reason']}")
        L.append("\n⚡ 지금(장중) 매도 권장")
        ok, info = send_telegram("\n".join(L)); sent = True
        print(f"[SWING-SELL] {len(sw_sells)} alerts send={ok} {info}")

    if in_buy and buys:
        payload = {"date": date_str, "sells": [], "buys": buys, "holdings": pl.get("holdings", [])}
        ok, info = send_telegram(format_swing_telegram(payload)); sent = True
        print(f"[SWING-BUY] {len(buys)} signals send={ok} {info}")

    if not sent:
        print(f"[MONITOR] no event @ KST {kst.strftime('%H:%M')}")

if __name__ == "__main__":
    main()
