"""장중 모니터 (GitHub Actions) — 모멘텀 + 스윙 통합, ~1분 해상도.
- 크론 */5(5분, GitHub 최소) + 잡 내부 미니루프(MONITOR_ITERS회 × MONITOR_SLEEP초)
  → 실질 1분 간격으로 매도/손절 점검(스팸 방지: 팔린 포지션은 파일에서 제거돼 재알림 없음)
- 매수 스캔: 스윙 15:20~15:24 단일 틱에서 1회만(중복 방지). 모멘텀 매수도 15:20 daily_notify.
- 이벤트(매도/손절/매수신호) 있을 때만 텔레그램 발송.
"""
import os, sys, time
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from swing_tracker import run_swing, format_swing_telegram
from momentum_tracker import check_sells as mom_check_sells
from app import send_telegram


def now_kst():
    return datetime.now(timezone.utc) + timedelta(hours=9)


def do_momentum_sells(date_str, kst):
    """모멘텀 보유 매도/손절 점검 → 알림. 이벤트 발생 여부 반환."""
    try:
        sells = mom_check_sells(date_str, intraday=True)
    except Exception as e:
        print(f"[MOM] check error: {e}")
        return False
    if not sells:
        return False
    L = [f"⚡ <b>모멘텀 청산 신호!</b> ({kst.strftime('%m/%d %H:%M')} 장중)", ""]
    for s in sells:
        L.append(f"  • <b>{s['name']}</b>({s['code']}) {s['close']:,}원 "
                 f"<b>{s['profit']:+.1f}%</b> — {s['reason']}")
    ok, info = send_telegram("\n".join(L))
    print(f"[MOM-SELL] {len(sells)} alerts send={ok} {info}")
    return True


def do_swing(date_str, kst, do_buys):
    """스윙 매도 점검(+ do_buys면 매수 스캔) → 알림. (이벤트여부, 매수실행여부) 반환."""
    try:
        pl = run_swing(intraday=True, do_buys=do_buys, date_str=date_str)
    except Exception as e:
        print(f"[SWING] run error: {e}")
        return False, False
    event = False
    sells = pl.get("sells", [])
    if sells:
        L = [f"🔴 <b>스윙 매도 신호!</b> ({kst.strftime('%m/%d %H:%M')} 장중)", ""]
        for s in sells:
            L.append(f"  • <b>{s['name']}</b>({s['code']}) {s['close']:,}원 "
                     f"<b>{s['profit']:+.1f}%</b> — {s['reason']}")
        L.append("\n⚡ 지금(장중) 매도 권장")
        ok, info = send_telegram("\n".join(L))
        print(f"[SWING-SELL] {len(sells)} alerts send={ok} {info}")
        event = True
    buys = pl.get("buys", [])
    if do_buys and buys:
        payload = {"date": date_str, "sells": [], "buys": buys, "holdings": pl.get("holdings", [])}
        ok, info = send_telegram(format_swing_telegram(payload))
        print(f"[SWING-BUY] {len(buys)} signals send={ok} {info}")
        event = True
    return event, do_buys


def one_pass(buy_done, report_done):
    """반환 (이벤트?, did_buy, did_report)."""
    kst = now_kst()
    date_str = kst.strftime("%Y-%m-%d")
    hm = kst.hour * 60 + kst.minute
    # 1) 매도/손절 점검 (매 틱)
    e1 = do_momentum_sells(date_str, kst)
    e2, _ = do_swing(date_str, kst, False)   # 스윙 매도만(매수는 통합알림이 처리)
    # 2) 15:20 단일 틱: 통합 매수 알림 (모멘텀+스윙)
    bought = False
    if (not buy_done) and (15 * 60 + 20) <= hm <= (15 * 60 + 24):
        try:
            from notify_send import run_buy_alert
            run_buy_alert(); bought = True
        except Exception as e:
            print(f"[BUY-ALERT] error: {e}")
    # 3) 16:40 단일 틱: 일일 엑셀 보고 + 자동점검
    reported = False
    if (not report_done) and (16 * 60 + 40) <= hm <= (16 * 60 + 44):
        try:
            from report_daily import main as report_main
            report_main(); reported = True
        except Exception as e:
            print(f"[REPORT] error: {e}")
    return (e1 or e2 or bought or reported), bought, reported


def main():
    iters = int(os.environ.get("MONITOR_ITERS", "1"))
    sleep_s = int(os.environ.get("MONITOR_SLEEP", "60"))
    buy_done = report_done = False
    any_event = False
    for k in range(max(1, iters)):
        ev, did_buy, did_report = one_pass(buy_done, report_done)
        any_event = any_event or ev
        if did_buy: buy_done = True
        if did_report: report_done = True
        if k < iters - 1:
            time.sleep(sleep_s)
    if not any_event:
        print(f"[MONITOR] no event @ KST {now_kst().strftime('%H:%M')}")


if __name__ == "__main__":
    main()
