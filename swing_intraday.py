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


RENDER_URL = os.environ.get("RENDER_URL", "https://stock-screener-kr.onrender.com")

def keep_render_warm(date_str, hm):
    """장중(08:50~16:00 KST)엔 Render 대시보드를 미리 깨우고 캐시를 데워둠.
    → 사용자가 15:15에 열면 콜드스타트 없이 즉시 결과 표시. (외부 핑서비스 불필요)"""
    if not (8 * 60 + 50 <= hm <= 16 * 60):
        return
    try:
        import requests
        requests.get(f"{RENDER_URL}/api/scan", params={"date": date_str, "market": "ALL"}, timeout=20)
    except Exception:
        pass

def one_pass(buy_done, report_done):
    """반환 (이벤트?, did_buy, did_report)."""
    kst = now_kst()
    date_str = kst.strftime("%Y-%m-%d")
    hm = kst.hour * 60 + kst.minute
    keep_render_warm(date_str, hm)   # 0) Render 깨우기+캐시 예열 (대시보드 즉시표시)
    # 1) 매도/손절 점검 (매 틱)
    e1 = do_momentum_sells(date_str, kst)
    e2, _ = do_swing(date_str, kst, False)   # 스윙 매도만(매수는 통합알림이 처리)
    # 2) 매수 알림: 15:18 이후 하루 1회 (늦게 시작해도 즉시 발송 보장)
    bought = False
    if (not buy_done) and hm >= (15 * 60 + 18) and hm < (16 * 60 + 30):
        try:
            from notify_send import run_buy_alert
            run_buy_alert(); bought = True
        except Exception as e:
            print(f"[BUY-ALERT] error: {e}")
    # 3) 일일 보고: 16:38 이후 하루 1회
    reported = False
    if (not report_done) and hm >= (16 * 60 + 38):
        try:
            from report_daily import main as report_main
            report_main(); reported = True
        except Exception as e:
            print(f"[REPORT] error: {e}")
    return (e1 or e2 or bought or reported), bought, reported


def main():
    # 장시작~16:50 KST까지 종일 상주(LONG_RUN=1) — 하루1회 크론으로 시작, 내부 루프로 정시 보장.
    # GitHub */10 스케줄이 throttle로 거의 안 떠서, 하루1회 시작 + 종일 상주로 전환.
    long_run = os.environ.get("LONG_RUN", "1") == "1"
    sleep_s = int(os.environ.get("MONITOR_SLEEP", "150"))
    end_hm = 16 * 60 + 50          # 16:50 KST에 종료
    buy_done = report_done = False
    if not long_run:
        # (수동 단발 테스트용)
        ev, db, dr = one_pass(buy_done, report_done)
        print(f"[MONITOR] single pass @ {now_kst().strftime('%H:%M')} buy={db} report={dr}")
        return
    print(f"[MONITOR] 상주 시작 @ KST {now_kst().strftime('%H:%M')} (16:50까지)")
    while True:
        kst = now_kst(); hm = kst.hour * 60 + kst.minute
        if hm >= end_hm and (buy_done or hm >= 16*60+30):
            break
        try:
            ev, db, dr = one_pass(buy_done, report_done)
            if db: buy_done = True; print(f"[MONITOR] 매수알림 발송 @ {kst.strftime('%H:%M')}")
            if dr: report_done = True; print(f"[MONITOR] 일일보고 발송 @ {kst.strftime('%H:%M')}")
        except Exception as e:
            print(f"[MONITOR] pass error: {e}")
        if report_done and buy_done and hm >= 16*60+45:
            break
        time.sleep(sleep_s)
    print(f"[MONITOR] 종료 @ KST {now_kst().strftime('%H:%M')} (buy={buy_done} report={report_done})")


if __name__ == "__main__":
    main()
