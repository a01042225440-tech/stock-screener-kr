"""
모멘텀 포지션 추적기 — 매수 기록 + 장중 매도/손절 실시간 판정.
매수: 메인 스캔(HUNT/BREAKOUT)의 상위 3종목.
청산: 목표가 도달(+10%, BREAKOUT +15%) 익절 / -5% 손절 / 6영업일 만기.
포지션은 momentum_positions.json에 저장(깃 커밋으로 영속).
"""
import json, os
from datetime import datetime
import pandas as pd
from swing_tracker import _fetch   # 장중 분봉 프록시 포함 OHLCV
from app import tick

POS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "momentum_positions.json")

def load_positions():
    try:
        with open(POS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("positions", [])
    except Exception:
        return []

def save_positions(positions):
    with open(POS_FILE, "w", encoding="utf-8") as f:
        json.dump({"positions": positions}, f, ensure_ascii=False, indent=2)

def record_buys(results, date_str, max_n=3):
    """스캔 결과 상위 max_n 종목을 모멘텀 포지션으로 추가(중복 코드 제외)."""
    positions = load_positions()
    held = {p["code"] for p in positions}
    added = []
    for r in (results or [])[:max_n]:
        code = r.get("code")
        if not code or code in held:
            continue
        buy = r.get("buyPrice") or r.get("close")
        if not buy:
            continue
        tp = r.get("targetPct", 15 if r.get("grade") == "BREAKOUT" else 10)
        positions.append({
            "code": code, "name": r.get("name", code), "grade": r.get("grade", ""),
            "buyDate": date_str, "buyPrice": int(buy),
            "target": tick(int(buy * (1 + tp / 100)), buy),
            "stop": r.get("stoploss") or tick(int(buy * 0.90), buy),   # 10일저점, fallback -10%
            "targetPct": tp, "maxHold": 6,
        })
        held.add(code); added.append(code)
    save_positions(positions)
    return added

def sell_check(df, p):
    """모멘텀 포지션 청산 판정 (장중 고/저 기준). 반환 (sell?, reason, info).
    ※ 매수 당일 봉은 제외 — 종가 매수이므로 그날의 고/저는 매수 전에 발생."""
    if df is None or len(df) < 2:
        return False, "", {}
    after = df[df.index > pd.Timestamp(p["buyDate"])]   # 매수일 이후 봉만
    if len(after) == 0:
        return False, "", {"close": p["buyPrice"], "profit": 0.0}   # 아직 매수 당일
    last = after.iloc[-1]
    hi = int(last["High"]); lo = int(last["Low"]); px = int(last["Close"])
    buy = p["buyPrice"]; target = p["target"]; stop = p["stop"]
    profit = (px - buy) / buy * 100
    held_days = len(after)
    sell, reason = False, ""
    if hi >= target:
        sell, reason, px = True, f"🎯 익절(+{p.get('targetPct',10)}% 도달)", target
        profit = (target - buy) / buy * 100
    elif lo <= stop:
        sell, reason, px = True, f"🛑 손절(10일저점 이탈, {(stop-buy)/buy*100:.0f}%)", stop
        profit = (stop - buy) / buy * 100
    elif held_days >= p.get("maxHold", 6):
        sell, reason = True, f"⏱ 6일만기({profit:+.1f}%)"
    info = {"close": px, "profit": round(profit, 1)}
    return sell, reason, info

def check_sells(date_str, intraday=True):
    """보유 모멘텀 포지션 전수 매도 점검. 청산 종목 alert리스트 반환 + 포지션 갱신."""
    positions = load_positions()
    sells, keep = [], []
    for p in positions:
        df = _fetch(p["code"], date_str, intraday=intraday)
        s, reason, info = sell_check(df, p)
        if s:
            sells.append({**p, "reason": reason, **info})
            try:
                from trade_log import log_closed
                log_closed("모멘텀", p["code"], p.get("name", p["code"]), p["buyDate"],
                           p["buyPrice"], info.get("close", p["buyPrice"]), info.get("profit", 0), reason)
            except Exception as e:
                print(f"[MOM] log error: {e}")
        else:
            p["last"] = info.get("close", p["buyPrice"])
            p["profit"] = info.get("profit", 0)
            keep.append(p)
    save_positions(keep)
    return sells
