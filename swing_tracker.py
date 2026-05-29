"""
저점매수·고점매도 스윙 추적기
- 매일 저점매수 신호 종목 스캔 (RSI<=35 또는 %B<=15 + 양봉반등 + 장기우상향)
- 신호 종목을 swing_positions.json에 포지션으로 기록
- 매일 보유 포지션의 적응형 매도조건 체크 → 매도 시점 알림
조건식(검증):
  매수: (RSI(14)<=35 OR 볼린저%B<=15) AND 종가>전일종가 AND 종가>시가 AND 종가>200MA
  매도(적응형):
    [추세] 20MA상승 & 종가>60MA: 고점대비-8%(트레일) OR RSI>=78
    [박스] 그외: (RSI>=70 OR %B>=95) AND 종가<전일종가
    손절: 매수가 -8%
"""
import json, os
from datetime import datetime
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
from app import (naver_ohlcv_fast, naver_all_rising_parallel,
                 parse_num, is_excluded_by_name, naver_today_ohlc)

POS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swing_positions.json")

def _fetch(code, date_str, intraday=False):
    """OHLCV 조회. intraday=True면 확정봉에 오늘 없을 때 분봉 종가근사 봉을 부착."""
    df = naver_ohlcv_fast(code, 300, target_date=date_str)
    if df is None:
        return None
    df = df[df.index <= pd.Timestamp(date_str)]
    if intraday and len(df) > 0 and df.index[-1] < pd.Timestamp(date_str):
        ohlc = naver_today_ohlc(code, date_str)
        if ohlc:
            o, h, l, c = ohlc
            row = pd.DataFrame([{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 0}],
                               index=[pd.Timestamp(date_str)])
            df = pd.concat([df, row])
    return df

def _indicators(df):
    s = df["Close"]
    d = s.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100/(1 + up/dn)
    m20 = s.rolling(20).mean(); sd = s.rolling(20).std()
    lower = m20 - 2*sd; upper = m20 + 2*sd
    pctB = (s - lower)/(upper - lower)*100
    m60 = s.rolling(60).mean(); m200 = s.rolling(200).mean()
    return rsi, pctB, m20, m60, m200

def buy_signal(df):
    """최신 봉 기준 저점매수 신호 여부 + 지표 dict"""
    if df is None or len(df) < 200:
        return False, {}
    s = df["Close"]; o = df["Open"]
    rsi, pctB, m20, m60, m200 = _indicators(df)
    i = -1
    oversold = (rsi.iloc[i] <= 35) or (pctB.iloc[i] <= 15)
    reversal = (s.iloc[i] > s.iloc[i-1]) and (s.iloc[i] > o.iloc[i])
    healthy = (not pd.isna(m200.iloc[i])) and (s.iloc[i] > m200.iloc[i])   # 장기 우상향만(칼 회피)
    ok = bool(oversold and reversal and healthy)
    info = {"rsi": round(float(rsi.iloc[i]), 1), "pctB": round(float(pctB.iloc[i])),
            "close": int(s.iloc[i])}
    return ok, info

def sell_check(df, buy_price, buy_date):
    """보유 포지션의 적응형 매도 판정. 반환 (sell?, reason, info)"""
    if df is None or len(df) < 60:
        return False, "", {}
    s = df["Close"]
    rsi, pctB, m20, m60, m200 = _indicators(df)
    i = -1; px = float(s.iloc[i])
    # 매수일 이후 고점(High)
    held = df[df.index >= pd.Timestamp(buy_date)]
    peak = float(held["High"].max()) if len(held) else px
    strong = (not pd.isna(m20.iloc[i]) and m20.iloc[i] > m20.iloc[i-5]) and (px > float(m60.iloc[i]))
    overbought = (rsi.iloc[i] >= 70) or (pctB.iloc[i] >= 95)
    profit = (px - buy_price)/buy_price*100
    held_days = len(held)
    sell, reason = False, ""
    if px <= buy_price * 0.92:
        sell, reason = True, "손절(-8%)"
    elif held_days >= 30:
        sell, reason = True, f"기간만료(30일, {profit:+.0f}%)"
    elif strong:
        if px <= peak * 0.92 and peak > buy_price * 1.03:
            sell, reason = True, f"트레일링(고점{int(peak):,}대비-8%)"
        elif rsi.iloc[i] >= 78:
            sell, reason = True, "극과매수(RSI78+)"
    else:
        if overbought and px < float(s.iloc[i-1]):
            sell, reason = True, "과매수 꺾임"
    info = {"close": int(px), "rsi": round(float(rsi.iloc[i]), 1),
            "peak": int(peak), "profit": round(profit, 1),
            "regime": "추세" if strong else "박스"}
    return sell, reason, info

def _load_positions():
    try:
        with open(POS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("positions", [])
    except Exception:
        return []

def _save_positions(positions):
    with open(POS_FILE, "w", encoding="utf-8") as f:
        json.dump({"positions": positions, "updated": datetime.now().strftime("%Y-%m-%d %H:%M")},
                  f, ensure_ascii=False, indent=2)

def scan_buys(date_str, max_picks=10, intraday=False):
    """전 종목에서 저점매수 신호 스캔 (우량주 사전필터 + 거래대금순 상위 max_picks)"""
    stocks = naver_all_rising_parallel()
    cands = []
    for st in stocks:
        code = st.get("itemCode", ""); name = st.get("stockName", "")
        if st.get("stockEndType", "") not in ("stock", ""): continue
        if is_excluded_by_name(name, code): continue
        cl = parse_num(st.get("closePrice", "0")); trd = parse_num(st.get("accumulatedTradingValue", "0"))
        mc = parse_num(st.get("marketValue", "0")); vol = parse_num(st.get("accumulatedTradingVolume", "0"))
        if cl < 2000 or cl > 200000 or trd < 1000 or mc < 1000 or vol <= 0: continue
        cands.append({"code": code, "name": name, "trd": trd})
    hits = []
    def chk(c):
        df = _fetch(c["code"], date_str, intraday=intraday)
        ok, info = buy_signal(df)
        if ok:
            hits.append({**c, **info})
    with ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(chk, cands))
    hits.sort(key=lambda x: -x["trd"])
    return hits[:max_picks]

def run_swing(date_str=None, auto_open=True, intraday=False, do_buys=True):
    """매도 점검(+선택적 신규 매수신호 스캔). 포지션 갱신. payload 반환.
    intraday=True: 장중 분봉 종가근사로 신호 계산(장중 즉시 알림용).
    do_buys=False: 매수 스캔 생략(매도 모니터링만, 가볍게)."""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    positions = _load_positions()
    held_codes = {p["code"] for p in positions}

    # 1) 보유 포지션 매도 점검
    sells, keep = [], []
    for p in positions:
        df = _fetch(p["code"], date_str, intraday=intraday)
        s, reason, info = sell_check(df, p["buyPrice"], p["buyDate"])
        if s:
            sells.append({**p, "reason": reason, **info})
        else:
            p["last"] = info.get("close", p.get("buyPrice"))
            p["profit"] = info.get("profit", 0)
            p["regime"] = info.get("regime", "")
            keep.append(p)

    # 2) 신규 매수신호 스캔 (do_buys일 때만)
    buys = scan_buys(date_str, intraday=intraday) if do_buys else []
    new_positions = list(keep)
    MAX_OPEN = 15   # 동시 보유 상한 (시드 분산 현실 고려)
    if auto_open:
        for b in buys:
            if len(new_positions) >= MAX_OPEN: break
            if b["code"] in held_codes: continue
            new_positions.append({"code": b["code"], "name": b["name"],
                                  "buyDate": date_str, "buyPrice": b["close"]})
            held_codes.add(b["code"])

    _save_positions(new_positions)
    return {"date": date_str, "sells": sells, "buys": buys,
            "holdings": keep, "count_open": len(new_positions)}

def format_swing_telegram(payload):
    L = [f"📈 <b>스윙 추적 (저점매수·고점매도)</b> ({payload['date']})", ""]
    sells = payload.get("sells", [])
    if sells:
        L.append("🔴 <b>매도 신호!</b>")
        for s in sells:
            L.append(f"  • <b>{s['name']}</b>({s['code']}) {s['close']:,}원 "
                     f"<b>{s['profit']:+.1f}%</b> — {s['reason']}")
        L.append("")
    buys = payload.get("buys", [])
    if buys:
        L.append("🟢 <b>저점매수 신호</b> (과매도+반등시작)")
        for b in buys[:10]:
            L.append(f"  • <b>{b['name']}</b>({b['code']}) {b['close']:,}원 "
                     f"(RSI{b['rsi']} %B{b['pctB']})")
        L.append("")
    hold = payload.get("holdings", [])
    if hold:
        L.append(f"👜 <b>보유 추적중 {len(hold)}종목</b>")
        for h in sorted(hold, key=lambda x: -x.get("profit", 0))[:15]:
            L.append(f"  · {h['name']} {h.get('profit',0):+.1f}% [{h.get('regime','')}]")
    if not sells and not buys and not hold:
        L.append("오늘 신호 없음")
    L.append("")
    L.append("⏱ 매수: 신호일 종가 | 매도: 위 신호 시 / -8% 손절")
    return "\n".join(L)

if __name__ == "__main__":
    import sys
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass
    pl = run_swing()
    print(format_swing_telegram(pl))
