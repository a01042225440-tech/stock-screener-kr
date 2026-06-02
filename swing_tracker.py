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

def _fetch(code, date_str, intraday=False, today_vol=0):
    """OHLCV 조회. intraday=True면 확정봉에 오늘 없을 때 분봉 종가근사 봉을 부착.
    today_vol: 장중 당일 누적거래량(거래량 필터용; 분봉엔 부정확하므로 리스트값 주입)."""
    df = naver_ohlcv_fast(code, 300, target_date=date_str)
    if df is None:
        return None
    df = df[df.index <= pd.Timestamp(date_str)]
    if intraday and len(df) > 0 and df.index[-1] < pd.Timestamp(date_str):
        ohlc = naver_today_ohlc(code, date_str)
        if ohlc:
            o, h, l, c = ohlc
            row = pd.DataFrame([{"Open": o, "High": h, "Low": l, "Close": c, "Volume": today_vol}],
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

def _bb(df):
    """볼린저(20,2) 상/중/하 + RSI/신호선 + 거래량5MA 반환"""
    s = df["Close"]
    m20 = s.rolling(20).mean(); sd = s.rolling(20).std()
    lower = m20 - 2*sd; upper = m20 + 2*sd
    d = s.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100/(1 + up/dn); sig = rsi.rolling(9).mean()
    vma5 = df["Volume"].rolling(5).mean()
    return lower, m20, upper, rsi, sig, vma5

def buy_signal(df):
    """저점매수 신호: 오늘 BB(20,2) 하한선 상향돌파 + 오늘 종가 양봉(시가<종가) + RSI골든크로스 + 거래량1.3x.
    (사용자 확정 조건. 검증 +208%/MDD~9%)"""
    if df is None or len(df) < 60:
        return False, {}
    s = df["Close"]; o = df["Open"]; h = df["High"]; l = df["Low"]; v = df["Volume"]
    lower, m20, upper, rsi, sig, vma5 = _bb(df)
    i = -1
    if pd.isna(lower.iloc[i]) or pd.isna(lower.iloc[i-1]) or pd.isna(sig.iloc[i-1]) or pd.isna(vma5.iloc[i]):
        return False, {}
    cross  = (s.iloc[i-1] <= lower.iloc[i-1]) and (s.iloc[i] > lower.iloc[i])   # ① 오늘 하한선 상향돌파
    rng    = h.iloc[i] - l.iloc[i]
    bullish = o.iloc[i] < s.iloc[i]                                            # ② 오늘 양봉(시가 < 종가) — 필수
    strong  = rng > 0 and (s.iloc[i] - l.iloc[i]) / rng >= 0.5                 #    + 종가가 캔들 상단 50% (강한 양봉)
    golden  = (rsi.iloc[i] > sig.iloc[i]) and (rsi.iloc[i-1] <= sig.iloc[i-1]) # ③ RSI 골든크로스
    volok   = v.iloc[i] > vma5.iloc[i] * 1.3                                   # ④ 거래량 1.3배
    ok = bool(cross and bullish and strong and golden and volok)
    info = {"rsi": round(float(rsi.iloc[i]), 1), "close": int(s.iloc[i]),
            "lower": int(lower.iloc[i]), "upper": int(upper.iloc[i])}
    return ok, info

def sell_check(df, buy_price, buy_date):
    """매도 판정: 상한선 도달(익절) / 하한선 재이탈(무효손절) / -10% 재난손절 / 40일 만료.
    반환 (sell?, reason, info)"""
    if df is None or len(df) < 60:
        return False, "", {}
    s = df["Close"]
    lower, m20, upper, rsi, sig, vma5 = _bb(df)
    i = -1; px = float(s.iloc[i])
    profit = (px - buy_price)/buy_price*100
    held_days = len(df[df.index >= pd.Timestamp(buy_date)])
    sell, reason = False, ""
    if px <= buy_price * 0.90:
        sell, reason = True, "재난손절(-10%)"
    elif (not pd.isna(lower.iloc[i])) and px < float(lower.iloc[i]):
        sell, reason = True, f"무효손절(하한이탈, {profit:+.0f}%)"
    elif (not pd.isna(upper.iloc[i])) and px >= float(upper.iloc[i]):
        sell, reason = True, f"상한선 도달 익절(+{profit:.0f}%)"
    elif held_days >= 40:
        sell, reason = True, f"40일만료({profit:+.0f}%)"
    info = {"close": int(px), "rsi": round(float(rsi.iloc[i]), 1), "profit": round(profit, 1),
            "regime": "BB"}
    return sell, reason, info

def _load_positions():
    try:
        with open(POS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("positions", [])
    except Exception:
        return []

def _save_positions(positions):
    # 'updated' 타임스탬프를 넣지 않음 → 포지션 실제 변동시에만 git diff 발생(커밋 노이즈 방지)
    with open(POS_FILE, "w", encoding="utf-8") as f:
        json.dump({"positions": positions}, f, ensure_ascii=False, indent=2)

def _halted(st):
    tst = st.get("tradeStopType", {})
    nm = (tst.get("name", "") if isinstance(tst, dict) else str(tst)).upper()
    return nm == "HALTED"

def scan_buys(date_str, max_picks=10, intraday=False, market="ALL"):
    """전 종목에서 저점매수 신호 스캔 (우량주 사전필터 + 거래대금순 상위 max_picks).
    market: 'ALL'/'KOSPI'/'KOSDAQ' (대상변경). 거래정지(HALTED) 자동 제외."""
    mkt_sel = (market or "ALL").upper()
    stocks = naver_all_rising_parallel()
    cands = []
    for st in stocks:
        code = st.get("itemCode", ""); name = st.get("stockName", "")
        if mkt_sel in ("KOSPI", "KOSDAQ") and st.get("_market", "") != mkt_sel: continue
        if st.get("stockEndType", "") not in ("stock", ""): continue
        if _halted(st): continue
        if is_excluded_by_name(name, code): continue
        cl = parse_num(st.get("closePrice", "0")); trd = parse_num(st.get("accumulatedTradingValue", "0"))
        mc = parse_num(st.get("marketValue", "0")); vol = parse_num(st.get("accumulatedTradingVolume", "0"))
        if cl < 2000 or cl > 200000 or trd < 1000 or mc < 1000 or vol <= 0: continue
        cands.append({"code": code, "name": name, "trd": trd, "vol": vol})
    hits = []
    def chk(c):
        df = _fetch(c["code"], date_str, intraday=intraday, today_vol=c.get("vol", 0))
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
            try:
                from trade_log import log_closed
                log_closed("스윙", p["code"], p.get("name", p["code"]), p["buyDate"],
                           p["buyPrice"], info.get("close", p["buyPrice"]), info.get("profit", 0), reason)
            except Exception as e:
                print(f"[SWING] log error: {e}")
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
        L.append("🟢 <b>저점매수 신호</b> (BB하한 상향돌파+강한양봉+RSI골든+거래량)")
        for b in buys[:10]:
            L.append(f"  • <b>{b['name']}</b>({b['code']}) {b['close']:,}원 "
                     f"(목표 상한 {b.get('upper','-'):,} / RSI{b['rsi']})")
        L.append("")
    hold = payload.get("holdings", [])
    if hold:
        L.append(f"👜 <b>보유 추적중 {len(hold)}종목</b>")
        for h in sorted(hold, key=lambda x: -x.get("profit", 0))[:15]:
            L.append(f"  · {h['name']} {h.get('profit',0):+.1f}%")
    if not sells and not buys and not hold:
        L.append("오늘 신호 없음")
    L.append("")
    L.append("⏱ 매수: 신호일 종가 | 매도: BB상한 도달/하한 재이탈/-10% / 40일")
    return "\n".join(L)

if __name__ == "__main__":
    import sys
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass
    pl = run_swing()
    print(format_swing_telegram(pl))
