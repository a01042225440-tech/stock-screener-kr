#!/usr/bin/env python3
"""
미국 주식 스윙 스크리너 - 한국 스크리너 동일 로직
HUNT(눌림목) + BREAKOUT(돌파) + TREND(추세) 3단계 등급
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import threading
import json, os, time

# =============================================
#  종목 유니버스 (73개)
# =============================================
US_UNIVERSE = {
    "AAPL":"Apple", "MSFT":"Microsoft", "NVDA":"NVIDIA", "GOOGL":"Alphabet",
    "META":"Meta", "AMZN":"Amazon", "TSLA":"Tesla", "AMD":"AMD",
    "AVGO":"Broadcom", "ORCL":"Oracle", "CRM":"Salesforce", "ADBE":"Adobe",
    "NFLX":"Netflix", "QCOM":"Qualcomm", "NOW":"ServiceNow",
    "AMAT":"Applied Mat.", "MU":"Micron", "PANW":"Palo Alto",
    "CRWD":"CrowdStrike", "PLTR":"Palantir", "APP":"AppLovin",
    "COIN":"Coinbase", "UBER":"Uber", "SNOW":"Snowflake",
    "JPM":"JPMorgan", "BAC":"BofA", "GS":"Goldman", "V":"Visa",
    "MA":"Mastercard", "AXP":"Amex", "SPGI":"S&P Global",
    "JNJ":"J&J", "UNH":"UnitedHealth", "LLY":"Eli Lilly",
    "ABBV":"AbbVie", "MRK":"Merck", "TMO":"Thermo Fisher",
    "ISRG":"Intuitive Surg.", "REGN":"Regeneron", "VRTX":"Vertex",
    "HD":"Home Depot", "WMT":"Walmart", "COST":"Costco",
    "NKE":"Nike", "SBUX":"Starbucks", "MCD":"McDonald's",
    "CMG":"Chipotle", "LULU":"Lululemon", "DECK":"Deckers",
    "CAT":"Caterpillar", "DE":"Deere", "HON":"Honeywell",
    "RTX":"RTX", "LMT":"Lockheed", "GD":"General Dynamics",
    "XOM":"ExxonMobil", "CVX":"Chevron", "COP":"ConocoPhillips",
    "OXY":"Occidental",
    "DIS":"Disney", "CMCSA":"Comcast",
    "PG":"P&G", "KO":"Coca-Cola", "PEP":"PepsiCo",
    "AXON":"Axon", "CELH":"Celsius", "ONON":"On Running",
    "ENPH":"Enphase", "NEE":"NextEra", "FSLR":"First Solar",
    "SPOT":"Spotify", "DUOL":"Duolingo", "RBLX":"Roblox",
}

us_scan_status = {
    "running": False, "progress": 0, "total": 0,
    "found": 0, "message": "", "phase": ""
}

# 펀더멘탈 캐시 (24시간)
_fund_cache = {}
_fund_cache_date = ""
_FUND_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "us_fundamentals.json")


# =============================================
#  기술적 지표
# =============================================
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean().replace(0, np.nan)
    return 100 - (100 / (1 + ag / al))


def calc_macd(s, f=12, sl=26, sg=9):
    ef = s.ewm(span=f, adjust=False).mean()
    es = s.ewm(span=sl, adjust=False).mean()
    m = ef - es
    sig = m.ewm(span=sg, adjust=False).mean()
    return m, sig, m - sig


def calc_atr(h, l, c, period=14):
    tr_list = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(1, len(c))]
    return float(np.mean(tr_list[-period:])) if len(tr_list) >= period else 0


# =============================================
#  펀더멘탈 조회 (EPS, 목표주가) - 캐시 활용
# =============================================
def load_fundamentals(tickers):
    """yfinance로 EPS·목표가 일괄 조회. 당일 캐시 재활용."""
    global _fund_cache, _fund_cache_date
    today = datetime.now().strftime("%Y-%m-%d")

    # 파일 캐시 로드
    if _fund_cache_date != today:
        try:
            if os.path.exists(_FUND_CACHE_FILE):
                mtime = os.path.getmtime(_FUND_CACHE_FILE)
                if time.time() - mtime < 86400:  # 24시간 이내
                    with open(_FUND_CACHE_FILE, encoding="utf-8") as f:
                        _fund_cache = json.load(f)
                    _fund_cache_date = today
                    print(f"[US FUND] 캐시 로드: {len(_fund_cache)}개")
                    return
        except Exception as e:
            print(f"[US FUND] 캐시 로드 실패: {e}")

    print(f"[US FUND] 펀더멘탈 조회 중 ({len(tickers)}개)...")
    result = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            eps = float(info.get("trailingEps") or 0)
            target = float(info.get("targetMeanPrice") or 0)
            pe = float(info.get("trailingPE") or 0)
            result[ticker] = {"eps": eps, "target": target, "pe": pe}
        except Exception:
            result[ticker] = {"eps": 0, "target": 0, "pe": 0}

    _fund_cache = result
    _fund_cache_date = today

    # 파일 캐시 저장
    try:
        os.makedirs(os.path.dirname(_FUND_CACHE_FILE), exist_ok=True)
        with open(_FUND_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f)
        print(f"[US FUND] 캐시 저장 완료")
    except Exception as e:
        print(f"[US FUND] 캐시 저장 실패: {e}")


# =============================================
#  환율 조회 (USD → KRW)
# =============================================
def get_usdkrw(date_str=None):
    try:
        if date_str:
            end_dt = datetime.strptime(date_str, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=7)
            fx = yf.download("USDKRW=X",
                             start=start_dt.strftime("%Y-%m-%d"),
                             end=(end_dt+timedelta(days=1)).strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
        else:
            fx = yf.download("USDKRW=X", period="5d", progress=False, auto_adjust=True)
        if fx is not None and len(fx) > 0:
            if isinstance(fx.columns, pd.MultiIndex):
                fx.columns = fx.columns.get_level_values(0)
            rate = float(fx["Close"].iloc[-1])
            if 900 < rate < 2000:
                return round(rate, 1)
    except Exception as e:
        print(f"  [FX] 환율 조회 실패: {e}")
    return 1380.0


# =============================================
#  종목별 스크리닝 (한국 스크리너 동일 로직)
# =============================================
def screen_us_swing(df, ticker="", fund=None):
    """
    한국 스크리너와 동일한 A/B/D(HUNT)/E(BREAKOUT)/K 조건
    """
    if len(df) < 230: return None

    c = df["Close"].values.astype(float)
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].values.astype(float)
    i = len(df) - 1

    # 기본 품질 필터
    if c[i] < 15: return None                          # 페니스톡 제외 ($15 미만)
    dv_m = c[i] * v[i] / 1_000_000
    if dv_m < 30: return None                          # 일 거래대금 $30M 미만 제외
    chg_today = (c[i] - c[i-1]) / c[i-1] * 100 if c[i-1] > 0 else 0
    if abs(chg_today) >= 20: return None               # 20% 이상 급등락 제외

    # ── A 조건: 추세 정배열 (Weinstein Stage 2 + Minervini) ──
    sma5   = float(np.mean(c[i-4:i+1]))
    sma10  = float(np.mean(c[i-9:i+1]))
    sma20  = float(np.mean(c[i-19:i+1]))
    sma60  = float(np.mean(c[i-59:i+1])) if i >= 59 else sma20
    sma200 = float(np.mean(c[i-199:i+1]))
    sma200_30d = float(np.mean(c[i-229:i-29])) if i >= 229 else sma200

    a1 = bool(c[i] > sma200)             # A1. 종가 > 200MA
    a2 = bool(c[i] > sma60)              # A2. 종가 > 60MA
    a3 = bool(sma60 > sma200)            # A3. 60MA > 200MA 정배열
    a4 = bool(sma200 > sma200_30d)       # A4. 200MA 우상향
    a_count = int(a1) + int(a2) + int(a3) + int(a4)

    if not a1: return None               # A1은 무조건 필수

    # ── B 조건: 펀더멘탈 ──
    fund = fund or {}
    eps    = float(fund.get("eps", 0) or 0)
    target = float(fund.get("target", 0) or 0)
    pe     = float(fund.get("pe", 0) or 0)

    b1 = bool(eps > 0)                                          # B1. EPS > 0 (흑자)
    b2 = bool(0 < pe <= 60)                                     # B2. PE 적정 (미국 성장주 기준 60 이하)
    b3 = bool(target > 0 and target >= c[i] * 1.05)            # B3. 목표가 +5%+
    b_count = int(b1) + int(b2) + int(b3)
    b_required_ok = bool(b1 and b3) if (b1 or b3) else True    # 펀더 데이터 없으면 통과

    # ── D 트리거: HUNT 눌림목 (남석관 + Marty Schwartz) ──
    ma20_dist  = (c[i] - sma20) / sma20 * 100 if sma20 > 0 else 0
    avg_vol_5  = float(np.mean(v[i-5:i])) if i >= 5 else 1
    avg_vol_20 = float(np.mean(v[i-20:i])) if i >= 20 else avg_vol_5
    vol_mult_5  = v[i] / avg_vol_5  if avg_vol_5 > 0 else 1.0
    vol_mult_20 = v[i] / avg_vol_20 if avg_vol_20 > 0 else 1.0

    align   = bool(sma5 > sma20 > sma60)
    hi10    = float(np.max(h[i-10:i])) if i >= 10 else float(h[i])
    retrace = (hi10 - l[i]) / hi10 * 100 if hi10 > 0 else 0
    rng_d   = h[i] - l[i]
    lower_tail  = bool(rng_d > 0 and (min(o[i], c[i]) - l[i]) / rng_d >= 0.25)
    touched_ma  = bool(l[i] <= max(sma5, sma20) * 1.005)
    vol_up      = bool(v[i] > v[i-1])

    d1_pullback   = bool(3 <= retrace <= 18)     # 되돌림 3~18%
    d2_above_ma5  = bool(-2 <= ma20_dist <= 8)   # 20MA 근접
    d3_bullish    = bool(o[i] < c[i])            # 양봉
    d4_vol_pickup = bool(lower_tail or touched_ma)
    d5_rsi_ok     = bool(vol_up)

    hunt_trigger = bool(align and d1_pullback and d2_above_ma5
                        and d3_bullish and d4_vol_pickup and d5_rsi_ok)

    # 눌림목 횟수 (1차/2차만 유효)
    pullback_count = 0
    if i >= 79:
        c_win  = c[i-78:i+1]
        s20_win = pd.Series(c_win).rolling(20).mean().values
        above_segs, state = 0, "below"
        for j in range(19, 79):
            sv = s20_win[j]
            if not np.isnan(sv) and sv > 0:
                ratio = c_win[j] / sv
                if state == "below" and ratio > 1.02:
                    above_segs += 1; state = "above"
                elif state == "above" and ratio < 0.97:
                    state = "below"
        pullback_count = max(0, above_segs - 1)
    f1_first_pullback = bool(pullback_count <= 2)

    # ── E 트리거: BREAKOUT 돌파 (Minervini + O'Neil) ──
    high_52w = float(np.max(h[max(0, i-251):i+1]))
    low_52w  = float(np.min(l[max(0, i-251):i+1]))
    proximity_52w = c[i] / high_52w if high_52w > 0 else 0

    e1_new_high  = bool(proximity_52w >= 0.90)    # 52주 고가 10% 이내
    e2_vol_burst = bool(vol_mult_20 >= 2.0)       # 거래량 20일평균 2배+
    e3_price_up  = bool(chg_today >= 3.0)         # 당일 +3% 이상
    e4_rsi_ok    = True                            # RSI는 아래서 계산
    e5_ma20_near = bool(abs(ma20_dist) < 20)      # 20MA 거리 < 20%

    breakout_trigger = bool(e1_new_high and e2_vol_burst and e3_price_up and e5_ma20_near)

    # ── RSI ──
    rv_series = calc_rsi(pd.Series(c), 14)
    rv = float(rv_series.iloc[-1]) if not np.isnan(rv_series.iloc[-1]) else 50.0
    e4_rsi_ok = bool(rv <= 75)
    if not e4_rsi_ok and breakout_trigger:
        breakout_trigger = False

    # ── MACD ──
    _, _, macd_hist = calc_macd(pd.Series(c), 12, 26, 9)
    mh_now = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0

    # ── BB(20,2) 하단 돌파 ──
    bb_mid  = sma20
    bb_std  = float(np.std(c[i-19:i+1], ddof=1))
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos   = (c[i] - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
    bb_width_pct = (bb_upper - bb_lower) / bb_mid * 100 if bb_mid > 0 else 100

    if i >= 20:
        bb_mid_p  = float(np.mean(c[i-20:i]))
        bb_std_p  = float(np.std(c[i-20:i], ddof=1))
        bb_lower_p = bb_mid_p - 2 * bb_std_p
    else:
        bb_lower_p = bb_lower
    bb_break_close    = bool(c[i-1] <= bb_lower_p and c[i] > bb_lower)
    bb_break_intraday = bool(l[i] <= bb_lower and c[i] > bb_lower)
    bb_lower_break    = bool(bb_break_close or bb_break_intraday)

    # ── K 조건: 캔들 품질 ──
    body        = abs(c[i] - o[i])
    upper_wick  = h[i] - max(c[i], o[i])
    candle_range = h[i] - l[i]
    body_ratio  = body / candle_range if candle_range > 0 else 0
    upper_wick_ratio = upper_wick / body if body > 0 else 999

    k1_up_close  = bool(c[i] > c[i-1])
    k2_bullish   = bool(o[i] < c[i])
    k3_short_wick = bool(body > 0 and upper_wick <= body * 0.3)
    candle_pass  = bool(k2_bullish)           # 양봉 필수

    # ── 등급 판정 ──
    trend_full = bool(a_count == 4)
    fund_ok    = b_required_ok

    grade = None
    if trend_full and fund_ok and hunt_trigger and candle_pass:
        grade = "HUNT"
    elif trend_full and fund_ok and breakout_trigger and candle_pass:
        grade = "BREAKOUT"
    elif trend_full and fund_ok and candle_pass:
        grade = "TREND"
    elif a_count >= 3 and candle_pass:
        grade = "WATCH"
    else:
        return None

    # ── 최종 점수 ──
    trigger_score = 0
    if hunt_trigger:      trigger_score += 30
    if breakout_trigger:  trigger_score += 25
    if bb_lower_break:    trigger_score += 10

    target_upside = (target - c[i]) / c[i] * 100 if target > 0 and c[i] > 0 else 0
    candle_score = (int(k1_up_close) + int(k2_bullish) + int(k3_short_wick)) * 5
    final_score = (a_count * 10) + (b_count * 10) + trigger_score + candle_score
    if target_upside >= 20: final_score += 10
    elif target_upside >= 10: final_score += 5

    # 통과 조건 리스트
    P = []
    if a1: P.append("A1.>200MA")
    if a2: P.append("A2.>60MA")
    if a3: P.append("A3.정배열")
    if a4: P.append("A4.200MA↑")
    if b1: P.append(f"B1.EPS+{eps:.2f}")
    if b2: P.append(f"B2.PE{pe:.0f}")
    if b3: P.append(f"B3.목표+{target_upside:.0f}%")
    if hunt_trigger:      P.append(f"🟢HUNT 눌림{retrace:.0f}%/20MA{ma20_dist:+.1f}%")
    if breakout_trigger:  P.append(f"🔴BREAKOUT 52H{proximity_52w*100:.0f}%/{vol_mult_20:.1f}x")
    if bb_lower_break:    P.append("🟣BB하단돌파")

    atr = calc_atr(h, l, c)
    ret_3m = (c[i] - c[max(0, i-63)]) / c[max(0, i-63)] * 100 if i >= 63 else 0
    ret_6m = (c[i] - c[i-126]) / c[i-126] * 100 if i >= 126 and c[i-126] > 0 else 0

    return {
        "passed": P,
        "pass_count": len(P),
        "grade": grade,
        "momentum": round(min(final_score, 100), 1),
        "atr": round(atr, 2),
        "rsi": round(rv, 1),
        "macd_hist": round(mh_now, 4),
        "volume_ratio": round(vol_mult_20, 2),
        "chg_today": round(chg_today, 2),
        "ma20_dist": round(ma20_dist, 2),
        "proximity_52w": round(proximity_52w * 100, 1),
        "pullback_count": int(pullback_count),
        "f1_first_pullback": f1_first_pullback,
        "hunt_trigger": hunt_trigger,
        "breakout_trigger": breakout_trigger,
        "bb_lower_break": bb_lower_break,
        "k1_up_close": k1_up_close,
        "k2_bullish": k2_bullish,
        "k3_short_wick": k3_short_wick,
        "target_upside": round(target_upside, 1),
        "eps": round(eps, 2),
        "pe": round(pe, 1),
        "ret_3m": round(ret_3m, 1),
        "ret_6m": round(ret_6m, 1),
        "a_count": a_count,
        "b_count": b_count,
        "retrace": round(retrace, 1),
        "vol_mult_5": round(vol_mult_5, 2),
    }


# =============================================
#  매매가 계산 (다음날 개장 직후 매수)
# =============================================
def calc_price_us(cl, lo, atr):
    if cl <= 0 or atr <= 0: return None
    buy = round(cl * 1.003, 2)           # 종가 +0.3% (개장 직후 지정가)
    sl  = max(round(buy - 2.0 * atr, 2),
              round(lo * 0.995, 2),
              round(buy * 0.90, 2))      # 최대 -10%
    risk = buy - sl
    if risk <= 0: return None
    t1 = round(buy + 2.0 * risk, 2)
    t2 = round(buy + 3.0 * risk, 2)
    return {
        "buy": buy, "t1": t1, "t2": t2, "sl": sl,
        "rr": round((t1 - buy) / risk, 2),
        "risk_pct": round((buy - sl) / buy * 100, 1),
        "atr": round(atr, 2),
    }


# =============================================
#  메인 스캔
# =============================================
def run_us_scan(date_str):
    global us_scan_status
    results = []
    tickers = list(US_UNIVERSE.keys())
    total   = len(tickers)

    us_scan_status = {"running": True, "progress": 0, "total": total,
                      "found": 0, "message": "환율 조회 중...", "phase": "init"}

    usdkrw = get_usdkrw(date_str)
    print(f"\n[US SCAN] {date_str} | USD/KRW={usdkrw}")

    # 펀더멘탈 로드 (캐시 우선)
    us_scan_status["message"] = "펀더멘탈 조회 중..."
    load_fundamentals(tickers)

    # OHLCV 배치 다운로드
    us_scan_status["message"] = f"데이터 다운로드 중 ({total}개)..."
    us_scan_status["phase"]   = "download"
    try:
        end_dt   = datetime.strptime(date_str, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=600)
        raw = yf.download(
            tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True, timeout=60, group_by="ticker"
        )
    except Exception as e:
        print(f"[US SCAN] Download error: {e}")
        us_scan_status.update({"running": False, "message": f"Download error: {e}"})
        return []

    us_scan_status["message"] = "분석 중..."
    us_scan_status["phase"]   = "analyze"
    print(f"[US SCAN] 다운로드 완료. 분석 시작...")

    grade_order = {"HUNT": 0, "BREAKOUT": 1, "TREND": 2, "WATCH": 3}

    for idx, ticker in enumerate(tickers):
        us_scan_status["progress"] = idx + 1
        if (idx + 1) % 10 == 0:
            us_scan_status["message"] = f"분석 중 {ticker} ({idx+1}/{total})"
        try:
            df = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df = df[df.index <= pd.Timestamp(date_str)]
            if len(df) < 230: continue

            fund = _fund_cache.get(ticker, {})
            r = screen_us_swing(df, ticker, fund)
            if r is None: continue

            last_c  = float(df["Close"].iloc[-1])
            last_o  = float(df["Open"].iloc[-1])
            last_h  = float(df["High"].iloc[-1])
            last_l  = float(df["Low"].iloc[-1])
            last_v  = int(df["Volume"].iloc[-1])
            support_low = float(df["Low"].iloc[max(0, len(df)-10):].min())

            p = calc_price_us(last_c, last_l, r["atr"])
            if p is None: continue

            def to_krw(usd): return int(round(usd * usdkrw, -1))

            results.append({
                "ticker":    ticker,
                "name":      US_UNIVERSE.get(ticker, ticker),
                "grade":     r["grade"],
                "close":     to_krw(last_c),
                "open":      to_krw(last_o),
                "high":      to_krw(last_h),
                "low":       to_krw(last_l),
                "closeUsd":  round(last_c, 2),
                "volume":    last_v,
                "buyPrice":  to_krw(p["buy"]),
                "target1":   to_krw(p["t1"]),
                "target2":   to_krw(p["t2"]),
                "stoploss":  to_krw(p["sl"]),
                "rrRatio":   p["rr"],
                "riskPct":   p["risk_pct"],
                "atr":       to_krw(p["atr"]),
                "usdkrw":    usdkrw,
                "momentum":  r["momentum"],
                "conditionsMet":    r["passed"],
                "conditionsDetail": f'{r["pass_count"]} conds',
                "rsi":        r["rsi"],
                "macdHist":   r["macd_hist"],
                "volumeRatio": r["volume_ratio"],
                "changeRate": r["chg_today"],
                "ma20Dist":   r["ma20_dist"],
                "proximity52w": r["proximity_52w"],
                "pullbackCount": r["pullback_count"],
                "targetUpside": r["target_upside"],
                "eps":        r["eps"],
                "pe":         r["pe"],
                "ret3m":      r["ret_3m"],
                "ret6m":      r["ret_6m"],
                "huntTrigger":     r["hunt_trigger"],
                "breakoutTrigger": r["breakout_trigger"],
                "dataDate":   df.index[-1].strftime("%Y-%m-%d"),
            })
            us_scan_status["found"] = len(results)

        except Exception as e:
            print(f"  [US] {ticker} error: {e}")
            continue

    # 등급 → 점수 순 정렬
    results.sort(key=lambda x: (grade_order.get(x["grade"], 9), -x["momentum"]))
    us_scan_status.update({"running": False, "progress": total, "total": total,
                           "found": len(results), "message": "done", "phase": "done"})

    print(f"[US SCAN] 완료: {len(results)}개 발견")
    for i, s in enumerate(results[:5], 1):
        print(f"  {i} [{s['grade']}] {s['ticker']:<8} ${s['closeUsd']:>8.2f}  Score:{s['momentum']}")
    return results
