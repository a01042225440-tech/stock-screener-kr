#!/usr/bin/env python3
"""
키움증권 종가매수 조건검색 스크리너 (고속 버전)
원본: 종가매수.xls (A~L 전체 AND) + BB하한 상향돌파 + 보조지표
매수: 종가+0.5% | 손절: ATR x1.5 | T1: BB상한 | T2: 3R
"""
import sys, os, json, time, traceback, pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except: pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except: pass

import numpy as np, pandas as pd
import requests
from requests.adapters import HTTPAdapter
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request as flask_request


app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="/static")

# =============================================
#  고속 HTTP 세션 (커넥션 풀링)
# =============================================
def create_fast_session():
    """커넥션 풀 20개, keep-alive 재사용"""
    s = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=20,
        pool_maxsize=25,
        max_retries=1
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    return s

# 글로벌 세션 (서버 시작시 1회 생성, 모든 요청에 재사용)
_session = create_fast_session()

# =============================================
#  메모리 캐시 (같은 날 재검색시 즉시)
# =============================================
_ohlcv_cache = {}
_cache_date = ""
_cache_lock = threading.Lock()

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')

def get_cached_ohlcv(key):
    """메모리 캐시에서 OHLCV 조회 (key = "code_YYYY-MM-DD" 또는 "code")"""
    global _ohlcv_cache, _cache_date
    today = datetime.now().strftime("%Y%m%d")
    if _cache_date != today:
        # 날짜 변경시 캐시 리셋 + 디스크에서 로드 시도
        with _cache_lock:
            _ohlcv_cache = {}
            _cache_date = today
            cache_file = os.path.join(CACHE_DIR, f"{today}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        _ohlcv_cache = pickle.load(f)
                    print(f"  [CACHE] Loaded {len(_ohlcv_cache)} stocks from disk")
                except:
                    pass
    return _ohlcv_cache.get(key)

def set_cached_ohlcv(key, df):
    """메모리 캐시에 OHLCV 저장"""
    with _cache_lock:
        _ohlcv_cache[key] = df

def save_cache_to_disk():
    """캐시를 디스크에 저장 (스캔 완료 후)"""
    if not _ohlcv_cache:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{_cache_date}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(_ohlcv_cache, f)
        print(f"  [CACHE] Saved {len(_ohlcv_cache)} stocks to disk")
    except Exception as e:
        print(f"  [CACHE] Save failed: {e}")

# =============================================
#  네이버 금융 API (세션 기반 고속)
# =============================================
def naver_stock_list(market, sort_type="up"):
    all_stocks = []
    page, page_size = 1, 100
    while True:
        url = f"https://m.stock.naver.com/api/stocks/{sort_type}/{market}?page={page}&pageSize={page_size}"
        try:
            r = _session.get(url, timeout=15)
            if r.status_code != 200: break
            data = r.json()
            stocks = data.get("stocks", [])
            if not stocks: break
            all_stocks.extend(stocks)
            total = data.get("totalCount", 0)
            if len(all_stocks) >= total or page * page_size >= total: break
            page += 1
            time.sleep(0.05)  # 0.1 → 0.05 (세션 재사용이라 부하 적음)
        except Exception as e:
            print(f"  [NAVER] {market} page {page} fail: {e}")
            break
    return all_stocks

def naver_all_rising_parallel():
    """KOSPI + KOSDAQ 전체 종목 조회 (시총순) - BB하단 돌파 탐지용"""
    results = {}
    def fetch(market):
        # 시총순 전체 종목 조회 (상승/하락 무관)
        stocks = naver_stock_list(market, "marketValue")
        for s in stocks:
            s["_market"] = market
        results[market] = stocks

    t1 = threading.Thread(target=fetch, args=("KOSPI",))
    t2 = threading.Thread(target=fetch, args=("KOSDAQ",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    kospi = results.get("KOSPI", [])
    kosdaq = results.get("KOSDAQ", [])
    print(f"  [ALL] KOSPI: {len(kospi)} | KOSDAQ: {len(kosdaq)} (시총순 전체)")
    return kospi + kosdaq

def naver_ohlcv_fast(code, days=250, target_date=None):
    """세션 기반 고속 OHLCV 조회 + 캐시"""
    cache_key = f"{code}_{target_date}" if target_date else code
    cached = get_cached_ohlcv(cache_key)
    if cached is not None:
        return cached

    end = datetime.strptime(target_date, "%Y-%m-%d") if target_date else datetime.now()
    start = end - timedelta(days=int(days * 1.5))
    params = {
        "symbol": code, "requestType": "1",
        "startTime": start.strftime("%Y%m%d"),
        "endTime": end.strftime("%Y%m%d"),
        "timeframe": "day"
    }
    try:
        r = _session.get("https://fchart.stock.naver.com/siseJson.naver",
                         params=params, timeout=15)
        text = r.text.strip()
        rows = []
        for line in text.split('\n'):
            line = line.strip().rstrip(',')
            if not line: continue
            if ("'2" in line or '"2' in line) and line.startswith('['):
                line = line.strip('[]')
                parts = [p.strip().strip("'\"") for p in line.split(',')]
                if len(parts) >= 6:
                    try:
                        dt = parts[0].strip()
                        o, h, l, c, v = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                        if c > 0:
                            rows.append({"Date": dt, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
                    except: continue
        if len(rows) < 201:
            set_cached_ohlcv(cache_key, None)
            return None
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df = df.set_index("Date").sort_index()
        set_cached_ohlcv(cache_key, df)
        return df
    except:
        return None

# =============================================
#  위험종목 제외 필터
# =============================================
EXCLUDE_KEYWORDS = [
    "ETF", "ETN", "KODEX", "TIGER", "KOSEF", "KINDEX", "HANARO",
    "SOL ", "ACE ", "ARIRANG", "BNK", "FOCUS", "KBSTAR", "TREX",
    "파워", "인버스", "레버리지", "2X", "3X",
    "스팩", "SPAC", "기업인수",
    "리츠", "REIT",
    "선물", "옵션",
]

def is_excluded_by_name(name, code):
    name_upper = name.upper()
    for kw in EXCLUDE_KEYWORDS:
        if kw.upper() in name_upper:
            return True
    if len(code) == 6 and code[-1] in ("5", "7", "8", "9", "K", "L"):
        return True
    if name.endswith("우") or name.endswith("우B") or name.endswith("우C"):
        return True
    if "우선" in name:
        return True
    return False

def check_warning_batch(codes):
    """투자경고/관리종목 병렬 체크"""
    excluded = set()
    def check_one(code):
        try:
            r = _session.get(f"https://m.stock.naver.com/api/stock/{code}/basic", timeout=4)
            if r.status_code != 200: return
            data = r.json()
            caution = str(data.get("investCautionType", "")).upper()
            warning = str(data.get("stockWarningType", "")).upper()
            if caution and caution not in ("NONE", "NORMAL", ""):
                excluded.add(code)
            elif warning and warning not in ("NONE", "NORMAL", ""):
                excluded.add(code)
            elif data.get("tradingHalt"):
                excluded.add(code)
            elif data.get("delistingReason"):
                excluded.add(code)
        except:
            pass

    with ThreadPoolExecutor(max_workers=15) as executor:
        executor.map(check_one, codes)
    return excluded

def fetch_investor_data(code):
    """네이버 integration API에서 수급 + 재무 + 업종 + 컨센서스 일괄 조회"""
    try:
        r = _session.get(f"https://m.stock.naver.com/api/stock/{code}/integration", timeout=5)
        if r.status_code != 200:
            return None
        data = r.json()

        # ── 1. 외국인/기관 수급 (5일) ──
        deals = data.get("dealTrendInfos", [])
        foreign_net_5d = 0
        inst_net_5d = 0
        foreign_buy_days = 0
        inst_buy_days = 0
        foreign_hold = 0.0

        for d in deals[:5]:
            fb = d.get("foreignerPureBuyQuant", "0")
            ob = d.get("organPureBuyQuant", "0")
            fb_val = int(str(fb).replace(",", "").replace("+", "")) if fb else 0
            ob_val = int(str(ob).replace(",", "").replace("+", "")) if ob else 0
            foreign_net_5d += fb_val
            inst_net_5d += ob_val
            if fb_val > 0: foreign_buy_days += 1
            if ob_val > 0: inst_buy_days += 1

        if deals:
            hold_str = deals[0].get("foreignerHoldRatio", "0%")
            try: foreign_hold = float(str(hold_str).replace("%", "").replace(",", ""))
            except: foreign_hold = 0.0

        # ── 2. 재무제표 (totalInfos) ──
        total_map = {}
        for item in data.get("totalInfos", []):
            total_map[item.get("code", "")] = item.get("value", "")

        def parse_val(s):
            try: return float(str(s).replace(",", "").replace("배", "").replace("원", "").replace("%", "").replace("억", "").replace("백만", ""))
            except: return 0.0

        per = parse_val(total_map.get("per", "0"))
        pbr = parse_val(total_map.get("pbr", "0"))
        eps = parse_val(total_map.get("eps", "0"))
        bps = parse_val(total_map.get("bps", "0"))
        div_yield = parse_val(total_map.get("dividendYieldRatio", "0"))
        high_52w = parse_val(total_map.get("highPriceOf52Weeks", "0"))
        low_52w = parse_val(total_map.get("lowPriceOf52Weeks", "0"))

        # ── 3. 증권사 컨센서스 ──
        cons = data.get("consensusInfo") or {}
        target_price = parse_val(cons.get("priceTargetMean", "0"))
        recomm_mean = parse_val(cons.get("recommMean", "0"))  # 1=강력매수 ~ 5=강력매도

        # ── 4. 동종업종 비교 (업종 모멘텀) ──
        sector_stocks = data.get("industryCompareInfo", [])
        sector_up = 0
        sector_total = len(sector_stocks)
        for ss in sector_stocks:
            chg = parse_val(ss.get("fluctuationsRatio", "0"))
            if chg > 0: sector_up += 1
        sector_ratio = sector_up / sector_total * 100 if sector_total > 0 else 50

        return {
            "foreign_net_5d": foreign_net_5d,
            "inst_net_5d": inst_net_5d,
            "foreign_buy_days": foreign_buy_days,
            "inst_buy_days": inst_buy_days,
            "foreign_hold": foreign_hold,
            "per": per, "pbr": pbr, "eps": eps, "bps": bps,
            "div_yield": div_yield,
            "high_52w": high_52w, "low_52w": low_52w,
            "target_price": target_price, "recomm_mean": recomm_mean,
            "sector_ratio": sector_ratio,
        }
    except:
        return None

def parse_num(s):
    try: return int(str(s).replace(",", ""))
    except: return 0

def parse_float(s):
    try: return float(str(s).replace(",", ""))
    except: return 0.0

# =============================================
#  기술적 지표 계산
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

def calc_stoch(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(dp).mean()

def calc_atr(h, l, c, period=14):
    tr_list = []
    for j in range(1, len(c)):
        tr = max(h[j] - l[j], abs(h[j] - c[j-1]), abs(l[j] - c[j-1]))
        tr_list.append(tr)
    if len(tr_list) < period: return 0
    return float(np.mean(tr_list[-period:]))

def calc_obv_trend(c, v, lookback=20):
    i = len(c) - 1
    if i < lookback: return 0
    obv = 0
    obv_start = None
    for j in range(max(0, i - lookback), i + 1):
        if j == 0: continue
        if c[j] > c[j-1]: obv += v[j]
        elif c[j] < c[j-1]: obv -= v[j]
        if obv_start is None: obv_start = obv
    if obv_start is None: return 0
    return obv - obv_start

# =============================================
#  고급 팩터 계산 (상승확률 정밀화)
# =============================================
def calc_advanced_factors(df):
    """OHLCV에서 10가지 고급 팩터 계산 → rankScore에 반영"""
    c = df["Close"].values
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    v = df["Volume"].values
    i = len(df) - 1
    result = {}

    # ── F1. 52주 고가 저항 분석 ──
    high_52w = max(h[max(0,i-249):i+1])
    low_52w = min(l[max(0,i-249):i+1])
    range_52w = high_52w - low_52w if high_52w > low_52w else 1
    pos_52w = (c[i] - low_52w) / range_52w * 100  # 52주 내 위치(%)
    dist_from_high = (high_52w - c[i]) / high_52w * 100  # 고가 대비 하락률
    result["pos52w"] = round(pos_52w, 1)
    result["distFromHigh"] = round(dist_from_high, 1)
    # 90% 이상이면 저항 근접 → 감점
    if pos_52w >= 95: result["f1_score"] = -5   # 52주 고가 근접 = 저항
    elif pos_52w >= 85: result["f1_score"] = -2
    elif 60 <= pos_52w < 85: result["f1_score"] = 5  # 상승추세 중간 = 최적
    elif 40 <= pos_52w < 60: result["f1_score"] = 3  # 중간
    else: result["f1_score"] = 0

    # ── F2. 변동성 수축 (Volatility Squeeze) ──
    atr_14 = np.mean([max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-13, i+1)])
    atr_60 = np.mean([max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-59, i+1)])
    atr_ratio = atr_14 / atr_60 if atr_60 > 0 else 1
    bb_width_now = (np.std(c[i-19:i+1]) * 4) / np.mean(c[i-19:i+1]) if np.mean(c[i-19:i+1]) > 0 else 0
    bb_widths = [(np.std(c[j-19:j+1]) * 4) / np.mean(c[j-19:j+1]) if np.mean(c[j-19:j+1]) > 0 else 0 for j in range(i-59, i+1)]
    squeeze = bool(atr_ratio < 0.8 and bb_width_now <= sorted(bb_widths)[len(bb_widths)//4])
    result["atrRatio"] = round(float(atr_ratio), 2)
    result["squeeze"] = squeeze
    if squeeze: result["f2_score"] = 8
    elif atr_ratio < 0.9: result["f2_score"] = 4
    else: result["f2_score"] = 0

    # ── F3. VWAP 거리 (기관 매매 기준선) ──
    tp_arr = (h[i-19:i+1] + l[i-19:i+1] + c[i-19:i+1]) / 3
    vwap_20 = np.sum(tp_arr * v[i-19:i+1]) / np.sum(v[i-19:i+1]) if np.sum(v[i-19:i+1]) > 0 else c[i]
    vwap_dist = (c[i] - vwap_20) / vwap_20 * 100
    result["vwapDist"] = round(vwap_dist, 1)
    if 0 <= vwap_dist <= 3: result["f3_score"] = 6   # VWAP 바로 위 = 지지
    elif -2 <= vwap_dist < 0: result["f3_score"] = 4  # VWAP 근접 하방
    elif 3 < vwap_dist <= 8: result["f3_score"] = 2
    else: result["f3_score"] = 0

    # ── F4. Chaikin Money Flow (자금유입 품질) ──
    mf_sum = 0; vol_sum = 0
    for j in range(i-19, i+1):
        rng = h[j] - l[j]
        mf_mult = ((c[j] - l[j]) - (h[j] - c[j])) / rng if rng > 0 else 0
        mf_sum += mf_mult * v[j]
        vol_sum += v[j]
    cmf = mf_sum / vol_sum if vol_sum > 0 else 0
    result["cmf"] = round(cmf, 3)
    if cmf > 0.15: result["f4_score"] = 7
    elif cmf > 0.05: result["f4_score"] = 5
    elif cmf > 0: result["f4_score"] = 2
    else: result["f4_score"] = 0

    # ── F5. ROC 모멘텀 가속도 ──
    roc_10 = (c[i] - c[i-10]) / c[i-10] * 100 if c[i-10] > 0 else 0
    roc_10_prev = (c[i-5] - c[i-15]) / c[i-15] * 100 if c[i-15] > 0 else 0
    roc_accel = roc_10 - roc_10_prev
    result["rocAccel"] = round(roc_accel, 2)
    if roc_accel > 3: result["f5_score"] = 6
    elif roc_accel > 0: result["f5_score"] = 4
    elif roc_accel > -2: result["f5_score"] = 1
    else: result["f5_score"] = -3  # 모멘텀 급감속

    # ── F6. 가격 압축도 (Consolidation Tightness) ──
    range_10 = (max(h[i-9:i+1]) - min(l[i-9:i+1])) / c[i] if c[i] > 0 else 0
    range_20 = (max(h[i-19:i+1]) - min(l[i-19:i+1])) / c[i] if c[i] > 0 else 0
    compression = range_10 / range_20 if range_20 > 0 else 1
    # Inside day count
    inside_days = sum(1 for j in range(i-9, i+1) if j > 0 and h[j] < h[j-1] and l[j] > l[j-1])
    result["compression"] = round(compression, 2)
    result["insideDays"] = inside_days
    if compression < 0.5 and inside_days >= 2: result["f6_score"] = 7
    elif compression < 0.6: result["f6_score"] = 4
    else: result["f6_score"] = 0

    # ── F7. 갭 분석 (최근 5일 내 갭업 확인) ──
    gap_score = 0
    for j in range(i-4, i+1):
        if j <= 0: continue
        gap_pct = (o[j] - c[j-1]) / c[j-1] * 100 if c[j-1] > 0 else 0
        if gap_pct > 1.5:  # 갭업 1.5% 이상
            gap_held = min(l[j:i+1]) > c[j-1]  # 갭 메우지 않음
            vol_ratio_gap = v[j] / np.mean(v[max(0,j-20):j]) if np.mean(v[max(0,j-20):j]) > 0 else 1
            if gap_held and vol_ratio_gap > 1.5:
                gap_score = 6  # 거래량 동반 미충전 갭업 = 매우 강세
            elif gap_held:
                gap_score = max(gap_score, 4)
    # 갭다운(저항) 체크
    for j in range(i-9, i+1):
        if j <= 0: continue
        gap_down = (c[j-1] - o[j]) / c[j-1] * 100 if c[j-1] > 0 else 0
        if gap_down > 2.0:
            gap_top = c[j-1]
            if c[i] < gap_top and c[i] > gap_top * 0.97:
                gap_score = max(gap_score - 3, 0)  # 갭다운 저항 근접
    result["f7_score"] = gap_score

    # ── F8. 수급 가속도 (5일 데이터 내에서 최근 vs 이전 비교) ──
    # → rankScore에서 investor data로 계산 (여기선 placeholder)
    result["f8_score"] = 0

    # ── F9. 다중 시간대 Z-Score ──
    mean_20 = np.mean(c[i-19:i+1])
    std_20 = np.std(c[i-19:i+1], ddof=1) if len(c[i-19:i+1]) > 1 else 1
    z_20 = (c[i] - mean_20) / std_20 if std_20 > 0 else 0
    mean_60 = np.mean(c[i-59:i+1])
    std_60 = np.std(c[i-59:i+1], ddof=1) if len(c[i-59:i+1]) > 1 else 1
    z_60 = (c[i] - mean_60) / std_60 if std_60 > 0 else 0
    result["z20"] = round(z_20, 2)
    result["z60"] = round(z_60, 2)
    # 단기 눌림 + 장기 상승 = 최적
    if -1.5 <= z_20 <= -0.5 and z_60 > 0: result["f9_score"] = 7
    elif -0.5 <= z_20 <= 1.0 and z_60 > 0: result["f9_score"] = 4
    elif z_20 > 2.0: result["f9_score"] = -3  # 과열
    else: result["f9_score"] = 1

    # ── F10. 추세 품질 (ADX + R-squared) ──
    # ADX 계산 (간략화)
    plus_dm = [max(h[j]-h[j-1], 0) if h[j]-h[j-1] > l[j-1]-l[j] else 0 for j in range(i-27, i+1)]
    minus_dm = [max(l[j-1]-l[j], 0) if l[j-1]-l[j] > h[j]-h[j-1] else 0 for j in range(i-27, i+1)]
    tr_list = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-27, i+1)]
    atr_adx = np.mean(tr_list[-14:]) if len(tr_list) >= 14 else 1
    plus_di = 100 * np.mean(plus_dm[-14:]) / atr_adx if atr_adx > 0 else 0
    minus_di = 100 * np.mean(minus_dm[-14:]) / atr_adx if atr_adx > 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
    adx = dx  # 단순화

    # R-squared (20일 선형회귀 적합도)
    x = np.arange(20)
    y = c[i-19:i+1]
    if len(y) == 20:
        slope = np.polyfit(x, y, 1)[0]
        y_pred = np.polyval(np.polyfit(x, y, 1), x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        slope = 0; r_sq = 0

    result["adx"] = round(adx, 1)
    result["rSquared"] = round(r_sq, 2)
    result["trendSlope"] = round(slope, 1)
    trend_quality = adx * r_sq
    if trend_quality > 20 and slope > 0: result["f10_score"] = 6
    elif trend_quality > 10 and slope > 0: result["f10_score"] = 4
    elif adx < 15: result["f10_score"] = 1  # 횡보장
    else: result["f10_score"] = 0

    # 총 고급팩터 점수 합산 (디버그용)
    total = sum(result.get(f"f{n}_score", 0) for n in range(1, 11))
    result["advancedTotal"] = total

    return result

# =============================================
#  조건별 탈락 카운터 (디버그용)
# =============================================
_debug_reject = {
    "T1": 0, "T2": 0, "T3": 0, "T4": 0, "T5": 0,
    "T6": 0, "T7": 0, "T8": 0, "T9": 0, "T10": 0,
    "T11": 0, "T12": 0, "T13": 0, "T14": 0,
    "limit": 0, "data": 0, "total": 0
}
_debug_lock = threading.Lock()

def reset_debug():
    global _debug_reject
    _debug_reject = {k: 0 for k in _debug_reject}

def print_debug_stats():
    print("\n  ===== CONDITION REJECTION STATS =====")
    for k, v in _debug_reject.items():
        if k in ("total", "data"): continue
        bar = "#" * min(v, 50)
        print(f"  {k:>5}: {v:>5} rejected  {bar}")
    print(f"  {'total':>5}: {_debug_reject['total']:>5} analyzed")
    print(f"  {'data':>5}: {_debug_reject['data']:>5} insufficient data")
    print("  =====================================\n")

# =============================================
#  키움증권 종가매수 조건검색 + 보조지표 (11+5 조건)
#  원본: 종가매수.xls (A~L 전체 AND) + 추가 보조지표
# =============================================
def screen_pro(df, name="", code="", mcap=0):
    """
    ★ 전세계 검증 조건식 통합 (한달 10%+ 수익 최적화)
    - Minervini Trend Template
    - O'Neil CANSLIM 기술적 조건
    - Weinstein Stage 2 진입
    - George & Hwang 52주 고가 모멘텀 (학술 1위 검증)
    - 한국시장 기관/외인 수급 가중
    - VCP (변동성 수축 패턴)
    """
    global _debug_reject
    if len(df) < 201:
        with _debug_lock: _debug_reject["data"] += 1
        return None

    with _debug_lock: _debug_reject["total"] += 1

    c = df["Close"].values
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    v = df["Volume"].values
    i = len(df) - 1

    # ── 이동평균 사전 계산 ──
    sma10  = np.mean(c[i-9:i+1])
    sma20  = np.mean(c[i-19:i+1])
    sma50  = np.mean(c[i-49:i+1])
    sma150 = np.mean(c[i-149:i+1])
    sma200 = np.mean(c[i-199:i+1])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [필수 1] Minervini Trend Template (8개 조건)
    # 검증: 2003년 이후 대형 상승주 95%+ 해당
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # T1. 종가 > 200MA (장기 상승추세)
    if c[i] <= sma200:
        with _debug_lock: _debug_reject["T1"] += 1
        return None

    # T2. 종가 > 150MA (중장기 상승추세)
    if c[i] <= sma150:
        with _debug_lock: _debug_reject["T2"] += 1
        return None

    # T3. 종가 > 50MA (중기 상승추세)
    if c[i] <= sma50:
        with _debug_lock: _debug_reject["T3"] += 1
        return None

    # T4. 50MA > 150MA > 200MA (이평선 정배열)
    if not (sma50 > sma150 > sma200):
        with _debug_lock: _debug_reject["T4"] += 1
        return None

    # T5. 200MA 상승 추세 (1개월 이상) - 20일 전보다 높아야
    sma200_20d_ago = np.mean(c[i-219:i-19]) if i >= 219 else sma200
    if sma200 <= sma200_20d_ago:
        with _debug_lock: _debug_reject["T5"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [필수 2] George & Hwang 52주 고가 모멘텀
    # 학술 검증: 가장 강력한 단월 수익 예측 지표
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # T6. 52주 최고가 대비 75% 이상 (상승 모멘텀 구간)
    high_52w = np.max(h[max(0, i-251):i+1])
    low_52w  = np.min(l[max(0, i-251):i+1])
    proximity_52w = c[i] / high_52w if high_52w > 0 else 0
    if proximity_52w < 0.75:  # 52주 고가 25% 이내
        with _debug_lock: _debug_reject["T6"] += 1
        return None

    # T7. 52주 저가 대비 25% 이상 상승 (기저 확인)
    if low_52w > 0 and c[i] < low_52w * 1.25:
        with _debug_lock: _debug_reject["T7"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [필수 3] O'Neil CANSLIM - 거래량 급증 (브레이크아웃)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # T8. 거래량 50일 평균 대비 150%+ (기관 매집 신호)
    av50 = np.mean(v[max(0, i-50):i]) if i >= 50 else np.mean(v[:i])
    vol_ratio = v[i] / av50 if av50 > 0 else 1.0
    if vol_ratio < 1.5:
        with _debug_lock: _debug_reject["T8"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [필수 4] 당일 캔들 품질
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # T9. 양봉 (시가 < 종가)
    if o[i] >= c[i]:
        with _debug_lock: _debug_reject["T9"] += 1
        return None

    # T10. 종가 상승 (전일 대비)
    if c[i] <= c[i-1]:
        with _debug_lock: _debug_reject["T10"] += 1
        return None

    # T11. 윗꼬리 과다 제외 (윗꼬리 > 몸통*1.5 = 매도 강함)
    body = abs(c[i] - o[i])
    upper_wick = h[i] - max(c[i], o[i])
    if body > 0 and upper_wick > body * 1.5:
        with _debug_lock: _debug_reject["T11"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [필수 5] 기본 조건
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # T12. 주가 5,000원 이상 (한국 기관 매수 최소 기준)
    if c[i] < 5000:
        with _debug_lock: _debug_reject["T12"] += 1
        return None

    # T13. 거래대금 50억 이상 (주도주 유동성)
    trading_value = c[i] * v[i] / 100000000
    if trading_value < 50:
        with _debug_lock: _debug_reject["T13"] += 1
        return None

    # T14. 시가총액 1000억 이상
    if mcap > 0 and mcap < 1000:
        with _debug_lock: _debug_reject["T14"] += 1
        return None

    # T15. 상한가 제외
    chg = (c[i] - c[i-1]) / c[i-1] * 100
    if chg >= 29:
        with _debug_lock: _debug_reject["limit"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 여기까지 통과 = 필수 15개 조건 ALL PASS
    # 아래는 점수화 (보너스)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    P = [
        "T1.200MA위", "T2.150MA위", "T3.50MA위",
        "T4.정배열", "T5.200MA상승", f"T6.52주{proximity_52w*100:.0f}%",
        "T7.저점+25%", f"T8.거래량{vol_ratio:.1f}x",
        "T9.양봉", "T10.전일↑", "T11.캔들양호",
        "T12.주가5천↑", f"T13.거래대금{trading_value:.0f}억", "T14.시총1천억↑"
    ]
    F_list = []
    score = 50  # 기본 필수 통과 = 50점

    # ── 보너스 1. RSI 모멘텀 구간 (IBD/Minervini: 50~75 최적)
    rv = calc_rsi(pd.Series(c), 14).iloc[-1]
    if not np.isnan(rv):
        if 50 <= rv <= 75:
            P.append(f"B1.RSI최적{rv:.0f}"); score += 10
        elif 40 <= rv < 50:
            P.append(f"B1.RSI적정{rv:.0f}"); score += 5
        elif rv > 75:
            F_list.append(f"B1.RSI과매수{rv:.0f}")
        else:
            F_list.append(f"B1.RSI약{rv:.0f}")
    else:
        rv = 50.0

    # ── 보너스 2. MACD 히스토그램 양전환 (모멘텀 가속)
    _, _, macd_hist = calc_macd(pd.Series(c), 12, 26, 9)
    mh_now  = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0
    mh_prev = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 and not np.isnan(macd_hist.iloc[-2]) else 0
    if mh_now > 0 and mh_now > mh_prev:
        P.append("B2.MACD가속↑"); score += 8
    elif mh_now > mh_prev:
        P.append("B2.MACD반전"); score += 5
    else:
        F_list.append("B2.MACD약화")

    # ── 보너스 3. 52주 신고가 근접 (George&Hwang: 8% 이내 = 최강 신호)
    if proximity_52w >= 0.97:
        P.append(f"B3.52주신고가{proximity_52w*100:.0f}%"); score += 12
    elif proximity_52w >= 0.92:
        P.append(f"B3.52주근접{proximity_52w*100:.0f}%"); score += 8
    elif proximity_52w >= 0.85:
        P.append(f"B3.52주권내{proximity_52w*100:.0f}%"); score += 4
    else:
        F_list.append(f"B3.52주원격{proximity_52w*100:.0f}%")

    # ── 보너스 4. VCP 변동성 수축 패턴 (Minervini 핵심)
    # ATR이 50일 평균 대비 수축 + 현재 거래량 폭발
    atr_now = calc_atr(h, l, c, 10)
    atr_50  = calc_atr(h[max(0,i-50):i+1], l[max(0,i-50):i+1], c[max(0,i-50):i+1], 20) if i >= 50 else atr_now
    atr_ratio = atr_now / atr_50 if atr_50 > 0 else 1.0
    if atr_ratio <= 0.7 and vol_ratio >= 2.0:  # 수축 후 폭발 = VCP 브레이크아웃
        P.append(f"B4.VCP폭발({atr_ratio:.1f}ATR)"); score += 12
    elif atr_ratio <= 0.85:
        P.append(f"B4.VCP수축({atr_ratio:.1f}ATR)"); score += 5
    else:
        F_list.append(f"B4.VCP없음({atr_ratio:.1f}ATR)")

    # ── 보너스 5. Stochastic 골든크로스 (과매도 탈출)
    sk, sd = calc_stoch(pd.Series(h), pd.Series(l), pd.Series(c), 14, 3)
    sk_v = sk.iloc[-1]; sd_v = sd.iloc[-1]
    if not np.isnan(sk_v) and not np.isnan(sd_v):
        if sk_v > sd_v and 20 < sk_v < 80:
            P.append(f"B5.Stoch골든{sk_v:.0f}"); score += 7
        elif sk_v > sd_v:
            P.append(f"B5.Stoch상승{sk_v:.0f}"); score += 3
        else:
            F_list.append(f"B5.Stoch데드{sk_v:.0f}")
    else:
        F_list.append("B5.Stoch-")

    # ── 보너스 6. OBV 매집 (기관 누적 매수 신호)
    obv_delta = calc_obv_trend(c, v, 20)
    if obv_delta > 0:
        P.append("B6.OBV매집"); score += 6
    else:
        F_list.append("B6.OBV이탈")

    # ── 보너스 7. 볼린저밴드 위치 (BB 상단 돌파 or 상단 근접 = 강한 추세)
    bb_mid = np.mean(c[i-19:i+1])
    bb_std = np.std(c[i-19:i+1], ddof=1)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos = (c[i] - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
    if bb_pos >= 90:
        P.append(f"B7.BB상단돌파{bb_pos:.0f}%"); score += 10
    elif bb_pos >= 70:
        P.append(f"B7.BB상단{bb_pos:.0f}%"); score += 6
    elif bb_pos >= 50:
        P.append(f"B7.BB중심↑{bb_pos:.0f}%"); score += 3
    else:
        F_list.append(f"B7.BB중심↓{bb_pos:.0f}%")

    # ── 보너스 8. 캔들 강도 (장대양봉)
    rng = h[i] - l[i]
    body_ratio = body / rng if rng > 0 else 0
    if body_ratio >= 0.7:
        P.append("B8.장대양봉"); score += 5
    elif body_ratio >= 0.5:
        P.append("B8.양봉양호"); score += 3
    else:
        F_list.append("B8.약한캔들")

    # ── 보너스 9. 단기 10MA 상승 돌파 (O'Neil 추세 확인)
    if c[i] > sma10 and sma10 > sma20:
        P.append("B9.10MA돌파"); score += 5
    else:
        F_list.append("B9.10MA미달")

    # ── 보너스 10. 상대 강도 (최근 3개월 상승률 vs 시장)
    ret_3m = (c[i] - c[max(0, i-63)]) / c[max(0, i-63)] * 100 if i >= 63 else 0
    if ret_3m >= 20:
        P.append(f"B10.RS강{ret_3m:.0f}%"); score += 8
    elif ret_3m >= 10:
        P.append(f"B10.RS양{ret_3m:.0f}%"); score += 4
    else:
        F_list.append(f"B10.RS약{ret_3m:.0f}%")

    # ── 보너스 11. 한국 특화: 기관/외인 동시 매수 (최강 신호)
    # → rankScore에서 investor data로 처리, 여기선 거래량 기반 대리 지표
    if vol_ratio >= 3.0:
        P.append(f"B11.거래폭발{vol_ratio:.1f}x"); score += 8
    elif vol_ratio >= 2.0:
        P.append(f"B11.거래급증{vol_ratio:.1f}x"); score += 5
    else:
        P.append(f"B11.거래증가{vol_ratio:.1f}x"); score += 2

    vol_ratio_display = vol_ratio  # 표시용

    # 50일 평균 거래량 비율 (화면 표시용)
    av50 = np.mean(v[max(0, i-49):i]) if i >= 50 else np.mean(v[:i])
    vr50 = v[i] / av50 if av50 > 0 else vol_ratio

    atr = calc_atr(h, l, c)

    # MACD(12,26,9) 표시용
    macd_line_std, _, macd_hist_std = calc_macd(pd.Series(c), 12, 26, 9)
    mh_display = float(macd_hist_std.iloc[-1]) if not np.isnan(macd_hist_std.iloc[-1]) else 0

    return {
        "passed": P, "failed": F_list,
        "pass_count": len(P), "total": len(P) + len(F_list),
        "momentum": round(min(score, 100), 1),
        "atr": round(atr),
        "rsi": round(rv, 1),
        "macd_hist": round(mh_display, 2),
        "volume_ratio": round(vr50, 1),
        "bb_lower": round(bb_lower),
        "bb_upper": round(bb_upper),
        "bb_mid": round(bb_mid),
        "bb_pos": round(bb_pos, 1)
    }

# =============================================
#  매매가 계산 (종가매수 전략)
# =============================================
def tick(p, ref):
    for lim, t in [(2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)]:
        if ref < lim:
            return (p // t) * t
    return (p // 1000) * 1000

def calc_price_pro(cl, lo, atr, bb_info=None):
    if cl <= 0 or atr <= 0:
        return None
    # 매수가: 종가 +0.3% (익일 시초가 슬리피지, 최적화 결과 0.3%)
    buy = tick(int(cl * 1.003), cl)

    # 손절: ATR x1.5 또는 저가-1% 또는 최대-7% 중 높은 값
    sl_atr = int(buy - 1.5 * atr)
    sl_low = int(lo * 0.99)
    sl_max = int(buy * 0.93)
    sl = tick(max(sl_atr, sl_low, sl_max), cl)

    risk = buy - sl
    if risk <= 0:
        return None

    # 목표1: BB상한선 또는 2R
    t1_bb = int(bb_info["bb_upper"]) if bb_info and bb_info.get("bb_upper", 0) > buy else int(buy + 2.0 * risk)
    t1 = tick(max(t1_bb, int(buy + 1.5 * risk)), cl)

    # 목표2: 3R (추세 확장)
    t2 = tick(int(buy + 3.0 * risk), cl)

    rr = round((t1 - buy) / risk, 2) if risk > 0 else 0
    risk_pct = round((buy - sl) / buy * 100, 1)
    return {"buy": buy, "t1": t1, "t2": t2, "sl": sl,
            "rr": rr, "atr": round(atr), "risk_pct": risk_pct, "risk_won": risk}

# =============================================
#  고속 스캔 엔진
# =============================================
scan_status = {"running": False, "progress": 0, "total": 0, "found": 0, "message": "", "phase": ""}

def run_scan(date_str, demo=False):
    global scan_status
    results = []

    t_start = time.time()

    # ── Phase 1: KOSPI + KOSDAQ 병렬 조회 ──
    scan_status = {"running": True, "progress": 0, "total": 0, "found": 0,
                   "message": "KOSPI+KOSDAQ parallel fetch...", "phase": "krx"}
    print(f"\n{'='*60}")
    print(f"  [SCAN] BB Lower Bounce | {date_str}")
    print(f"{'='*60}")

    t1 = time.time()
    all_stocks = naver_all_rising_parallel()
    t_phase1 = time.time() - t1

    if not all_stocks:
        scan_status = {"running": False, "progress": 0, "total": 0, "found": 0,
                       "message": "No data", "phase": "done"}
        return []

    print(f"  [Phase1] {len(all_stocks)} stocks in {t_phase1:.1f}s")

    # ── Phase 2: 강화 필터 (후보 수 대폭 감축) ──
    scan_status["message"] = "Filtering..."
    scan_status["phase"] = "filter"
    candidates = []
    excluded_count = 0

    for s in all_stocks:
        code = s.get("itemCode", "")
        name = s.get("stockName", "")
        market = s.get("_market", "")
        end_type = s.get("stockEndType", "")

        if end_type not in ("stock", ""):
            excluded_count += 1; continue
        if is_excluded_by_name(name, code):
            excluded_count += 1; continue

        cl = parse_num(s.get("closePrice", "0"))
        vol = parse_num(s.get("accumulatedTradingVolume", "0"))
        trdval = parse_num(s.get("accumulatedTradingValue", "0"))
        mcap_eok = parse_num(s.get("marketValue", "0"))
        chg_rate = parse_float(s.get("fluctuationsRatio", "0"))

        # 기본 필터 (주도주 기준)
        if cl <= 0 or vol <= 0: continue
        if cl < 1000: continue              # E조건: 1,000원 이상
        if trdval < 1000: continue          # 거래대금 10억 이상
        if mcap_eok < 3000: continue        # 주도주: 시총 3000억 이상

        candidates.append({
            "code": code, "name": name, "market": market,
            "close": cl, "volume": vol, "trdval": trdval,
            "mcap_eok": mcap_eok, "chg_rate": chg_rate
        })

    print(f"  [Phase2] Excluded: {excluded_count} | Candidates: {len(candidates)}")
    scan_status["total"] = len(candidates)
    scan_status["message"] = f"Analyzing {len(candidates)} candidates..."
    scan_status["phase"] = "detail"

    # ── Phase 3: 20개 워커 병렬 기술 분석 ──
    reset_debug()
    t3 = time.time()
    scanned = 0

    def analyze_one(cand):
        code = cand["code"]
        df = naver_ohlcv_fast(code, 250, target_date=date_str)
        if df is None:
            return None
        # 지정 날짜까지의 데이터만 사용
        df = df[df.index <= pd.Timestamp(date_str)]
        if len(df) < 201:
            return None
        r = screen_pro(df, cand["name"], code, cand["mcap_eok"])
        if r is None:
            return None
        last_close = int(df["Close"].iloc[-1])
        last_open = int(df["Open"].iloc[-1])
        last_high = int(df["High"].iloc[-1])
        last_low = int(df["Low"].iloc[-1])
        bb_info = {"bb_lower": r.get("bb_lower", 0), "bb_mid": r.get("bb_mid", 0), "bb_upper": r.get("bb_upper", 0)}
        p = calc_price_pro(last_close, last_low, r["atr"], bb_info)
        if p is None:
            return None
        dd = df.index[-1]
        dd = dd.strftime("%Y-%m-%d") if hasattr(dd, 'strftime') else date_str

        # 외국인/기관 수급 데이터 조회
        inv = fetch_investor_data(code)
        inv_score = 0
        foreign_net = 0
        inst_net = 0
        foreign_hold = 0.0
        foreign_buy_days = 0
        inst_buy_days = 0

        per = 0; pbr = 0; eps = 0; div_yield = 0
        target_price = 0; recomm_mean = 0; sector_ratio = 50
        high_52w = 0; low_52w = 0

        if inv:
            foreign_net = inv["foreign_net_5d"]
            inst_net = inv["inst_net_5d"]
            foreign_hold = inv["foreign_hold"]
            foreign_buy_days = inv["foreign_buy_days"]
            inst_buy_days = inv["inst_buy_days"]
            per = inv["per"]; pbr = inv["pbr"]; eps = inv["eps"]
            div_yield = inv["div_yield"]
            target_price = inv["target_price"]; recomm_mean = inv["recomm_mean"]
            sector_ratio = inv["sector_ratio"]
            high_52w = inv["high_52w"]; low_52w = inv["low_52w"]

            # S6. 수급 점수 (최대 15점)
            if foreign_net > 0 and inst_net > 0:
                r["passed"].append("S6.쌍끌이매수")
                inv_score = 15
            elif inst_net > 0:
                r["passed"].append(f"S6.기관매수{inst_buy_days}일")
                inv_score = 10
            elif foreign_net > 0:
                r["passed"].append(f"S6.외인매수{foreign_buy_days}일")
                inv_score = 8
            else:
                r["failed"].append("S6.수급약세")

            # S7. 외국인 보유비율
            if foreign_hold >= 20:
                r["passed"].append(f"S7.외인{foreign_hold:.0f}%")
                inv_score += 5
            elif foreign_hold >= 10:
                r["passed"].append(f"S7.외인{foreign_hold:.0f}%")
                inv_score += 3

            # S8. 재무 건전성 (PER 적정 + PBR 저평가)
            if 0 < per <= 15:
                r["passed"].append(f"S8.PER{per:.1f}")
                inv_score += 5
            elif 0 < per <= 25:
                r["passed"].append(f"S8.PER{per:.1f}")
                inv_score += 2
            elif per > 50 or per < 0:
                r["failed"].append(f"S8.PER{per:.1f}")

            # S9. 증권사 컨센서스 (목표가 괴리율)
            if target_price > 0 and last_close > 0:
                upside = (target_price - last_close) / last_close * 100
                if upside >= 30:
                    r["passed"].append(f"S9.목표+{upside:.0f}%")
                    inv_score += 5
                elif upside >= 15:
                    r["passed"].append(f"S9.목표+{upside:.0f}%")
                    inv_score += 3
                elif upside < 0:
                    r["failed"].append(f"S9.목표초과")

            # S10. 업종 모멘텀
            if sector_ratio >= 70:
                r["passed"].append(f"S10.업종강세{sector_ratio:.0f}%")
                inv_score += 5
            elif sector_ratio >= 50:
                r["passed"].append(f"S10.업종보통")
                inv_score += 2
            else:
                r["failed"].append(f"S10.업종약세{sector_ratio:.0f}%")

        final_score = min(r["momentum"] + inv_score, 100)

        # 고급 팩터 계산 (F1~F10)
        try:
            adv = calc_advanced_factors(df)
        except Exception:
            adv = {f"f{n}_score": 0 for n in range(1, 11)}
            adv.update({"advancedTotal": 0, "pos52w": 0, "distFromHigh": 0,
                        "atrRatio": 1, "squeeze": False, "vwapDist": 0,
                        "cmf": 0, "rocAccel": 0, "compression": 1, "insideDays": 0,
                        "z20": 0, "z60": 0, "adx": 0, "rSquared": 0, "trendSlope": 0})

        return {
            "code": code, "name": cand["name"], "market": cand["market"],
            "close": last_close, "open": last_open,
            "high": last_high, "low": last_low,
            "volume": cand["volume"], "marketCap": cand["mcap_eok"],
            "buyPrice": p["buy"], "target1": p["t1"], "target2": p["t2"],
            "stoploss": p["sl"], "rrRatio": p["rr"],
            "atr": p["atr"], "riskPct": p["risk_pct"],
            "momentum": final_score,
            "conditionsMet": r["passed"],
            "conditionsDetail": f'{r["pass_count"]}/{r["total"]}',
            "rsi": r["rsi"],
            "macdHist": r["macd_hist"],
            "volumeRatio": r["volume_ratio"],
            "dataDate": dd,
            "changeRate": cand["chg_rate"],
            "foreignNet5d": foreign_net,
            "instNet5d": inst_net,
            "foreignHold": foreign_hold,
            "foreignBuyDays": foreign_buy_days,
            "instBuyDays": inst_buy_days,
            "per": per, "pbr": pbr, "eps": eps,
            "divYield": div_yield,
            "targetPrice": target_price,
            "recommMean": recomm_mean,
            "sectorRatio": sector_ratio,
            "high52w": high_52w, "low52w": low_52w,
            # 고급 팩터 (F1~F10)
            "pos52w": float(adv.get("pos52w", 0)),
            "distFromHigh": float(adv.get("distFromHigh", 0)),
            "atrRatio": float(adv.get("atrRatio", 1)),
            "squeeze": bool(adv.get("squeeze", False)),
            "vwapDist": float(adv.get("vwapDist", 0)),
            "cmf": float(adv.get("cmf", 0)),
            "rocAccel": float(adv.get("rocAccel", 0)),
            "compression": float(adv.get("compression", 1)),
            "insideDays": int(adv.get("insideDays", 0)),
            "z20": float(adv.get("z20", 0)),
            "z60": float(adv.get("z60", 0)),
            "adx": float(adv.get("adx", 0)),
            "rSquared": float(adv.get("rSquared", 0)),
            "trendSlope": float(adv.get("trendSlope", 0)),
            "advancedTotal": int(adv.get("advancedTotal", 0)),
            "f1_score": int(adv.get("f1_score", 0)),
            "f2_score": int(adv.get("f2_score", 0)),
            "f3_score": int(adv.get("f3_score", 0)),
            "f4_score": int(adv.get("f4_score", 0)),
            "f5_score": int(adv.get("f5_score", 0)),
            "f6_score": int(adv.get("f6_score", 0)),
            "f7_score": int(adv.get("f7_score", 0)),
            "f8_score": int(adv.get("f8_score", 0)),
            "f9_score": int(adv.get("f9_score", 0)),
            "f10_score": int(adv.get("f10_score", 0)),
        }

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_one, cand): cand for cand in candidates}
        for future in as_completed(futures):
            scanned += 1
            cand = futures[future]
            scan_status["progress"] = scanned
            if scanned % 10 == 0 or scanned == len(candidates):
                scan_status["message"] = f"{cand['name']}({cand['code']}) ({scanned}/{len(candidates)})"
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    scan_status["found"] = len(results)
            except:
                pass

    t_phase3 = time.time() - t3
    print(f"  [Phase3] {len(results)} found in {t_phase3:.1f}s ({scanned} analyzed)")
    print_debug_stats()

    # ── Phase 4: 경고 체크 병렬 (15개 동시) ──
    t4 = time.time()
    warning_excluded = 0
    if results:
        scan_status["message"] = f"Warning check ({len(results)})..."
        scan_status["phase"] = "warning"
        codes_to_check = [r["code"] for r in results]
        excluded_codes = check_warning_batch(codes_to_check)
        if excluded_codes:
            before = len(results)
            results = [r for r in results if r["code"] not in excluded_codes]
            warning_excluded = before - len(results)
    t_phase4 = time.time() - t4

    # 캐시 저장
    save_cache_to_disk()

    # ── 실전 랭킹: 상승확률 종합점수 (100점 만점) ──
    # 기본 8항목 (70점) + 고급팩터 10항목 (30점) = 총 100점
    for r in results:
        rank_score = 0.0

        # 1. 수급 (35점 배점) - 최고 가중치! 외국인·기관이 핵심
        fn = r.get("foreignNet5d", 0)
        ins = r.get("instNet5d", 0)
        fb = r.get("foreignBuyDays", 0)
        ib = r.get("instBuyDays", 0)
        if fn > 0 and ins > 0:
            rank_score += 35  # 쌍끌이 = 만점
        elif ins > 0:
            rank_score += 22 + min(ib * 2, 6)  # 기관 매수 + 연속일수
        elif fn > 0:
            rank_score += 15 + min(fb * 2, 6)  # 외인 매수 + 연속일수
        else:
            rank_score += 0  # 수급 없음

        # 2. 기술적 조건 충족도 (12점 배점)
        conds_str = r.get("conditionsDetail", "0/0").split("/")
        met = int(conds_str[0]) if conds_str[0].isdigit() else 0
        total = int(conds_str[1]) if len(conds_str) > 1 and conds_str[1].isdigit() else 16
        rank_score += (met / max(total, 1)) * 12

        # 3. 거래량 강도 (6점 배점)
        vr = r.get("volumeRatio", 0)
        if vr >= 3.0: rank_score += 6
        elif vr >= 2.0: rank_score += 4
        elif vr >= 1.5: rank_score += 3
        elif vr >= 1.0: rank_score += 1

        # 4. R/R 비율 (5점 배점)
        rr = r.get("rrRatio", 0)
        if rr >= 3.0: rank_score += 5
        elif rr >= 2.0: rank_score += 4
        elif rr >= 1.5: rank_score += 2
        else: rank_score += 1

        # 5. 증권사 목표가 괴리율 (5점 배점)
        tp = r.get("targetPrice", 0)
        cl = r.get("close", 1)
        if tp > 0 and cl > 0:
            upside = (tp - cl) / cl * 100
            if upside >= 40: rank_score += 5
            elif upside >= 25: rank_score += 4
            elif upside >= 15: rank_score += 2
            elif upside >= 0: rank_score += 1

        # 6. 재무 건전성 (4점 배점)
        per = r.get("per", 0)
        pbr = r.get("pbr", 0)
        if 0 < per <= 10: rank_score += 2
        elif 0 < per <= 15: rank_score += 1
        if 0 < pbr <= 1.0: rank_score += 2
        elif 0 < pbr <= 2.0: rank_score += 1

        # 7. 업종 모멘텀 (3점 배점)
        sr = r.get("sectorRatio", 50)
        if sr >= 80: rank_score += 3
        elif sr >= 60: rank_score += 2
        elif sr >= 50: rank_score += 1

        # 8. RSI 적정구간 보너스 (2점 배점) - 40~55가 최적
        rsi = r.get("rsi", 50)
        if 40 <= rsi <= 55: rank_score += 2
        elif 35 <= rsi <= 60: rank_score += 1

        # ── 9. 고급 팩터 F1~F10 (30점 배점) ──
        # F1: 52주 고가 저항 (-5~5) → 정규화 0~5
        f1 = r.get("f1_score", 0)
        rank_score += max(0, min((f1 + 5) / 2, 5))  # -5→0, 0→2.5, 5→5

        # F2: 변동성 수축 (0~8) → 정규화 0~4
        rank_score += min(r.get("f2_score", 0) / 2, 4)

        # F3: VWAP 거리 (0~6) → 정규화 0~3
        rank_score += min(r.get("f3_score", 0) / 2, 3)

        # F4: CMF 자금유입 (0~7) → 정규화 0~4
        rank_score += min(r.get("f4_score", 0) * 4 / 7, 4)

        # F5: ROC 가속도 (-3~6) → 정규화 0~3
        f5 = r.get("f5_score", 0)
        rank_score += max(0, min((f5 + 3) / 3, 3))

        # F6: 가격 압축 (0~7) → 정규화 0~3
        rank_score += min(r.get("f6_score", 0) * 3 / 7, 3)

        # F7: 갭 분석 (0~6) → 정규화 0~2
        rank_score += min(r.get("f7_score", 0) / 3, 2)

        # F8: 수급 가속도 (placeholder) → 0
        rank_score += r.get("f8_score", 0)

        # F9: Z-Score (-3~7) → 정규화 0~3
        f9 = r.get("f9_score", 0)
        rank_score += max(0, min((f9 + 3) / 10 * 3, 3))

        # F10: 추세 품질 (0~6) → 정규화 0~3
        rank_score += min(r.get("f10_score", 0) / 2, 3)

        r["rankScore"] = round(min(rank_score, 100), 1)

    results.sort(key=lambda x: x["rankScore"], reverse=True)

    # 순위 부여
    for i, r in enumerate(results):
        r["rank"] = i + 1
    t_total = time.time() - t_start

    scan_status = {"running": False, "progress": len(candidates), "total": len(candidates),
                   "found": len(results), "message": "done", "phase": "done"}

    print(f"\n{'='*82}")
    print(f"  [RESULT] {date_str} | {len(results)} stocks | Warning excluded: {warning_excluded}")
    print(f"  Time: Phase1={t_phase1:.1f}s Phase3={t_phase3:.1f}s Phase4={t_phase4:.1f}s TOTAL={t_total:.1f}s")
    print(f"{'='*82}")
    if results:
        for idx, s in enumerate(results[:20], 1):
            print(f"  {idx:>2} {s['name']:<12} {s['close']:>9,} {s['market']:<6} Score:{s['momentum']:>5.1f}")
    print(f"{'='*82}\n")
    return results

# =============================================
#  Flask Routes
# =============================================
@app.route("/")
def index():
    return render_template("index.html")

def _save_and_sync_results(results, date_str):
    """스캔 결과를 JSON 파일로 저장 + GitHub push (Render 동기화)"""
    import subprocess
    dow = datetime.strptime(date_str, "%Y-%m-%d").weekday()
    dow_names = ["월","화","수","목","금","토","일"]
    payload = {
        "results": results,
        "date": date_str,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "KRX+Naver",
        "count": len(results),
        "dayOfWeek": dow_names[dow],
        "isTuesday": dow == 1,
        "tradeRules": {
            "slip": "0.3%", "t1SellRatio": "1/2", "t2SellRatio": "1/2",
            "trailingATR": 1.5, "maxHold": 20,
            "skipTuesday": True, "skip3Loss": True,
        }
    }
    # JSON 저장
    results_path = os.path.join(os.path.dirname(__file__), "latest_results.json")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"  [SYNC] Saved latest_results.json ({len(results)} stocks)")
    except Exception as e:
        print(f"  [SYNC] Save error: {e}")
        return payload

    # Git push (로컬에서만, Render에서는 스킵)
    is_render = os.environ.get("RENDER", "") == "true"
    if not is_render:
        try:
            app_dir = os.path.dirname(__file__)
            subprocess.Popen(
                ["git", "add", "latest_results.json"],
                cwd=app_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).wait(timeout=5)
            subprocess.Popen(
                ["git", "commit", "-m", f"sync: {date_str} scan results ({len(results)} stocks)"],
                cwd=app_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).wait(timeout=5)
            subprocess.Popen(
                ["git", "push", "origin", "master"],
                cwd=app_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"  [SYNC] Git push started")
        except Exception as e:
            print(f"  [SYNC] Git push skipped: {e}")

    return payload

def _load_cached_results(date_str):
    """저장된 결과 파일 로드 (Render에서 사용)"""
    results_path = os.path.join(os.path.dirname(__file__), "latest_results.json")
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == date_str:
            print(f"  [CACHE] Serving cached results for {date_str} ({data.get('count',0)} stocks)")
            return data
    except Exception:
        pass
    return None

@app.route("/api/scan")
def api_scan():
    date_str = flask_request.args.get("date", "")
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        tgt = datetime.strptime(date_str, "%Y-%m-%d")
        if tgt > datetime.now():
            date_str = datetime.now().strftime("%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Date format error"}), 400

    # 1) 캐시된 결과가 있으면 우선 사용 (Render에서 동일 결과 보장)
    cached = _load_cached_results(date_str)
    if cached:
        return jsonify(cached)

    # 2) 없으면 라이브 스캔
    results = run_scan(date_str)

    # 3) 결과 저장 + GitHub 동기화
    payload = _save_and_sync_results(results, date_str)
    return jsonify(payload)

@app.route("/api/status")
def api_status():
    return jsonify(scan_status)

# =============================================
#  미국 주식 라우트
# =============================================
@app.route("/us")
def index_us():
    return render_template("index_us.html")

@app.route("/api/us/scan")
def api_us_scan():
    from us_screener import run_us_scan
    date_str = flask_request.args.get("date", "")
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        tgt = datetime.strptime(date_str, "%Y-%m-%d")
        if tgt > datetime.now():
            date_str = datetime.now().strftime("%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Date format error"}), 400

    results = run_us_scan(date_str)
    return jsonify({
        "results": results,
        "date": date_str,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(results)
    })

@app.route("/api/us/status")
def api_us_status():
    from us_screener import us_scan_status
    return jsonify(us_scan_status)

if __name__ == "__main__":
    import os as _os
    _port = int(_os.environ.get("PORT", 5000))
    print("\n" + "=" * 50)
    print(f"  Pro Stock Screener (High-Speed)")
    print(f"  http://localhost:{_port}")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=_port, debug=False)
