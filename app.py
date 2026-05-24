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
    # 코일스프링 패턴 탈락 카운터
    "qual_mcap": 0, "qual_avgtv": 0, "qual_price": 0, "qual_listing": 0,
    "p1_total": 0,           # Phase1 점수 부족 (3점 미만)
    "p2_volume": 0,          # 폭발 트리거 거래량 미달
    "p2_price": 0,           # 폭발 트리거 가격 미달
    "p2_trend": 0,           # 폭발 트리거 추세 미달 (MA20)
    "grade_none": 0,         # 등급 산출 불가
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
#  섹터 분류 (미래 주도주 가중치)
# =============================================
SECTOR_CORE_CODES = {
    # AI반도체 (+30)
    "AI반도체": {"000660","042700","058470","039030","319660","240810","036930","357780","089030"},
    # 로봇 (+30)
    "로봇": {"277810","454910","090360","056080","117730","058610"},
    # 전력인프라 (+30)
    "전력인프라": {"010120","267260","298040","001440","000500","011690","082920"},
    # 방산 (+25)
    "방산": {"012450","047810","079550","064350","103140","073570"},
    # AI소프트웨어 (+25)
    "AI소프트웨어": {"035420","304100","388790","402030"},
    # 우주항공 (+25)
    "우주항공": {"099320","189300","357550"},
}

SECTOR_KEYWORDS = [
    # 순서가 우선순위 - 위에서 매칭되면 종료
    ("AI반도체", 30, ["반도체","메모리","HBM","파운드리","팹리스"]),
    ("로봇", 30, ["로보","로봇","자동화"]),
    ("전력인프라", 30, ["전력","변압기","전선","케이블","발전"]),
    ("방산", 25, ["방산","무기","탄약"]),
    ("AI소프트웨어", 25, ["AI","인공지능","소프트웨어","플랫폼"]),
    ("우주항공", 25, ["우주","위성","항공"]),
    ("2차전지", 15, ["배터리","양극재","음극재","전해질","2차전지"]),
    ("바이오", 15, ["바이오","제약","신약"]),
    ("양자컴퓨팅", 15, ["양자"]),
    ("사이버보안", 15, ["보안","사이버"]),
    ("신재생에너지", 10, ["태양광","풍력","신재생"]),
    ("수소", 10, ["수소"]),
    ("의료기기", 10, ["의료기기","진단"]),
]

SECTOR_BONUS = {
    "AI반도체": 30, "로봇": 30, "전력인프라": 30,
    "방산": 25, "AI소프트웨어": 25, "우주항공": 25,
    "2차전지": 15, "바이오": 15, "양자컴퓨팅": 15, "사이버보안": 15,
    "신재생에너지": 10, "수소": 10, "의료기기": 10,
}

def classify_sector(name, code):
    """종목코드/이름으로 섹터 분류. 핵심 종목 리스트 우선, 그다음 키워드 매칭."""
    # 1) 핵심 종목 강제 분류
    for sec, codes in SECTOR_CORE_CODES.items():
        if code in codes:
            return sec, SECTOR_BONUS.get(sec, 0)
    # 2) 키워드 매칭
    nu = (name or "").upper()
    for sec, bonus, kws in SECTOR_KEYWORDS:
        for kw in kws:
            if kw.upper() in nu:
                return sec, bonus
    return "일반", 0

# =============================================
#  코일스프링 스크리닝 (압축 → 폭발 패턴)
#  Phase1: 압축 감지 (10점) | Phase2: 폭발 트리거 (당일)
#  보조: Minervini Trend Template (가산점)
# =============================================
def screen_pro(df, name="", code="", mcap=0):
    """
    ★ 코일스프링 패턴 스크리너
    - Phase1: 거래량/변동성 압축 감지 (10점 만점)
    - Phase2: 당일 폭발 트리거 (거래량 2x + 가격 +3% + MA20 위)
    - Minervini Trend Template (보조 가산점 5개)
    - 등급: BUY(P1≥6 & P2충족) / WATCH(P1≥4) / COILING(P1≥3)
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

    # ── 데이터 품질 필터 ──
    # 신규상장 60일 미만 제외
    if len(df) < 60:
        with _debug_lock: _debug_reject["qual_listing"] += 1
        return None
    # 현재가 1,000원 이상
    if c[i] < 1000:
        with _debug_lock: _debug_reject["qual_price"] += 1
        return None
    # 시가총액 300억 이상 (mcap 단위: 억)
    if mcap > 0 and mcap < 300:
        with _debug_lock: _debug_reject["qual_mcap"] += 1
        return None
    # 일평균 거래대금(20일) 3억 이상
    avg_trdval_20 = np.mean(c[i-19:i+1] * v[i-19:i+1]) / 1e8  # 단위: 억
    if avg_trdval_20 < 3:
        with _debug_lock: _debug_reject["qual_avgtv"] += 1
        return None
    # 상한가 제외
    chg_today = (c[i] - c[i-1]) / c[i-1] * 100 if c[i-1] > 0 else 0
    if chg_today >= 29:
        with _debug_lock: _debug_reject["limit"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 1 — 압축 감지 (10점 만점)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    p1_score = 0
    p1_details = []

    # ① 거래량 추세 감소 (3점)
    # 최근10일 거래량평균 / 이전10일 거래량평균 < 0.75
    # 당일 폭발 거래량을 제외하기 위해 [i-10:i] vs [i-20:i-10]
    vol_recent10 = np.mean(v[i-10:i]) if i >= 10 else 1
    vol_prev10   = np.mean(v[i-20:i-10]) if i >= 20 else 1
    vol_trend_ratio = vol_recent10 / vol_prev10 if vol_prev10 > 0 else 1.0
    if vol_trend_ratio < 0.75:
        p1_score += 3
        p1_details.append(f"①거래량감소{vol_trend_ratio:.2f}")

    # ② 거래량 절대 수준 (2점)
    # 최근10일 거래량평균 / 60일 거래량평균 < 0.65
    vol_60 = np.mean(v[i-60:i]) if i >= 60 else vol_recent10
    vol_abs_ratio = vol_recent10 / vol_60 if vol_60 > 0 else 1.0
    if vol_abs_ratio < 0.65:
        p1_score += 2
        p1_details.append(f"②거래량저수준{vol_abs_ratio:.2f}")

    # ③ 변동성 극소화 (2점)
    # 20일 일간수익률 연환산 표준편차 < 35%
    daily_ret = np.diff(c[i-20:i+1]) / c[i-20:i]
    annual_vol = np.std(daily_ret, ddof=1) * np.sqrt(252) * 100 if len(daily_ret) > 1 else 100
    if annual_vol < 35:
        p1_score += 2
        p1_details.append(f"③변동성{annual_vol:.0f}%")

    # ④ 횡보 박스권 (2점)
    # |20일 가격변화율| < 8% AND 20일 (고가-저가)/저가 < 15%
    price_chg_20 = (c[i] - c[i-20]) / c[i-20] * 100 if i >= 20 and c[i-20] > 0 else 0
    high_20 = np.max(h[i-19:i+1])
    low_20  = np.min(l[i-19:i+1])
    box_range = (high_20 - low_20) / low_20 * 100 if low_20 > 0 else 100
    if abs(price_chg_20) < 8 and box_range < 15:
        p1_score += 2
        p1_details.append(f"④박스권{box_range:.0f}%")

    # ⑤ 볼린저밴드 수축 (1점) BB폭(20,2σ) < 18%
    bb_mid = np.mean(c[i-19:i+1])
    bb_std = np.std(c[i-19:i+1], ddof=1)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width_pct = (bb_upper - bb_lower) / bb_mid * 100 if bb_mid > 0 else 100
    if bb_width_pct < 18:
        p1_score += 1
        p1_details.append(f"⑤BB폭{bb_width_pct:.1f}%")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 2 — 폭발 트리거 (당일)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # A. 거래량: 당일 거래량 ≥ 20일평균 × 2.0배
    avg_vol_20 = np.mean(v[i-20:i]) if i >= 20 else np.mean(v[:i])
    vol_mult = v[i] / avg_vol_20 if avg_vol_20 > 0 else 1.0
    trig_volume = bool(vol_mult >= 2.0)

    # B. 가격: 당일 종가 ≥ 전일 종가 × 1.03
    trig_price = bool(c[i] >= c[i-1] * 1.03)

    # C. 추세: 당일 종가 > MA20 (필수) AND 종가 > MA60 (가산)
    sma20 = np.mean(c[i-19:i+1])
    sma60 = np.mean(c[i-59:i+1]) if i >= 59 else sma20
    trig_ma20 = bool(c[i] > sma20)
    trig_ma60 = bool(c[i] > sma60)

    phase2_pass = trig_volume and trig_price and trig_ma20

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Minervini Trend Template (보조 가산점)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    sma50  = np.mean(c[i-49:i+1])
    sma200 = np.mean(c[i-199:i+1])
    sma200_30d_ago = np.mean(c[i-229:i-29]) if i >= 229 else sma200
    high_52w = np.max(h[max(0, i-251):i+1])
    low_52w  = np.min(l[max(0, i-251):i+1])

    mn_count = 0
    mn_passed = []
    # T1. 현재가 > MA200
    if c[i] > sma200:
        mn_count += 1; mn_passed.append("T1.>MA200")
    # T2. MA200 우상향 (30일 전보다 높음)
    if sma200 > sma200_30d_ago:
        mn_count += 1; mn_passed.append("T2.MA200↑")
    # T3. MA50 > MA200
    if sma50 > sma200:
        mn_count += 1; mn_passed.append("T3.MA50>MA200")
    # T4. 현재가 ≥ 52주 최저가 × 1.25
    if low_52w > 0 and c[i] >= low_52w * 1.25:
        mn_count += 1; mn_passed.append("T4.저점+25%")
    # T5. 현재가 ≥ 52주 최고가 × 0.65
    if high_52w > 0 and c[i] >= high_52w * 0.65:
        mn_count += 1; mn_passed.append("T5.고점-35%이내")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 등급 판정
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    grade = None
    if p1_score >= 6 and phase2_pass:
        grade = "BUY"
    elif p1_score >= 4 and not phase2_pass:
        grade = "WATCH"
    elif p1_score >= 3:
        grade = "COILING"
    else:
        with _debug_lock: _debug_reject["p1_total"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 섹터 가중치
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    sector, sector_bonus = classify_sector(name, code)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 최종점수 = (Phase1점수 × 10) + 섹터가중치_보너스 + Minervini충족수
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    final_score = (p1_score * 10) + sector_bonus + mn_count

    # ── 보조 표시지표 계산 ──
    rv = calc_rsi(pd.Series(c), 14).iloc[-1]
    rv = float(rv) if not np.isnan(rv) else 50.0

    _, _, macd_hist = calc_macd(pd.Series(c), 12, 26, 9)
    mh_now = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0

    atr = calc_atr(h, l, c)

    bb_pos = (c[i] - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50

    trading_value_eok = c[i] * v[i] / 1e8  # 당일 거래대금(억)

    # 통과 조건 리스트 (UI 표시용)
    P = list(p1_details)
    if trig_volume: P.append(f"P2.거래량{vol_mult:.1f}x")
    if trig_price:  P.append(f"P2.가격+{(c[i]/c[i-1]-1)*100:.1f}%")
    if trig_ma20:   P.append("P2.MA20위")
    if trig_ma60:   P.append("P2.MA60위")
    P.extend(mn_passed)
    if sector_bonus > 0:
        P.append(f"섹터[{sector}+{sector_bonus}]")

    F_list = []
    if not trig_volume: F_list.append(f"P2.거래량부족{vol_mult:.1f}x")
    if not trig_price:  F_list.append(f"P2.가격{(c[i]/c[i-1]-1)*100:+.1f}%")
    if not trig_ma20:   F_list.append("P2.MA20아래")

    return {
        "passed": P, "failed": F_list,
        "pass_count": len(P), "total": len(P) + len(F_list),
        "momentum": round(min(final_score, 100), 1),  # UI 호환용
        "atr": round(atr),
        "rsi": round(rv, 1),
        "macd_hist": round(mh_now, 2),
        "volume_ratio": round(vol_mult, 1),
        "bb_lower": round(bb_lower),
        "bb_upper": round(bb_upper),
        "bb_mid": round(bb_mid),
        "bb_pos": round(bb_pos, 1),

        # ── 코일스프링 전용 필드 ──
        "grade": grade,                          # BUY/WATCH/COILING
        "p1_score": p1_score,                    # Phase1 점수 (0~10)
        "p1_details": p1_details,                # 점수 획득 사유
        "phase2_pass": bool(phase2_pass),        # 폭발 트리거 충족 여부
        "trigVolume": trig_volume,
        "trigPrice": trig_price,
        "trigMA20": trig_ma20,
        "trigMA60": trig_ma60,
        "volMult": round(vol_mult, 2),           # 거래량 배수
        "volTrendRatio": round(vol_trend_ratio, 2),  # 거래량 추세 비율
        "annualVol": round(annual_vol, 1),       # 연환산 변동성(%)
        "bbWidthPct": round(bb_width_pct, 1),    # BB폭(%)
        "boxRange": round(box_range, 1),         # 20일 박스 범위(%)
        "priceChg20": round(price_chg_20, 1),    # 20일 가격변화율(%)
        "chgToday": round(chg_today, 1),         # 당일 변화율(%)
        "tradingValueEok": round(trading_value_eok, 1),  # 당일 거래대금(억)
        "minerviniCount": mn_count,              # Minervini 충족 개수 (0~5)
        "sector": sector,                        # 분류된 섹터
        "sectorBonus": sector_bonus,             # 섹터 가산점
        "finalScore": round(final_score, 1),    # 최종 점수
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

        # 데이터 품질 필터 (코일스프링 사전 제외)
        if cl <= 0 or vol <= 0: continue
        if cl < 1000: continue              # 현재가 1,000원 이상
        if trdval < 300: continue           # 거래대금 3억 이상 (일평균은 Phase3에서 정밀 체크)
        if mcap_eok < 300: continue         # 시가총액 300억 이상

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
            # ── 코일스프링 전용 필드 ──
            "grade": r.get("grade", "COILING"),
            "p1Score": r.get("p1_score", 0),
            "p1Details": r.get("p1_details", []),
            "phase2Pass": r.get("phase2_pass", False),
            "trigVolume": r.get("trigVolume", False),
            "trigPrice": r.get("trigPrice", False),
            "trigMA20": r.get("trigMA20", False),
            "trigMA60": r.get("trigMA60", False),
            "volMult": r.get("volMult", 0),
            "volTrendRatio": r.get("volTrendRatio", 0),
            "annualVol": r.get("annualVol", 0),
            "bbWidthPct": r.get("bbWidthPct", 0),
            "boxRange": r.get("boxRange", 0),
            "priceChg20": r.get("priceChg20", 0),
            "chgToday": r.get("chgToday", 0),
            "tradingValueEok": r.get("tradingValueEok", 0),
            "minerviniCount": r.get("minerviniCount", 0),
            "sector": r.get("sector", "일반"),
            "sectorBonus": r.get("sectorBonus", 0),
            "finalScore": r.get("finalScore", 0),
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

    # ── 코일스프링 랭킹: finalScore (Phase1×10 + 섹터보너스 + Minervini) ──
    # rankScore = finalScore (UI 호환 유지)
    for r in results:
        r["rankScore"] = r.get("finalScore", 0)

    # 등급 우선순위(BUY=0, WATCH=1, COILING=2) → finalScore 내림차순
    GRADE_ORDER = {"BUY": 0, "WATCH": 1, "COILING": 2}
    results.sort(key=lambda x: (GRADE_ORDER.get(x.get("grade", "COILING"), 3),
                                -x.get("finalScore", 0)))

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
        # 등급별 개수
        buy_n   = sum(1 for s in results if s.get("grade") == "BUY")
        watch_n = sum(1 for s in results if s.get("grade") == "WATCH")
        coil_n  = sum(1 for s in results if s.get("grade") == "COILING")
        print(f"  Grade: 🔴BUY={buy_n}  🟡WATCH={watch_n}  🔵COILING={coil_n}")
        print(f"  {'-'*78}")
        for idx, s in enumerate(results[:20], 1):
            g = s.get("grade", "-")
            sec = s.get("sector", "-")
            fs = s.get("finalScore", 0)
            p1 = s.get("p1Score", 0)
            print(f"  {idx:>2} [{g:<7}] {s['name']:<10} {s['close']:>8,} P1={p1}/10 Sec={sec:<8} Final={fs:>5.0f}")
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
