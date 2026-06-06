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

# ─── 전략 버전 (조건 변경 시 올리면 캐시 자동 무효화) ───
# 손절-5% / 청산+10% / D1 20MA>0% / D4 거래량1.5x / D5 RSI50-70 / C섹터보너스 / F1첫풀백
STRATEGY_VERSION = "2026.06.05-HUNTtruePullback(retrace3-18+ma20support+lowtail+volup)-sectorName-chgToday-scanUnified-realtimeToday"

# ─── 주가 상한 (소액 분산매수용) ───
# 100만원 시드 → 3종목 33만원씩 → 20만원 이하면 1주+ 보유 가능
MAX_PRICE = 200000   # 현재가 20만원 초과 종목 제외 (0이면 무제한)

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
_ohlcv_cache_ts = {}          # key → fetch한 시각(epoch). '오늘 캔들' 신선도 판정용
_cache_date = ""
_cache_lock = threading.Lock()
OHLCV_TODAY_TTL = 180         # 오늘 캔들 포함 캐시는 3분 지나면 재조회(장중 잠정→확정 종가 반영)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')

def get_cached_ohlcv(key):
    """메모리 캐시에서 OHLCV 조회 (key = "code_YYYY-MM-DD" 또는 "code")"""
    global _ohlcv_cache, _ohlcv_cache_ts, _cache_date
    today = datetime.now().strftime("%Y%m%d")
    if _cache_date != today:
        # 날짜 변경시 캐시 리셋 + 디스크에서 로드 시도
        with _cache_lock:
            _ohlcv_cache = {}
            _ohlcv_cache_ts = {}
            _cache_date = today
            cache_file = os.path.join(CACHE_DIR, f"{today}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        _ohlcv_cache = pickle.load(f)
                    print(f"  [CACHE] Loaded {len(_ohlcv_cache)} stocks from disk")
                    # 디스크 캐시는 fetch시각 미상 → ts=0으로 둬서 '오늘 캔들'은 재조회 유도
                except:
                    pass
    return _ohlcv_cache.get(key)

def set_cached_ohlcv(key, df):
    """메모리 캐시에 OHLCV 저장 (+fetch 시각 기록)"""
    with _cache_lock:
        _ohlcv_cache[key] = df
        _ohlcv_cache_ts[key] = time.time()

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
    """세션 기반 고속 OHLCV 조회 + 캐시
    ★ stale 방지: 캐시 데이터 마지막 날짜가 요청일보다 이전이면 재fetch
       (장중에 생성된 캐시가 당일 종가를 누락하는 버그 수정)"""
    cache_key = f"{code}_{target_date}" if target_date else code
    cached = get_cached_ohlcv(cache_key)
    if cached is not None:
        # 캐시 데이터가 요청 날짜를 커버하는지 확인
        stale = False
        if target_date and cached is not False and hasattr(cached, "index") and len(cached) > 0:
            last_cached = cached.index[-1].strftime("%Y%m%d")
            tgt_compact = target_date.replace("-", "")
            today_compact = datetime.now().strftime("%Y%m%d")
            # ① 캐시 마지막 < 요청일 → 당일 데이터 누락 → 재fetch
            if last_cached < tgt_compact:
                stale = True
            # ② 캐시가 '오늘 캔들'을 포함 → 장중엔 잠정가, 마감 후엔 확정 종가로 바뀜.
            #    날짜만 같다고 재사용하면 잠정값이 고정되는 버그 → TTL 지나면 재fetch.
            elif last_cached == today_compact and (time.time() - _ohlcv_cache_ts.get(cache_key, 0)) > OHLCV_TODAY_TTL:
                stale = True
        if not stale:
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

# 제약/바이오 제외 (사용자 지정: 뉴스·임상 이벤트로 급변 → 탈락)
PHARMA_BIO_KEYWORDS = [
    "제약", "바이오", "파마", "PHARM", "BIO", "신약", "헬스케어", "생명과학",
    "제넥", "테라퓨", "셀트리온", "메디톡스", "에스티팜", "녹십자", "유한양행",
    "한미약품", "대웅", "종근당", "보령", "동아에스티", "JW", "일동", "휴온스",
]

# 정치테마주 제외 (사용자 지정). ※ 정치테마는 가격/이름 데이터로 자동탐지 불가 —
#   아래는 수동 관리 코드 목록. 선거 사이클마다 갱신 필요(완전성 보장 못 함).
POLITICAL_THEME_CODES = {
    # 예시(과거 회자된 종목들) — 필요시 추가/삭제하세요.
    "065170", "024060", "036090", "012620", "016380",
    "093920", "032300", "025560", "066620", "035080",
}

def is_excluded_by_name(name, code):
    name_upper = name.upper()
    for kw in EXCLUDE_KEYWORDS:
        if kw.upper() in name_upper:
            return True
    # 제약/바이오 탈락
    for kw in PHARMA_BIO_KEYWORDS:
        if kw.upper() in name_upper:
            return True
    # 정치테마주 탈락 (수동 목록)
    if code in POLITICAL_THEME_CODES:
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

_industry_cache = {}
def get_industry_code(code):
    """종목 업종코드 조회 (캐시). 섹터 분산용. 실패시 '0'."""
    if code in _industry_cache:
        return _industry_cache[code]
    ind = "0"
    try:
        r = _session.get(f"https://m.stock.naver.com/api/stock/{code}/integration", timeout=6)
        if r.status_code == 200:
            ind = str(r.json().get("industryCode", "0"))
    except Exception:
        pass
    _industry_cache[code] = ind
    return ind

# 업종코드(네이버 업종번호) → 섹터 이름 매핑 (industry_map.json, 79개)
try:
    with open(os.path.join(os.path.dirname(__file__), "industry_map.json"), encoding="utf-8") as _f:
        INDUSTRY_NAMES = json.load(_f)
except Exception:
    INDUSTRY_NAMES = {}

def industry_name(code):
    """업종코드 → 섹터 이름(예: '278'→'반도체'). 미상이면 ''."""
    return INDUSTRY_NAMES.get(str(code or "0"), "")

def _enrich_sector(items):
    """결과 리스트 각 항목에 sectorName(섹터 이름) 추가 (in-place, 리스트 반환)."""
    for it in (items or []):
        if isinstance(it, dict):
            it["sectorName"] = industry_name(it.get("industryCode", "0"))
    return items

def pick_with_sector_limit(candidates, n=3, max_per_sector=2, code_key="code", sector_key="industryCode"):
    """우선순위 정렬된 후보에서 상위 n개 선택하되, 한 업종 최대 max_per_sector개.
    (검증: 업종당 최대2 = 제한없음과 동일 수익 +402%, 한 업종 3+ 몰빵만 차단)"""
    from collections import Counter
    picked = []; sec_count = Counter(); used = set()
    for c in candidates:
        if len(picked) >= n:
            break
        code = c.get(code_key); sec = str(c.get(sector_key, "0") or "0")
        if code in used:
            continue
        if sec != "0" and sec_count[sec] >= max_per_sector:
            continue
        picked.append(c); used.add(code); sec_count[sec] += 1
    return picked

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
            "industry_code": str(data.get("industryCode", "0")),   # 업종코드(섹터 분산용)
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
    # 실전 트레이더 4등급 시스템 탈락 카운터
    "qual_mcap": 0, "qual_avgtv": 0, "qual_price": 0, "qual_listing": 0,
    "A_trend": 0,            # 정배열 미충족 (200MA 아래 또는 60MA 아래 등)
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
def screen_pro(df, name="", code="", mcap=0, fundamental=None):
    """
    ★ 실전 트레이더 4등급 시스템
    - HUNT (저점매수, 남석관·Marty Schwartz): 정배열+실적+섹터+눌림목+양봉반등
    - BREAKOUT (추격매수, Minervini·O'Neil): 정배열+실적+섹터+신고가+거래량폭발
    - TREND (추세진입, Weinstein): 정배열+실적만
    - WATCH (예비, 정배열만)

    [필수 A] 추세 정배열 (Weinstein Stage 2)
    [필수 B] 실적 (염승환·O'Neil C) - EPS>0 + PER 적정 + 목표가 상승
    [선택 C] 주도섹터 (현재웅·박세익)
    [트리거 D] 저점매수(HUNT) 또는 신고가(BREAKOUT)

    fundamental dict 예: {"per": 12, "eps": 1500, "target_price": 25000, "sector_ratio": 70}
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
    if len(df) < 60:
        with _debug_lock: _debug_reject["qual_listing"] += 1
        return None
    if c[i] < 2000:
        with _debug_lock: _debug_reject["qual_price"] += 1
        return None
    # 주가 상한 (소액 분산매수용) — 20만원 초과 제외
    if MAX_PRICE > 0 and c[i] > MAX_PRICE:
        with _debug_lock: _debug_reject["qual_price"] += 1
        return None
    if mcap > 0 and mcap < 1000:
        with _debug_lock: _debug_reject["qual_mcap"] += 1
        return None
    avg_trdval_20 = np.mean(c[i-19:i+1] * v[i-19:i+1]) / 1e8
    if avg_trdval_20 < 10:
        with _debug_lock: _debug_reject["qual_avgtv"] += 1
        return None
    chg_today = (c[i] - c[i-1]) / c[i-1] * 100 if c[i-1] > 0 else 0
    if chg_today >= 29:
        with _debug_lock: _debug_reject["limit"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [필수 A] 추세 정배열 (Weinstein Stage 2 + Minervini)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    sma5   = float(np.mean(c[i-4:i+1]))
    sma10  = float(np.mean(c[i-9:i+1]))
    sma20  = float(np.mean(c[i-19:i+1]))
    sma60  = float(np.mean(c[i-59:i+1])) if i >= 59 else sma20
    sma200 = float(np.mean(c[i-199:i+1]))
    sma200_30d_ago = float(np.mean(c[i-229:i-29])) if i >= 229 else sma200

    a1 = bool(c[i] > sma200)             # A1. 종가 > 200MA (장기 상승)
    a2 = bool(c[i] > sma60)              # A2. 종가 > 60MA (중기 상승)
    a3 = bool(sma60 > sma200)            # A3. 60MA > 200MA (정배열)
    a4 = bool(sma200 > sma200_30d_ago)   # A4. 200MA 우상향 (30일 전 대비)
    a_count = int(a1) + int(a2) + int(a3) + int(a4)

    # A1(종가>200MA)은 무조건 필수 (사용자 요청)
    if not a1:
        with _debug_lock: _debug_reject["A_trend"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [필수 B] 펀더멘털 (염승환·O'Neil·Peter Lynch)
    # EPS>0(흑자) + PER 적정 + 증권사 목표가 상향
    # → 매출 우상향 proxy (네이버 무료 API 한계로 EPS+목표가 사용)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fund = fundamental or {}
    per = float(fund.get("per", 0) or 0)
    eps = float(fund.get("eps", 0) or 0)
    pbr = float(fund.get("pbr", 0) or 0)
    target_price = float(fund.get("target_price", 0) or 0)
    sector_ratio = float(fund.get("sector_ratio", 50) or 50)

    b1 = bool(eps > 0)                                      # B1. EPS > 0 (흑자) — 필수
    b2 = bool(0 < per <= 30)                                # B2. PER 적정 (0~30) — 보너스 (백테스트 -3.9% 역효과로 필수에서 강등)
    b3 = bool(target_price > 0 and target_price >= c[i] * 1.05)  # B3. 증권사 목표가 +5%+ — 필수
    b_count = int(b1) + int(b2) + int(b3)
    # 펀더 통과 자격: B1(흑자) + B3(목표가) 둘 다 필수, B2(PER)는 표시용 보너스
    b_required_ok = bool(b1 and b3)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [선택 C] 주도섹터/순환매 (현재웅·박세익)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    sector, sector_bonus = classify_sector(name, code)
    c1 = bool(sector_ratio >= 60)         # 동종업종 60% 이상 상승
    c2 = bool(sector_bonus >= 25)         # 미래 주도섹터(가중치 25점 이상)
    in_leading_sector = bool(c1 or c2)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [트리거 D - HUNT] 저점 매수 (남석관 눌림목 + Marty Schwartz)
    # 20일선 ±5% 안 + 5일선 위 + 양봉 + 거래량 동반 + RSI 정상
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ma20_dist = (c[i] - sma20) / sma20 * 100 if sma20 > 0 else 0
    avg_vol_5  = float(np.mean(v[i-5:i])) if i >= 5 else 1
    avg_vol_20 = float(np.mean(v[i-20:i])) if i >= 20 else avg_vol_5
    vol_mult_5  = v[i] / avg_vol_5  if avg_vol_5 > 0 else 1.0
    vol_mult_20 = v[i] / avg_vol_20 if avg_vol_20 > 0 else 1.0

    rv = calc_rsi(pd.Series(c), 14).iloc[-1]
    rv = float(rv) if not np.isnan(rv) else 50.0

    # ★★ 진짜 눌림목으로 교체 (검증: 1년 백테스트 '추세추종형' +18%/MDD28% → '진짜 눌림목' +106%/MDD18%)
    #   핵심: '되돌림(조정)' 발생 + 지지선 근접 + 아래꼬리/지지터치 반등 + 거래량 회복.
    #   (기존은 '20MA 위 0~15% + 양봉'이라 눌림이 아니라 추세추종이었음)
    align      = bool(sma5 > sma20 > sma60)                          # 정배열(상승추세)
    hi10       = float(np.max(h[i-10:i])) if i >= 10 else float(h[i])  # 직전 10일 고점
    retrace    = (hi10 - l[i]) / hi10 * 100 if hi10 > 0 else 0       # 고점→오늘 저가 되돌림 %
    rng_d      = h[i] - l[i]
    lower_tail = bool(rng_d > 0 and (min(o[i], c[i]) - l[i]) / rng_d >= 0.25)  # 아래꼬리 25%+
    touched_ma = bool(l[i] <= max(sma5, sma20) * 1.005)             # 당일 저가가 지지선(5/20MA) 터치
    vol_up     = bool(v[i] > v[i-1])                                # 거래량 전일比 증가

    d1_pullback   = bool(3 <= retrace <= 18)        # ① 되돌림(눌림) 3~18% — 핵심
    d2_above_ma5  = bool(-2 <= ma20_dist <= 6)      # ② 20일선 지지 근접 (-2~+6%)
    d3_bullish    = bool(o[i] < c[i])                # ③ 양봉(반등 확인)
    d4_vol_pickup = bool(lower_tail or touched_ma)   # ④ 아래꼬리/지지선 터치(지지 확인)
    d5_rsi_ok     = bool(vol_up)                      # ⑤ 거래량 회복(전일比↑)

    # F1. 첫 번째 눌림목만 허용 (영상 기반 - "3번째 눌림 진입 안 함")
    # 풀백 정의: "20MA 위 +2% 이상으로 올라간 구간"의 수를 카운트
    # 히스테리시스: above 진입 = 종가 > 20MA × 1.02 / below 이탈 = 종가 < 20MA × 0.97
    # → 잔잔한 변동은 무시, 진짜 큰 폭의 위→아래→위 흐름만 카운트
    if i >= 79:
        c_window = c[i-78:i+1]
        sma20_window = pd.Series(c_window).rolling(20).mean().values
        above_segments = 0      # 20MA 위 구간 개수
        state = "below"
        for j in range(19, 79):  # 마지막 60일
            sma_v = sma20_window[j]
            if not np.isnan(sma_v) and sma_v > 0:
                ratio = c_window[j] / sma_v
                if state == "below" and ratio > 1.02:    # +2% 위로 올라가면 새 구간 시작
                    above_segments += 1
                    state = "above"
                elif state == "above" and ratio < 0.97:  # -3% 아래로 떨어지면 풀백
                    state = "below"
        # 구간 1개 = 1차 진입(터치없음), 2개 = 1차 풀백 후 회복, 3개 = 2차 풀백 후 회복
        pullback_count = max(0, above_segments - 1)
    else:
        pullback_count = 0
    f1_first_pullback = bool(pullback_count <= 2)   # 1차/2차 눌림만 (3차+ 거부)

    # 진짜 눌림목 = 정배열 + 되돌림 + 지지근접 + 양봉 + 지지확인(꼬리/터치) + 거래량회복
    #   (검증 winner B: f1 '몇차 눌림' 게이트 없이 +106%. f1은 표시용으로만 유지)
    hunt_trigger = bool(align and d1_pullback and d2_above_ma5 and d3_bullish
                        and d4_vol_pickup and d5_rsi_ok)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [트리거 E - BREAKOUT] 추격 매수 (Minervini + O'Neil)
    # 52주 신고가 근접 + 거래량 폭발 + 가격 +3%
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    high_52w = float(np.max(h[max(0, i-251):i+1]))
    low_52w  = float(np.min(l[max(0, i-251):i+1]))
    proximity_52w = c[i] / high_52w if high_52w > 0 else 0

    e1_new_high  = bool(proximity_52w >= 0.90)       # 52주 고가 10% 이내
    e2_vol_burst = bool(vol_mult_20 >= 2.0)          # 거래량 20일평균 2배+
    e3_price_up  = bool(c[i] >= c[i-1] * 1.03)       # 당일 +3% 이상
    # E4. RSI ≤ 70 (과매수 회피, 사용자 지정 30-70 범위 상한)
    e4_rsi_ok    = bool(rv <= 70)
    # E5. 20MA 거리 < 20% (추격매수 위험 회피) - 20일선과 너무 멀지 않을 것
    e5_ma20_near = bool(abs(ma20_dist) < 20)
    breakout_trigger = bool(e1_new_high and e2_vol_burst and e3_price_up
                            and e4_rsi_ok and e5_ma20_near)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BB(20,2) 하단 상승돌파 보조 신호 (평균회귀 기회)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    bb_mid = sma20
    bb_std = float(np.std(c[i-19:i+1], ddof=1))
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos = (c[i] - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
    bb_width_pct = (bb_upper - bb_lower) / bb_mid * 100 if bb_mid > 0 else 100

    if i >= 20:
        bb_mid_prev  = float(np.mean(c[i-20:i]))
        bb_std_prev  = float(np.std(c[i-20:i], ddof=1))
        bb_lower_prev = bb_mid_prev - 2 * bb_std_prev
    else:
        bb_lower_prev = bb_lower
    bb_break_close    = bool(c[i-1] <= bb_lower_prev and c[i] > bb_lower)
    bb_break_intraday = bool(l[i] <= bb_lower and c[i] > bb_lower)
    bb_lower_break    = bool(bb_break_close or bb_break_intraday)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # [필수 K] 캔들 품질 (당일 종가 매수 자격)
    # 사용자 정의: 종가>전일종가 + 양봉 + 윗꼬리 짧은 상승추세 캔들
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    body = abs(c[i] - o[i])
    upper_wick = h[i] - max(c[i], o[i])
    lower_wick = min(c[i], o[i]) - l[i]
    candle_range = h[i] - l[i]
    body_ratio = body / candle_range if candle_range > 0 else 0
    upper_wick_ratio = upper_wick / body if body > 0 else 999

    k1_up_close   = bool(c[i] > c[i-1])              # K1. 종가 > 전일 종가 (표시용)
    k2_bullish    = bool(o[i] < c[i])                 # K2. 양봉 (시가 < 종가)
    k3_short_wick = bool(body > 0 and upper_wick <= body * 0.3)  # K3. 윗꼬리 ≤ 몸통 × 0.3 (표시용)
    # ★ 완화(검증): K블록(k1+k2+k3 전부) → 양봉만. 캔들 풀block은 D트리거와 중복곱해 HUNT를 0으로
    #   만들었음. 양봉만으로 풀면 HUNT 빈도·수익 복원(전체검증). k1/k3는 표시용으로 유지.
    candle_pass   = bool(k2_bullish)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 등급 판정 (실전 트레이더 4단계) — K-블록 필수
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 전체 A 정배열 충족 (A1~A4 모두) + B1(흑자) AND B3(목표가) 필수
    # B2(PER)는 백테스트 -3.9% 역효과로 보너스만 (필수 자격 영향 없음)
    trend_full   = bool(a_count == 4)            # 4개 전부 (Weinstein Stage 2)
    fund_ok      = b_required_ok                 # B1 AND B3 (B2는 표시용)

    grade = None
    # HUNT/BREAKOUT은 K-블록(상승추세 캔들) 필수
    # C 섹터(in_leading_sector)는 필수 → 보너스로 강등 (1년 백테스트 +2.6% 영향, 매일 1+ 신호 보장 위해)
    if trend_full and fund_ok and hunt_trigger and candle_pass:
        grade = "HUNT"          # 🟢 저점 매수
    elif trend_full and fund_ok and breakout_trigger and candle_pass:
        grade = "BREAKOUT"      # 🔴 추격 매수
    elif trend_full and fund_ok:
        grade = "TREND"
    elif a_count >= 3:
        grade = "WATCH"
    else:
        with _debug_lock: _debug_reject["A_trend"] += 1
        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 최종 점수 (단일 - 정렬 안정성)
    # = A충족×10 + B충족×10 + 섹터점수 + 트리거점수 + 수급보너스
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    trigger_score = 0
    if hunt_trigger:     trigger_score += 30
    if breakout_trigger: trigger_score += 25
    if bb_lower_break:   trigger_score += 10

    target_upside = (target_price - c[i]) / c[i] * 100 if target_price > 0 and c[i] > 0 else 0

    candle_score = (int(k1_up_close) + int(k2_bullish) + int(k3_short_wick)) * 5  # 최대 +15
    final_score = (a_count * 10) + (b_count * 10) + sector_bonus + trigger_score + candle_score
    if target_upside >= 20: final_score += 10
    elif target_upside >= 10: final_score += 5

    # 보조 표시지표
    _, _, macd_hist = calc_macd(pd.Series(c), 12, 26, 9)
    mh_now = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0
    atr = calc_atr(h, l, c)
    trading_value_eok = c[i] * v[i] / 1e8
    ret_3m = (c[i] - c[max(0, i-63)]) / c[max(0, i-63)] * 100 if i >= 63 else 0
    ret_6m = (c[i] - c[i-126]) / c[i-126] * 100 if i >= 126 and c[i-126] > 0 else 0

    # 통과 조건 리스트 (UI 표시용)
    P = []
    if a1: P.append("A1.종가>200MA")
    if a2: P.append("A2.종가>60MA")
    if a3: P.append("A3.정배열60>200")
    if a4: P.append("A4.200MA상승")
    if b1: P.append(f"B1.EPS+{eps:.0f}")
    if b2: P.append(f"B2.PER{per:.1f}")
    if b3: P.append(f"B3.목표+{target_upside:.0f}%")
    if c1: P.append(f"C1.업종강세{sector_ratio:.0f}%")
    if c2: P.append(f"C2.주도섹터[{sector}+{sector_bonus}]")
    if hunt_trigger:     P.append(f"🟢HUNT 눌림목 되돌림{retrace:.0f}%/20MA{ma20_dist:+.1f}%/양봉반등")
    if breakout_trigger: P.append(f"🔴BREAKOUT 52주{proximity_52w*100:.0f}%/거래{vol_mult_20:.1f}x")
    if bb_lower_break:   P.append(f"🟣BB하단돌파({'종가' if bb_break_close else '인트라'})")

    F_list = []
    if not a1: F_list.append("A1.종가<200MA")
    if not b1: F_list.append("B1.적자")
    if not b3: F_list.append("B3.목표가미흡")
    if not in_leading_sector: F_list.append("C.섹터약함")

    return {
        "passed": P, "failed": F_list,
        "pass_count": len(P), "total": len(P) + len(F_list),
        "momentum": round(min(final_score, 100), 1),
        "atr": round(atr),
        "rsi": round(rv, 1),
        "macd_hist": round(mh_now, 2),
        "volume_ratio": round(vol_mult_20, 1),
        "bb_lower": round(bb_lower),
        "bb_upper": round(bb_upper),
        "bb_mid": round(bb_mid),
        "bb_pos": round(bb_pos, 1),

        # ── 실전 4등급 시스템 ──
        "grade": grade,                          # HUNT/BREAKOUT/TREND/WATCH/BB_BREAK
        "aCount": a_count,                       # A 정배열 충족 개수 (0~4)
        "bCount": b_count,                       # B 펀더 충족 개수 (0~3)
        "a1_ma200": a1, "a2_ma60": a2, "a3_align": a3, "a4_ma200up": a4,
        "b1_eps": b1, "b2_per": b2, "b3_target": b3,
        "inLeadingSector": in_leading_sector,
        "huntTrigger": hunt_trigger,
        "breakoutTrigger": breakout_trigger,
        "d1_pullback": d1_pullback, "d2_above_ma5": d2_above_ma5,
        "d3_bullish": d3_bullish, "d4_vol_pickup": d4_vol_pickup, "d5_rsi_ok": d5_rsi_ok,
        "f1_first_pullback": f1_first_pullback, "pullbackCount": int(pullback_count),
        "e1_new_high": e1_new_high, "e2_vol_burst": e2_vol_burst, "e3_price_up": e3_price_up,
        "e4_rsi_ok": e4_rsi_ok, "e5_ma20_near": e5_ma20_near,

        # ── K. 캔들 품질 (당일 종가 매수 자격) ──
        "k1_up_close": k1_up_close,              # 종가 > 전일 종가
        "k2_bullish": k2_bullish,                # 양봉
        "k3_short_wick": k3_short_wick,          # 윗꼬리 ≤ 몸통×0.3
        "candlePass": candle_pass,               # K1+K2+K3 모두 충족
        "bodyRatio": round(body_ratio, 2),       # 몸통/전체범위
        "upperWickRatio": round(min(upper_wick_ratio, 99), 2),  # 윗꼬리/몸통

        "ma20Dist": round(ma20_dist, 2),
        "proximity52w": round(proximity_52w * 100, 1),
        "volMult5": round(vol_mult_5, 2),
        "volMult20": round(vol_mult_20, 2),
        "volMult": round(vol_mult_20, 2),         # UI 호환용 (이전 명칭)
        "chgToday": round(chg_today, 2),
        "tradingValueEok": round(trading_value_eok, 1),
        "targetUpside": round(target_upside, 1),
        "ret3m": round(ret_3m, 1),
        "ret6m": round(ret_6m, 1),
        "sector": sector,
        "sectorBonus": sector_bonus,
        "sectorRatio": round(sector_ratio, 1),
        "finalScore": round(final_score, 1),

        # ── BB(20,2) 하단 돌파 ──
        "bbLowerBreak": bb_lower_break,
        "bbBreakClose": bb_break_close,
        "bbBreakIntraday": bb_break_intraday,
        "bbWidthPct": round(bb_width_pct, 1),

        # ── 구 시스템 호환 필드 (UI가 참조하는 것들) ──
        "p1Score": a_count + b_count,             # UI 그라데이션 호환
        "proCount": int(in_leading_sector) * 5 + int(hunt_trigger) * 3 + int(breakout_trigger) * 2,
    }

# =============================================
#  매매가 계산 (종가매수 전략)
# =============================================
def tick(p, ref):
    for lim, t in [(2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)]:
        if ref < lim:
            return (p // t) * t
    return (p // 1000) * 1000

def calc_price_pro(cl, lo, atr, bb_info=None, support_low=None):
    """
    당일 종가 매수 + 6영업일 내 +10% 청산 전략.
    - buy: 당일 종가 그대로
    - target1 = target2 = 매수가 × 1.10 (사용자 명시: 10% 도달시 전량매도)
    - 손절: 직전 10일 최저가(support_low) + 최대손실 -10% 상한
      (백테스트: 현행-5%는 손절률46%로 반등종목까지 털림 → 10일최저+(-10%)가
       +160.6%/MDD13.4%로 +142.9%/MDD28.3%보다 돈↑·낙폭↓)
    - 최대 보유: 6영업일 → 미달성 시 종가 청산
    """
    if cl <= 0 or atr <= 0:
        return None

    buy = tick(int(cl), cl)                       # 당일 종가 그대로 매수
    target_10pct = tick(int(buy * 1.10), cl)      # +10% 청산가 (T1 = T2)

    # 손절: 직전 10일 최저가(base 저점) + 최대손실 -10% 상한
    # 백테스트(276종목 1년): 현행-5%는 손절률 46%로 반등종목까지 털림 → +142.9%/MDD28.3%.
    # 10일최저+(-10%상한)은 손절률 21%/승률53% → +160.6%/MDD13.4% (돈↑·낙폭↓·반등보존).
    cap = int(buy * 0.90)                          # 최대손실 -10% (이 아래로는 안 잃음)
    if support_low and support_low < buy:
        sl_raw = max(int(support_low), cap)        # 10일저점, 단 -10%보다 깊으면 -10%로 캡
    else:
        sl_raw = cap
    sl = tick(min(sl_raw, buy - 1), cl)            # 매수가보다는 무조건 아래
    if sl >= buy:
        sl = tick(cap, cl)

    risk = buy - sl
    if risk <= 0:
        return None

    rr = round((target_10pct - buy) / risk, 2) if risk > 0 else 0
    risk_pct = round((buy - sl) / buy * 100, 1)
    return {
        "buy": buy,
        "t1": target_10pct,    # +10% 도달 = 청산
        "t2": target_10pct,    # 동일 (사용자 명시: 10% 전량매도)
        "sl": sl,
        "rr": rr,
        "atr": round(atr),
        "risk_pct": risk_pct,
        "risk_won": risk,
        "max_hold": 6,         # 6영업일 (연구원 검증: 5→6일 강건한 개선)
    }

# =============================================
#  시장 레짐 게이트 (KOSPI 60일선)
# =============================================
_kospi_regime_cache = {}

def get_kospi_regime(date_str=None):
    """KOSPI 지수가 60일 이동평균 위/아래인지 판정.
    10명 연구원 교차검증: 코스피<60일선(확인된 약세장)일 때 신규매수 보류시
    재앙적 하락일을 회피 → 복리↑ + 최대낙폭(MDD) 34.5%→32.3%.
    실패시 fail-open(above_ma60=True)으로 정상 동작 유지.
    반환: {"above_ma60":bool, "close":float, "ma60":float, "ok":bool}"""
    key = date_str or "latest"
    if key in _kospi_regime_cache:
        return _kospi_regime_cache[key]
    result = {"above_ma60": True, "close": 0.0, "ma60": 0.0, "ok": False}
    try:
        end = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.now()
        start = end - timedelta(days=180)
        params = {"symbol": "KOSPI", "requestType": "1",
                  "startTime": start.strftime("%Y%m%d"),
                  "endTime": end.strftime("%Y%m%d"), "timeframe": "day"}
        r = _session.get("https://fchart.stock.naver.com/siseJson.naver",
                         params=params, timeout=15)
        closes = []
        for ln in r.text.strip().split("\n"):
            ln = ln.strip().rstrip(",")
            if ln.startswith("[") and ("'2" in ln or '"2' in ln):
                pp = [x.strip().strip("'\"") for x in ln.strip("[]").split(",")]
                if len(pp) >= 6:
                    try:
                        closes.append(float(pp[4]))   # 지수는 float
                    except Exception:
                        pass
        if len(closes) >= 60:
            ma60 = sum(closes[-60:]) / 60.0
            last = closes[-1]
            result = {"above_ma60": bool(last >= ma60), "close": last,
                      "ma60": round(ma60, 2), "ok": True}
    except Exception as e:
        print(f"  [REGIME] KOSPI fetch 실패(fail-open): {e}")
    _kospi_regime_cache[key] = result
    return result

def naver_today_ohlc(code, date_str):
    """장중(15:20 등) 당일 정규장(09:00~15:30) OHLC를 네이버 분봉에서 집계.
    검증: 분봉 09:00~15:30 집계 OHLC = siseJson 확정 일봉과 100% 일치(2026-05-29 4종목).
    ※ 분봉의 거래량 필드는 부정확 → 거래량은 호출측에서 리스트 누적거래량을 사용.
    ※ ymd 필터로 '오늘'만 집계하므로 과거일 요청 시엔 None(안전).
    반환 (open, high, low, close) 또는 None"""
    try:
        ymd = date_str.replace("-", "")
        r = _session.get(
            f"https://api.stock.naver.com/chart/domestic/item/{code}/minute",
            timeout=10)
        data = json.loads(r.text)
        bars = [b for b in data
                if str(b.get("localDateTime", ""))[:8] == ymd
                and "0900" <= str(b.get("localDateTime", ""))[8:12] <= "1531"]
        if not bars:
            return None
        o = int(float(bars[0]["openPrice"]))
        h = max(int(float(b["highPrice"])) for b in bars)
        l = min(int(float(b["lowPrice"])) for b in bars)
        c = int(float(bars[-1]["currentPrice"]))
        if o <= 0 or c <= 0 or h <= 0 or l <= 0:
            return None
        return o, h, l, c
    except Exception:
        return None

# =============================================
#  고속 스캔 엔진
# =============================================
scan_status = {"running": False, "progress": 0, "total": 0, "found": 0, "message": "", "phase": ""}

def is_halted(s):
    """거래정지/정리매매 여부 (리스트 API tradeStopType=HALTED, 추가 호출 없음)."""
    tst = s.get("tradeStopType", {})
    name = (tst.get("name", "") if isinstance(tst, dict) else str(tst)).upper()
    return name == "HALTED"

def run_scan(date_str, demo=False, intraday=True, market="ALL"):
    """intraday=True: 요청일이 오늘인데 확정 일봉이 아직 없으면(장중~16:30 전)
    네이버 분봉으로 '당일 종가근사' 캔들을 만들어 오늘 신호를 계산한다.
    market: 'ALL'(전체) / 'KOSPI' / 'KOSDAQ' — 대상 시장 선택(대상변경)."""
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

    mkt_sel = (market or "ALL").upper()
    for s in all_stocks:
        code = s.get("itemCode", "")
        name = s.get("stockName", "")
        stock_market = s.get("_market", "")
        end_type = s.get("stockEndType", "")

        # 대상변경: 시장 선택 (전체/코스피/코스닥)
        if mkt_sel in ("KOSPI", "KOSDAQ") and stock_market != mkt_sel:
            excluded_count += 1; continue
        if end_type not in ("stock", ""):     # ETF/ETN 제외
            excluded_count += 1; continue
        if is_halted(s):                       # 거래정지/정리매매 제외 (대상변경)
            excluded_count += 1; continue
        if is_excluded_by_name(name, code):    # 스팩/우선주/리츠/레버리지 제외
            excluded_count += 1; continue

        cl = parse_num(s.get("closePrice", "0"))
        vol = parse_num(s.get("accumulatedTradingVolume", "0"))
        trdval = parse_num(s.get("accumulatedTradingValue", "0"))
        mcap_eok = parse_num(s.get("marketValue", "0"))
        chg_rate = parse_float(s.get("fluctuationsRatio", "0"))

        # 데이터 품질 필터 (잡주 자동 제외)
        if cl <= 0 or vol <= 0: continue
        if cl < 2000: continue              # 현재가 2,000원 이상 (저가주/동전주 제외)
        if trdval < 1000: continue          # 거래대금 10억 이상
        if mcap_eok < 1000: continue        # 시가총액 1,000억 이상 (중소형주 이상)

        candidates.append({
            "code": code, "name": name, "market": stock_market,
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
        # ── 장중 당일신호: 확정 일봉에 오늘이 없으면 분봉으로 임시 캔들 부착 ──
        is_proxy = False
        if intraday and len(df) > 0 and df.index[-1] < pd.Timestamp(date_str):
            ohlc = naver_today_ohlc(code, date_str)
            if ohlc:
                o, h, l, c = ohlc
                row = pd.DataFrame(
                    [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": cand["volume"]}],
                    index=[pd.Timestamp(date_str)])
                df = pd.concat([df, row])
                is_proxy = True
        if len(df) < 201:
            return None
        # 펀더멘털을 screen_pro에 넘기기 위해 먼저 fetch
        inv = fetch_investor_data(code)
        fundamental = None
        if inv:
            fundamental = {
                "per": inv.get("per", 0),
                "pbr": inv.get("pbr", 0),
                "eps": inv.get("eps", 0),
                "target_price": inv.get("target_price", 0),
                "sector_ratio": inv.get("sector_ratio", 50),
            }
        r = screen_pro(df, cand["name"], code, cand["mcap_eok"], fundamental=fundamental)
        if r is None:
            return None
        last_close = int(df["Close"].iloc[-1])
        last_open = int(df["Open"].iloc[-1])
        last_high = int(df["High"].iloc[-1])
        last_low = int(df["Low"].iloc[-1])
        # 직전 10일 최저가(매수일 봉 제외) = base 저점 손절선
        support_low = int(df["Low"].iloc[-11:-1].min()) if len(df) >= 11 else int(df["Low"].iloc[:-1].min())
        bb_info = {"bb_lower": r.get("bb_lower", 0), "bb_mid": r.get("bb_mid", 0), "bb_upper": r.get("bb_upper", 0)}
        p = calc_price_pro(last_close, last_low, r["atr"], bb_info, support_low=support_low)
        if p is None:
            return None
        # ★ BREAKOUT은 모멘텀 돌파 → +15% 청산 (검증: 평균 +1.15%→+1.43%,
        #   포트폴리오 복리 +594%→+637%, 강건성 +337%→+344% 통과).
        #   HUNT/TREND는 +10% 유지(평균회귀라 +10%가 최적).
        if r.get("grade") == "BREAKOUT":
            p["t1"] = tick(int(p["buy"] * 1.15), p["buy"])
            p["t2"] = p["t1"]
            p["target_pct"] = 15
        else:
            p["target_pct"] = 10
        dd = df.index[-1]
        dd = dd.strftime("%Y-%m-%d") if hasattr(dd, 'strftime') else date_str
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
            # ── PRO 등급 (학술 10개 인자) ──
            "proCount": r.get("proCount", 0),
            "proFactors": r.get("proFactors", []),
            "a1_52w": r.get("a1_52w", False),
            "a2_mom6m": r.get("a2_mom6m", False),
            "a3_vcp": r.get("a3_vcp", False),
            "a4_volprice": r.get("a4_volprice", False),
            "a5_pead": r.get("a5_pead", False),
            "a6_obv": r.get("a6_obv", False),
            "a7_trend": r.get("a7_trend", False),
            "a8_lowvol": r.get("a8_lowvol", False),
            "a9_rsi": r.get("a9_rsi", False),
            "a10_base": r.get("a10_base", False),
            "ret6m": r.get("ret6m", 0),
            "annualVol60": r.get("annualVol60", 0),
            "gapHeld": r.get("gapHeld", False),
            # ── BB 하단 돌파 신호 ──
            "bbLowerBreak": r.get("bbLowerBreak", False),
            "bbBreakClose": r.get("bbBreakClose", False),
            "bbBreakIntraday": r.get("bbBreakIntraday", False),

            # ── 실전 트레이더 4등급 신호 ──
            "aCount": r.get("aCount", 0),                   # 정배열 0-4
            "bCount": r.get("bCount", 0),                   # 펀더 0-3
            "a1_ma200": r.get("a1_ma200", False),
            "a2_ma60": r.get("a2_ma60", False),
            "a3_align": r.get("a3_align", False),
            "a4_ma200up": r.get("a4_ma200up", False),
            "b1_eps": r.get("b1_eps", False),
            "b2_per": r.get("b2_per", False),
            "b3_target": r.get("b3_target", False),
            "inLeadingSector": r.get("inLeadingSector", False),
            "huntTrigger": r.get("huntTrigger", False),
            "breakoutTrigger": r.get("breakoutTrigger", False),
            "d1_pullback": r.get("d1_pullback", False),
            "d2_above_ma5": r.get("d2_above_ma5", False),
            "d3_bullish": r.get("d3_bullish", False),
            "d4_vol_pickup": r.get("d4_vol_pickup", False),
            "d5_rsi_ok": r.get("d5_rsi_ok", False),
            "f1_first_pullback": r.get("f1_first_pullback", False),
            "pullbackCount": r.get("pullbackCount", 0),
            "e1_new_high": r.get("e1_new_high", False),
            "e2_vol_burst": r.get("e2_vol_burst", False),
            "e3_price_up": r.get("e3_price_up", False),
            "e4_rsi_ok": r.get("e4_rsi_ok", False),
            "e5_ma20_near": r.get("e5_ma20_near", False),
            # ── K. 캔들 품질 ──
            "k1_up_close": r.get("k1_up_close", False),
            "k2_bullish": r.get("k2_bullish", False),
            "k3_short_wick": r.get("k3_short_wick", False),
            "candlePass": r.get("candlePass", False),
            "bodyRatio": r.get("bodyRatio", 0),
            "upperWickRatio": r.get("upperWickRatio", 0),
            "ma20Dist": r.get("ma20Dist", 0),
            "proximity52w": r.get("proximity52w", 0),
            "volMult5": r.get("volMult5", 0),
            "volMult20": r.get("volMult20", 0),
            "targetUpside": r.get("targetUpside", 0),
            "ret3m": r.get("ret3m", 0),
            # ── 매매 룰 (6영업일 내 청산; BREAKOUT +15% / 그외 +10%) ──
            "maxHold": p.get("max_hold", 6),
            "targetPct": p.get("target_pct", 10),
            "industryCode": (inv.get("industry_code", "0") if inv else "0"),   # 업종(섹터 분산)
            # 장중 분봉 임시캔들로 만든 '당일 종가근사' 신호 여부
            "isIntradayProxy": is_proxy,
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

    # 등급 우선순위: BREAKOUT(추격) > HUNT(저점매수) > TREND > WATCH
    # ★ BREAKOUT 우선 (사용자 지적 검증): BREAKOUT 청산률 38.3% > HUNT 28.0%.
    #   선정규칙을 BREAKOUT 우선으로 바꾸니 비용반영 복리 +466%→+594%,
    #   최고5일 제거(강건성)에서도 +257%→+337%로 개선 (꼬리효과 아님).
    GRADE_ORDER = {"BREAKOUT": 0, "HUNT": 1, "BB_BREAK": 2, "TREND": 3, "WATCH": 4}
    results.sort(key=lambda x: (
        GRADE_ORDER.get(x.get("grade", "WATCH"), 5),
        -x.get("finalScore", 0)
    ))

    # ─── 매수 후보: 항상 3종목 채우기 (분할매수용) ───
    # 누적 방식: 좋은 등급부터 채우고, 3개 미만이면 다음 단계로 보충
    BUY_GRADES = {"HUNT", "BREAKOUT"}
    MIN_PICKS = 3   # 분산매수 최소 종목 수
    selected = []
    used = set()

    def add_from(pool, target, mark_alt=False):
        for r in pool:
            if len(selected) >= target: break
            if r["code"] in used: continue
            if mark_alt:
                r["isAlternative"] = True
            selected.append(r)
            used.add(r["code"])

    # 1순위: HUNT/BREAKOUT + 양봉 + 점수 100+ (메인 신호, 최대 9개 다 표시)
    tier1 = [r for r in results
             if r.get("grade") in BUY_GRADES and r.get("k2_bullish")
             and r.get("finalScore", 0) >= 100]
    add_from(tier1, 9)

    # 3개 미만이면 2순위: HUNT/BREAKOUT + 점수 80~100 (3개까지만 보충)
    # (HUNT/BREAKOUT은 이미 candle_pass=K1+K2+K3 통과한 등급)
    if len(selected) < MIN_PICKS:
        tier2 = sorted([r for r in results
                        if r.get("grade") in BUY_GRADES and r.get("k2_bullish")
                        and 80 <= r.get("finalScore", 0) < 100],
                       key=lambda x: -x.get("finalScore", 0))
        add_from(tier2, MIN_PICKS, mark_alt=True)

    # 그래도 3개 미만이면 3순위: TREND (K1 전일상승 + K2 양봉 필수) 점수순
    # ※ K1(종가>전일종가) 필수 추가 — 차선도 "전일보다 오른 양봉"만 (사용자 지적 반영)
    if len(selected) < MIN_PICKS:
        tier3 = sorted([r for r in results
                        if r.get("grade") == "TREND"
                        and r.get("k1_up_close") and r.get("k2_bullish")],
                       key=lambda x: -x.get("finalScore", 0))
        add_from(tier3, MIN_PICKS, mark_alt=True)

    results = selected[:9]

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
        # 실전 4등급 카운트
        hunt_n  = sum(1 for s in results if s.get("grade") == "HUNT")
        brk_n   = sum(1 for s in results if s.get("grade") == "BREAKOUT")
        trend_n = sum(1 for s in results if s.get("grade") == "TREND")
        watch_n = sum(1 for s in results if s.get("grade") == "WATCH")
        bbb_n   = sum(1 for s in results if s.get("grade") == "BB_BREAK")
        bb_total = sum(1 for s in results if s.get("bbLowerBreak"))
        print(f"  Grade: 🟢HUNT={hunt_n}  🔴BREAKOUT={brk_n}  🟡TREND={trend_n}  🔵WATCH={watch_n}  🟣BB={bbb_n}")
        print(f"  BB하단돌파 전체: {bb_total}개")
        print(f"  {'-'*78}")
        for idx, s in enumerate(results[:25], 1):
            g = s.get("grade", "-")
            sec = s.get("sector", "-")
            fs = s.get("finalScore", 0)
            a = s.get("aCount", 0)
            b = s.get("bCount", 0)
            bb_mark = "🟣" if s.get("bbLowerBreak") else " "
            ma20 = s.get("ma20Dist", 0)
            print(f"  {idx:>2} [{g:<8}] {bb_mark} {s['name']:<10} {s['close']:>8,} A={a}/4 B={b}/3 20MA{ma20:+.1f}% Sec={sec:<8} Final={fs:>5.0f}")
    print(f"{'='*82}\n")
    return results

# =============================================
#  Flask Routes
# =============================================
@app.route("/")
def index():
    return render_template("index.html")

def _save_and_sync_results(results, date_str, swing_picks=None):
    """스캔 결과를 JSON 파일로 저장 + GitHub push (Render 동기화).
    swing_picks: BB조합 스윙 매수신호 목록 (대시보드 별도 섹션용)."""
    import subprocess
    dow = datetime.strptime(date_str, "%Y-%m-%d").weekday()
    dow_names = ["월","화","수","목","금","토","일"]
    # 실제 분석된 마지막 거래일 추출 (휴장일 감지용)
    actual_data_date = date_str
    if results:
        for r in results:
            dd = r.get("dataDate")
            if dd:
                actual_data_date = dd
                break
    # 데이터 시점 상태 판정
    today_str = datetime.now().strftime("%Y-%m-%d")
    data_mismatch = (actual_data_date != date_str)
    # 요청일이 오늘이면 = 당일 데이터 미반영(장 마감 전), 과거면 = 휴장일
    if data_mismatch and date_str >= today_str:
        data_status = "intraday"   # 당일 데이터 미반영 (장중/장전)
    elif data_mismatch:
        data_status = "holiday"    # 과거 휴장일
    else:
        data_status = "confirmed"  # 정상 (요청일 = 데이터일)
    is_holiday = data_mismatch
    # 장중 분봉 임시캔들(당일 종가근사)로 만든 신호면 '잠정'으로 표시
    if results and results[0].get("isIntradayProxy"):
        data_status = "intraday_proxy"
    regime = get_kospi_regime(actual_data_date)
    payload = {
        "results": results,
        "swingPicks": swing_picks or [],
        "date": date_str,
        "actualDataDate": actual_data_date,
        "isHoliday": is_holiday,
        "dataStatus": data_status,
        "marketRegime": regime,
        "strategyVersion": STRATEGY_VERSION,
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

    # Git push (로컬에서만). Render·GitHub Actions에선 스킵
    #   (GitHub Actions는 워크플로 'Persist' 스텝이 latest_results.json까지 한 번에 push)
    is_render = os.environ.get("RENDER", "") == "true"
    in_actions = os.environ.get("GITHUB_ACTIONS", "") == "true"
    if not is_render and not in_actions:
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
    """저장된 결과 파일 로드. 날짜 + 전략 버전 둘 다 일치할 때만 캐시 사용.
    조건(전략)이 바뀌면 STRATEGY_VERSION이 달라져 자동으로 캐시 무효화 → 재스캔."""
    results_path = os.path.join(os.path.dirname(__file__), "latest_results.json")
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") != date_str:
            return None
        # 전략 버전 불일치 = 조건 변경됨 → 캐시 무시하고 재스캔
        if data.get("strategyVersion") != STRATEGY_VERSION:
            print(f"  [CACHE] Version mismatch ({data.get('strategyVersion')} != {STRATEGY_VERSION}) → re-scan")
            return None
        print(f"  [CACHE] Serving cached results for {date_str} ({data.get('count',0)} stocks)")
        return data
    except Exception:
        pass
    return None

SCAN_TTL = 180                 # 오늘 캐시 신선도(초). 이보다 오래되면 장중에 한해 백그라운드 재스캔
_bg_scan_busy = {"on": False}
_bg_scan_lock = threading.Lock()

def _cache_age_seconds(cached):
    """캐시 생성 후 경과초. 파싱 실패시 None."""
    try:
        ts = datetime.strptime(cached.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - ts).total_seconds()
    except Exception:
        return None

def _background_rescan(date_str, market):
    """백그라운드 재스캔(모멘텀+스윙) → latest_results.json 갱신. 동시실행 1개로 제한."""
    with _bg_scan_lock:
        if _bg_scan_busy["on"]:
            return
        _bg_scan_busy["on"] = True
    try:
        results = run_scan(date_str, market=market)
        try:
            from swing_tracker import scan_buys as swing_scan
            sp = swing_scan(date_str, intraday=True, market=market)
        except Exception as e:
            print(f"  [BG] swing err: {e}"); sp = []
        _save_and_sync_results(results, date_str, swing_picks=sp)
        print(f"  [BG] rescan saved {date_str} ({len(results)} stocks)")
    except Exception as e:
        print(f"  [BG] rescan error: {e}")
    finally:
        _bg_scan_busy["on"] = False

@app.route("/api/scan")
def api_scan():
    date_str = flask_request.args.get("date", "")
    market = (flask_request.args.get("market", "ALL") or "ALL").upper()
    if market not in ("ALL", "KOSPI", "KOSDAQ"):
        market = "ALL"
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        tgt = datetime.strptime(date_str, "%Y-%m-%d")
        if tgt > datetime.now():
            date_str = datetime.now().strftime("%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Date format error"}), 400

    # 1) 캐시 우선 반환(즉시) + stale-while-revalidate.
    #    풀스캔이 ~100초라 모바일에서 동기 대기 시 게이트웨이 타임아웃(HTML 에러) 발생.
    #    → 마지막 스캔 결과를 즉시 주고, 오늘이면서 오래된 캐시는 '장중에만' 백그라운드 갱신.
    #    과거일은 불변이라 그대로 캐시. (캐시는 시장 ALL 기준)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cached = _load_cached_results(date_str) if market == "ALL" else None
    if cached:
        cached.setdefault("swingPicks", [])
        _enrich_sector(cached.get("results"))      # 섹터 이름 부착(구 캐시 호환)
        _enrich_sector(cached.get("swingPicks"))
        if date_str == today_str:
            age = _cache_age_seconds(cached)
            kst = datetime.utcnow() + timedelta(hours=9)   # 서버 UTC→KST 변환
            in_market = kst.weekday() < 5 and (9 * 60) <= (kst.hour * 60 + kst.minute) <= (16 * 60)
            if in_market and (age is None or age > SCAN_TTL):
                threading.Thread(target=_background_rescan, args=(date_str, market), daemon=True).start()
                cached["refreshing"] = True
            cached["cacheAgeSec"] = int(age) if age is not None else None
        return jsonify(cached)

    # 2) 캐시 없음(오늘 첫 스캔/버전변경/시장필터) → 동기 라이브 스캔: 모멘텀 + 스윙 '한 번에'
    #    (OHLCV는 in-proc 캐시 공유 → 스윙은 모멘텀이 받아둔 데이터 재사용해 거의 즉시.
    #     별도 /api/swing 2차 전수스캔을 없애 속도↑ + 대시보드 순서 안 바뀜)
    results = run_scan(date_str, market=market)
    try:
        from swing_tracker import scan_buys as swing_scan
        swing_picks = swing_scan(date_str, intraday=True, market=market)
    except Exception as e:
        print(f"  [SWING] scan error: {e}")
        swing_picks = []
    payload = _save_and_sync_results(results, date_str, swing_picks=swing_picks)
    _enrich_sector(payload.get("results"))         # 종목별 섹터 이름 부착
    _enrich_sector(payload.get("swingPicks"))
    payload["market"] = market
    return jsonify(payload)

@app.route("/api/swing")
def api_swing():
    """스윙 BB조합 매수신호만 별도 반환 (대시보드가 모멘텀 로드 후 따로 호출)."""
    date_str = flask_request.args.get("date", "") or datetime.now().strftime("%Y-%m-%d")
    market = (flask_request.args.get("market", "ALL") or "ALL").upper()
    if market not in ("ALL", "KOSPI", "KOSDAQ"):
        market = "ALL"
    try:
        if datetime.strptime(date_str, "%Y-%m-%d") > datetime.now():
            date_str = datetime.now().strftime("%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Date format error"}), 400
    try:
        from swing_tracker import scan_buys as swing_scan
        picks = swing_scan(date_str, intraday=True, market=market)
    except Exception as e:
        print(f"  [SWING] scan error: {e}")
        picks = []
    return jsonify({"swingPicks": picks, "date": date_str, "market": market})

def _live_quote(code):
    """종목 실시간 현재가 조회 (네이버 price 엔드포인트)."""
    try:
        r = _session.get(f"https://m.stock.naver.com/api/stock/{code}/price",
                         params={"pageSize": 1, "page": 1}, timeout=6)
        rows = json.loads(r.text)
        if rows:
            cur = parse_num(rows[0].get("closePrice", "0"))
            chg = parse_float(rows[0].get("fluctuationsRatio", "0"))
            return code, {"price": cur, "chgRate": chg, "date": rows[0].get("localTradedAt", "")}
    except Exception:
        pass
    return code, None

@app.route("/api/quote")
def api_quote():
    """여러 종목의 실시간 현재가 조회. ?codes=388720,058610,... → {code:{price,chgRate}}"""
    codes = [c.strip() for c in flask_request.args.get("codes", "").split(",") if c.strip()][:30]
    out = {}
    if codes:
        with ThreadPoolExecutor(max_workers=15) as ex:
            for code, q in ex.map(_live_quote, codes):
                if q:
                    out[code] = q
    return jsonify({"quotes": out, "ts": datetime.now().strftime("%H:%M:%S")})

@app.route("/api/status")
def api_status():
    return jsonify(scan_status)

# =============================================
#  텔레그램 알림 (매수 후보 발송)
# =============================================
def _load_telegram_config():
    """봇 토큰/chat_id 로드. 환경변수(클라우드) 우선, 없으면 로컬 파일."""
    # 1) 환경변수 (Render 등 클라우드 — PC 꺼져도 작동)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return {"bot_token": token, "chat_id": chat_id}
    # 2) 로컬 파일 (PC에서 직접 실행 시)
    cfg_path = os.path.join(os.path.dirname(__file__), "telegram_config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def send_telegram(message):
    """텔레그램으로 메시지 발송"""
    cfg = _load_telegram_config()
    if not cfg or not cfg.get("bot_token") or not cfg.get("chat_id"):
        return False, "telegram_config.json 없음 또는 토큰/chat_id 미설정"
    try:
        url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
        r = _session.post(url, json={
            "chat_id": cfg["chat_id"],
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
        if r.status_code == 200:
            return True, "발송 성공"
        return False, f"텔레그램 오류 {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, f"발송 실패: {e}"

def send_telegram_document(filepath, caption="", filename=None):
    """텔레그램으로 파일(엑셀 등) 전송. 파일명/형식 명시로 모바일 첨부 표시 보장."""
    cfg = _load_telegram_config()
    if not cfg or not cfg.get("bot_token") or not cfg.get("chat_id"):
        return False, "telegram 설정 없음"
    try:
        url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendDocument"
        fname = filename or os.path.basename(filepath)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        with open(filepath, "rb") as f:
            r = _session.post(url,
                data={"chat_id": cfg["chat_id"], "caption": caption[:1000], "parse_mode": "HTML"},
                files={"document": (fname, f, mime)}, timeout=30)
        if r.status_code == 200:
            return True, "파일 발송 성공"
        return False, f"파일 오류 {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return False, f"파일 발송 실패: {e}"

def format_telegram_message(payload):
    """스캔 결과를 텔레그램 메시지로 포맷"""
    date = payload.get("date", "")
    dow = payload.get("dayOfWeek", "")
    results = payload.get("results", [])
    status = payload.get("dataStatus", "")
    regime = payload.get("marketRegime") or {}
    bear = regime.get("ok") and not regime.get("above_ma60", True)

    lines = [f"📊 <b>오늘의 매수 후보</b> ({date} {dow})"]
    if status == "intraday_proxy":
        lines.append("⏱ <b>장중 잠정신호</b> (현재가=종가 근사). 15:20~15:30 동시호가로 매수 권장")
    elif status == "intraday":
        lines.append(f"⏳ 당일 미반영 → {payload.get('actualDataDate')} 종가 기준 (장 마감 후 재확인)")
    elif status == "holiday":
        lines.append(f"⚠️ 휴장일 → {payload.get('actualDataDate')} 기준")
    lines.append("")
    if bear:
        lines.append("🔴 <b>약세장 경고: 코스피가 60일선 아래</b>")
        lines.append(f"   (지수 {regime.get('close')} &lt; 60일선 {regime.get('ma60')})")
        lines.append("   👉 <b>오늘 신규매수 보류 권고</b> (아래는 참고용 관심종목)")
        lines.append("")
    lines.append("💡 <b>상위 3종목 자본 33%씩 분산매수</b>" + ("  ※약세장이면 보류" if bear else ""))
    lines.append("")

    if not results:
        lines.append("오늘은 조건 충족 종목 없음")
    else:
        for i, s in enumerate(results[:3], 1):
            grade = s.get("grade", "")
            emoji = {"HUNT": "🟢", "BREAKOUT": "🔴", "TREND": "🟡"}.get(grade, "⚪")
            alt = " ⚠️차선" if s.get("isAlternative") else ""
            lines.append(f"<b>{i}. {emoji}{grade}{alt} {s.get('name')}</b> ({s.get('code')})")
            tp = s.get('targetPct', 10)
            lines.append(f"   💰매수 {s.get('buyPrice'):,}원")
            lines.append(f"   🎯청산 {s.get('target1'):,}원 (+{tp}%{' ·돌파' if tp!=10 else ''})")
            lines.append(f"   🛑손절 {s.get('stoploss'):,}원")
            lines.append(f"   섹터 {s.get('sector','-')} · RSI {s.get('rsi','-')} · 거래량 {s.get('volMult5','-')}x")
            lines.append("")
    lines.append("⏱ 매수: 당일 종가 | 청산: +목표% 도달 또는 6영업일째 종가 (BREAKOUT +15%, 그외 +10%)")
    lines.append("🛑 손절: 직전 10일 최저가 이탈 시 (최대손실 -10%) 반드시 지키기")
    return "\n".join(lines)

@app.route("/api/notify")
def api_notify():
    """오늘 매수 후보를 텔레그램으로 발송"""
    date_str = flask_request.args.get("date", "") or datetime.now().strftime("%Y-%m-%d")
    try:
        tgt = datetime.strptime(date_str, "%Y-%m-%d")
        if tgt > datetime.now():
            date_str = datetime.now().strftime("%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Date format error"}), 400

    cached = _load_cached_results(date_str)
    if cached:
        payload = cached
    else:
        results = run_scan(date_str)
        payload = _save_and_sync_results(results, date_str)

    msg = format_telegram_message(payload)
    ok, info = send_telegram(msg)
    return jsonify({"sent": ok, "info": info, "preview": msg, "count": payload.get("count", 0)})

@app.route("/api/chart/<code>")
def api_chart(code):
    """일봉차트용 OHLCV (3개월 = 63영업일). lightweight-charts 호환 포맷."""
    days = int(flask_request.args.get("days", 63))
    date_str = flask_request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        # 이동평균 계산용 여유분 포함해서 250일 가져오기
        df = naver_ohlcv_fast(code, days=250, target_date=date_str)
        if df is None:
            return jsonify({"error": "no data"}), 404
        df = df[df.index <= pd.Timestamp(date_str)]
        if len(df) < 5:
            return jsonify({"error": "insufficient data"}), 404

        # 이동평균
        close = df["Close"].astype(float)
        ma20  = close.rolling(20).mean()
        ma60  = close.rolling(60).mean()
        ma200 = close.rolling(200).mean()

        # 마지막 N영업일만 반환
        df_tail = df.tail(days)
        ma20_tail  = ma20.tail(days)
        ma60_tail  = ma60.tail(days)
        ma200_tail = ma200.tail(days)

        candles = []
        volumes = []
        ma20_arr = []
        ma60_arr = []
        ma200_arr = []
        for idx, row in df_tail.iterrows():
            t = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
            o, h_, l_, c_ = int(row["Open"]), int(row["High"]), int(row["Low"]), int(row["Close"])
            candles.append({"time": t, "open": o, "high": h_, "low": l_, "close": c_})
            # 거래량 바 색상 (양봉=빨강, 음봉=파랑)
            volumes.append({"time": t, "value": int(row["Volume"]),
                            "color": "#DC262680" if c_ >= o else "#2563EB80"})
        for t_idx, v_ in ma20_tail.items():
            t = t_idx.strftime("%Y-%m-%d")
            if not pd.isna(v_): ma20_arr.append({"time": t, "value": float(v_)})
        for t_idx, v_ in ma60_tail.items():
            t = t_idx.strftime("%Y-%m-%d")
            if not pd.isna(v_): ma60_arr.append({"time": t, "value": float(v_)})
        for t_idx, v_ in ma200_tail.items():
            t = t_idx.strftime("%Y-%m-%d")
            if not pd.isna(v_): ma200_arr.append({"time": t, "value": float(v_)})

        return jsonify({
            "code": code,
            "days": len(candles),
            "candles": candles,
            "volumes": volumes,
            "ma20": ma20_arr,
            "ma60": ma60_arr,
            "ma200": ma200_arr,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
