#!/usr/bin/env python3
"""
종가매수 스크리너 웹앱 - 전문가급 조건검색 (고속 버전)
Minervini SEPA + O'Neil CANSLIM + Weinstein Stage Analysis 기반
5가지 속도 최적화:
  1. requests.Session 커넥션 풀링 (TCP 재사용 → 요청당 30-50% 단축)
  2. Phase 1 병렬 (KOSPI+KOSDAQ 동시 조회)
  3. Phase 2 강화 필터 (731 → ~300 후보)
  4. Phase 3 워커 20개 (10 → 20)
  5. Phase 4 경고체크 병렬화 + 메모리 캐시
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

# 클라우드 환경 감지 (Render는 RENDER 환경변수가 있음)
IS_CLOUD = os.environ.get("RENDER") is not None

def check_naver_api():
    """네이버 API 접근 가능 여부 빠르게 확인"""
    try:
        r = requests.get(
            "https://m.stock.naver.com/api/stocks/up/KOSPI?page=1&pageSize=1",
            timeout=5
        )
        return r.status_code == 200 and len(r.json().get("stocks", [])) > 0
    except:
        return False

NAVER_OK = None  # 첫 스캔 때 확인

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
            r = _session.get(url, timeout=10)
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
    """KOSPI + KOSDAQ 동시 조회 (병렬)"""
    results = {}
    def fetch(market):
        stocks = naver_stock_list(market, "up")
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
    print(f"  [NAVER] KOSPI: {len(kospi)} | KOSDAQ: {len(kosdaq)} (parallel)")
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
                         params=params, timeout=7)
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
#  전문가급 조건검색 (16개 조건)
# =============================================
def screen_pro(df, name="", code="", mcap=0):
    if len(df) < 201:
        return None

    c = df["Close"].values
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    v = df["Volume"].values
    i = len(df) - 1

    # Must-pass
    if c[i] <= o[i]: return None
    if c[i] <= c[i-1]: return None
    chg = (c[i] - c[i-1]) / c[i-1] * 100
    if chg >= 29: return None

    av50 = np.mean(v[max(0, i-49):i]) if i >= 50 else np.mean(v[:i])
    vr = v[i] / av50 if av50 > 0 else 0
    if vr < 1.2: return None

    P, F = [], []
    score = 0

    # SMA
    sma50 = np.mean(c[i-49:i+1])
    sma150 = np.mean(c[i-149:i+1])
    sma200 = np.mean(c[i-199:i+1])
    sma200_1m = np.mean(c[i-221:i-22+1]) if i >= 221 else sma200

    # EMA
    ema10 = pd.Series(c).ewm(span=10, adjust=False).mean().values
    ema21 = pd.Series(c).ewm(span=21, adjust=False).mean().values
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values

    # 52주
    lookback_52w = min(i + 1, 250)
    high_52w = max(h[i - lookback_52w + 1:i + 1])
    low_52w = min(l[i - lookback_52w + 1:i + 1])

    # A. Price > 150MA & 200MA
    if c[i] > sma150 and c[i] > sma200:
        P.append("A.추세상위"); score += 7
    else:
        F.append("A.추세상위")

    # B. 150MA > 200MA
    if sma150 > sma200:
        P.append("B.중기정배열"); score += 5
    else:
        F.append("B.중기정배열")

    # C. 200MA rising
    if sma200 > sma200_1m:
        P.append("C.장기상승"); score += 5
    else:
        F.append("C.장기상승")

    # D. 50 > 150 > 200
    if sma50 > sma150 > sma200:
        P.append("D.완전정배열"); score += 8
    else:
        F.append("D.완전정배열")

    # E. Triple Stack
    if ema10[i] > ema21[i] > sma50:
        P.append("E.트리플스택"); score += 7
    else:
        F.append("E.트리플스택")

    # F. 52w low +25%
    if low_52w > 0 and c[i] >= low_52w * 1.25:
        P.append("F.52주저+25%"); score += 5
    else:
        F.append("F.52주저+25%")

    # G. 52w high 75%
    if high_52w > 0 and c[i] >= high_52w * 0.75:
        P.append("G.52주고근접"); score += 5
    else:
        F.append("G.52주고근접")

    # H. Volume surge
    if vr >= 3.0:
        P.append(f"H.폭증{vr:.1f}x"); score += 10
    elif vr >= 2.0:
        P.append(f"H.급증{vr:.1f}x"); score += 8
    elif vr >= 1.5:
        P.append(f"H.증가{vr:.1f}x"); score += 6
    else:
        P.append(f"H.소폭{vr:.1f}x"); score += 2

    # I. OBV
    obv_delta = calc_obv_trend(c, v, 20)
    if obv_delta > 0:
        P.append("I.OBV매집"); score += 5
    else:
        F.append("I.OBV이탈")

    # J. RSI
    rv = calc_rsi(pd.Series(c), 14).iloc[-1]
    if not np.isnan(rv):
        if 45 <= rv <= 75:
            P.append(f"J.RSI{rv:.0f}"); score += 7
        elif rv > 75:
            F.append(f"J.RSI과열{rv:.0f}")
        elif rv > 40:
            P.append(f"J.RSI{rv:.0f}"); score += 3
        else:
            F.append(f"J.RSI약세{rv:.0f}")
    else:
        F.append("J.RSI-")

    # K. MACD
    macd_line, signal_line, macd_hist = calc_macd(pd.Series(c), 12, 26, 9)
    mh_now = macd_hist.iloc[-1]
    mh_prev = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
    ml_now = macd_line.iloc[-1]
    if not np.isnan(mh_now):
        if mh_now > 0 and mh_now > mh_prev:
            P.append("K.MACD가속"); score += 8
        elif mh_now > 0:
            P.append("K.MACD양전"); score += 4
        elif ml_now > 0:
            P.append("K.MACD+"); score += 2
        else:
            F.append("K.MACD음전")
    else:
        F.append("K.MACD-")

    # L. Stochastic
    sk, sd = calc_stoch(pd.Series(h), pd.Series(l), pd.Series(c), 14, 3)
    sk_v = sk.iloc[-1]
    sd_v = sd.iloc[-1]
    if not np.isnan(sk_v) and not np.isnan(sd_v):
        if 20 <= sk_v <= 80 and sk_v > sd_v:
            P.append(f"L.Stoch{sk_v:.0f}"); score += 7
        elif sk_v > sd_v:
            P.append(f"L.Stoch{sk_v:.0f}"); score += 3
        else:
            F.append("L.Stoch역배열")
    else:
        F.append("L.Stoch-")

    # M. 20EMA rising
    if ema20[i] > ema20[i-1] > ema20[i-2]:
        P.append("M.EMA상승"); score += 5
    else:
        F.append("M.EMA횡보")

    # N. Bollinger
    bb_mid = np.mean(c[i-19:i+1])
    bb_std = np.std(c[i-19:i+1], ddof=1)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    if bb_mid <= c[i] <= bb_upper:
        P.append("N.BB상단구간"); score += 5
    elif c[i] > bb_lower:
        P.append("N.BB중간"); score += 2
    else:
        F.append("N.BB하단")

    # O. Candle
    body = abs(c[i] - o[i])
    rng = h[i] - l[i]
    body_ratio = body / rng if rng > 0 else 0
    if body_ratio >= 0.65:
        P.append("O.장대양봉"); score += 5
    else:
        cn = 0
        for j in range(i, max(i - 5, 0), -1):
            if c[j] > o[j]: cn += 1
            else: break
        if cn >= 3:
            P.append(f"O.{cn}연양봉"); score += 4
        elif cn >= 2:
            P.append(f"O.{cn}연양봉"); score += 2
        else:
            F.append("O.캔들보통")

    # P. RS
    if sma200 > 0:
        rs_pct = (c[i] / sma200 - 1) * 100
        if rs_pct > 20:
            P.append(f"P.RS강+{rs_pct:.0f}%"); score += 5
        elif rs_pct > 5:
            P.append(f"P.RS양+{rs_pct:.0f}%"); score += 3
        elif rs_pct > 0:
            P.append(f"P.RS+{rs_pct:.0f}%"); score += 1
        else:
            F.append("P.RS약세")
    else:
        F.append("P.RS-")

    if len(P) < 8:
        return None

    atr = calc_atr(h, l, c)

    return {
        "passed": P, "failed": F,
        "pass_count": len(P), "total": len(P) + len(F),
        "momentum": round(min(score, 100), 1),
        "atr": round(atr),
        "rsi": round(rv, 1) if not np.isnan(rv) else 0,
        "macd_hist": round(float(mh_now), 2) if not np.isnan(mh_now) else 0,
        "volume_ratio": round(vr, 1)
    }

# =============================================
#  매매가 계산
# =============================================
def tick(p, ref):
    for lim, t in [(2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)]:
        if ref < lim:
            return (p // t) * t
    return (p // 1000) * 1000

def calc_price_pro(cl, lo, atr):
    if cl <= 0 or atr <= 0:
        return None
    buy = tick(int(cl * 1.003), cl)
    sl_atr = int(buy - 2.0 * atr)
    sl_low = int(lo * 0.995)
    sl_max = int(buy * 0.92)
    sl = tick(max(sl_atr, sl_low, sl_max), cl)
    risk = buy - sl
    if risk <= 0:
        return None
    t1 = tick(int(buy + 2.0 * risk), cl)
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
    global scan_status, NAVER_OK
    results = []

    # 클라우드에서 네이버 API 차단 여부 확인 (최초 1회)
    if not demo and NAVER_OK is None:
        print("  [CLOUD CHECK] Naver API 접근 테스트 중...")
        NAVER_OK = check_naver_api()
        print(f"  [CLOUD CHECK] Naver API: {'OK' if NAVER_OK else 'BLOCKED - 데모모드로 전환'}")
    if not demo and NAVER_OK is False:
        demo = True
        print("  [AUTO] 해외서버 감지 → 데모모드 자동 전환")

    if demo:
        scan_status = {"running": True, "progress": 0, "total": 60, "found": 0,
                       "message": "Demo data...", "phase": "demo"}
        from app_demo import run_demo_scan
        results = run_demo_scan(date_str, scan_status)
        scan_status["running"] = False
        scan_status["message"] = "done"
        return results

    t_start = time.time()

    # ── Phase 1: KOSPI + KOSDAQ 병렬 조회 ──
    scan_status = {"running": True, "progress": 0, "total": 0, "found": 0,
                   "message": "KOSPI+KOSDAQ parallel fetch...", "phase": "krx"}
    print(f"\n{'='*60}")
    print(f"  [SCAN] Pro Screener | {date_str}")
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

        # 강화 필터 (후보 수 ~절반으로 감소)
        if cl <= 0 or vol <= 0: continue
        if chg_rate < 0.3: continue        # 0.3% 미만 상승 제외 (의미있는 상승만)
        if cl < 500: continue
        if trdval < 800: continue           # 거래대금 8억 미만 제외 (기존 5억)
        if mcap_eok < 500: continue

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
        p = calc_price_pro(last_close, last_low, r["atr"])
        if p is None:
            return None
        dd = df.index[-1]
        dd = dd.strftime("%Y-%m-%d") if hasattr(dd, 'strftime') else date_str
        return {
            "code": code, "name": cand["name"], "market": cand["market"],
            "close": last_close, "open": last_open,
            "high": last_high, "low": last_low,
            "volume": cand["volume"], "marketCap": cand["mcap_eok"],
            "buyPrice": p["buy"], "target1": p["t1"], "target2": p["t2"],
            "stoploss": p["sl"], "rrRatio": p["rr"],
            "atr": p["atr"], "riskPct": p["risk_pct"],
            "momentum": r["momentum"],
            "conditionsMet": r["passed"],
            "conditionsDetail": f'{r["pass_count"]}/{r["total"]}',
            "rsi": r["rsi"],
            "macdHist": r["macd_hist"],
            "volumeRatio": r["volume_ratio"],
            "dataDate": dd,
            "changeRate": cand["chg_rate"]
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

    results.sort(key=lambda x: x["momentum"], reverse=True)
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

@app.route("/api/scan")
def api_scan():
    date_str = flask_request.args.get("date", "")
    demo = flask_request.args.get("demo", "false") == "true"
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        tgt = datetime.strptime(date_str, "%Y-%m-%d")
        if tgt > datetime.now():
            date_str = datetime.now().strftime("%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Date format error"}), 400

    results = run_scan(date_str, demo=demo)
    # Naver API가 해외 서버에서 막힌 경우 자동으로 demo 모드로 재시도
    if not results and not demo:
        print("  [INFO] Naver API returned 0 results. Falling back to demo mode.")
        results = run_scan(date_str, demo=True)
        demo = True
    return jsonify({
        "results": results,
        "date": date_str,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "demo" if demo else "KRX+Naver Pro",
        "count": len(results)
    })

@app.route("/api/status")
def api_status():
    return jsonify(scan_status)

if __name__ == "__main__":
    import os as _os
    _port = int(_os.environ.get("PORT", 5000))
    print("\n" + "=" * 50)
    print(f"  Pro Stock Screener (High-Speed)")
    print(f"  http://localhost:{_port}")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=_port, debug=False)
