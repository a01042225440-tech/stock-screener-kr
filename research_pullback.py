"""
'현재 HUNT(추세추종 근사)' vs '진짜 눌림목(되돌림+지지근접+아래꼬리 반등)' 1년 백테스트.
같은 청산(목표+12%/손절 10일저점·-10%/8일)로 진입조건만 비교.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from app import naver_ohlcv_fast, naver_all_rising_parallel, parse_num, is_excluded_by_name, is_halted
from concurrent.futures import ThreadPoolExecutor
T0 = time.time()
def log(m): print(m, flush=True)
stocks = naver_all_rising_parallel(); uni = []
for s in stocks:
    code = s.get("itemCode", ""); name = s.get("stockName", "")
    if s.get("stockEndType", "") not in ("stock", ""): continue
    if is_halted(s) or is_excluded_by_name(name, code): continue
    cl = parse_num(s.get("closePrice", "0")); trd = parse_num(s.get("accumulatedTradingValue", "0")); mc = parse_num(s.get("marketValue", "0"))
    if cl < 2000 or cl > 200000 or trd < 1000 or mc < 1000: continue
    uni.append((code, trd))
uni.sort(key=lambda x: -x[1]); uni = uni[:350]
ST = {}
def load(u):
    code, trd = u
    df = naver_ohlcv_fast(code, 420, target_date="2026-06-02")
    if df is None or len(df) < 230: return
    s = df["Close"]; c = s.values; o = df["Open"].values; h = df["High"].values; l = df["Low"].values; v = df["Volume"].values
    sma5 = s.rolling(5).mean().values; sma20 = s.rolling(20).mean().values; sma60 = s.rolling(60).mean().values
    d = s.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean(); rsi = (100-100/(1+up/dn)).values
    vma5 = pd.Series(v).rolling(5).mean().values
    ST[code] = dict(trd=trd, idx=df.index, c=c, o=o, h=h, l=l, v=v, sma5=sma5, sma20=sma20, sma60=sma60, rsi=rsi, vma5=vma5)
with ThreadPoolExecutor(max_workers=30) as ex: list(ex.map(load, uni))
log(f"[데이터] {len(ST)}종목  T+{time.time()-T0:.0f}s")
cal = sorted(set().union(*[set(v["idx"]) for v in ST.values()]))
calidx = {d: i for i, d in enumerate(cal)}
S, E = pd.Timestamp("2025-06-02"), pd.Timestamp("2026-06-02")
def nidx(ts): return min(range(len(cal)), key=lambda i: abs((cal[i]-ts).days))
def tk(p, ref):
    for lim, st in [(2000,1),(5000,5),(20000,10),(50000,50),(200000,100),(500000,500)]:
        if ref < lim: return (p//st)*st
    return (p//1000)*1000
def uptrend(d, i):
    return (not np.isnan(d["sma60"][i])) and d["sma5"][i] > d["sma20"][i] > d["sma60"][i] and d["c"][i] > d["sma60"][i]

# A) 현재 HUNT 근사: 정배열 + 20MA위 0~15% + 5MA위 + 양봉 + (거래량1.5x or RSI50-70)
def hunt_now(d, i):
    if not uptrend(d, i): return False
    sma20 = d["sma20"][i]
    if sma20 <= 0: return False
    dist = (d["c"][i]-sma20)/sma20*100
    bull = d["o"][i] < d["c"][i]
    volx = d["v"][i] > d["vma5"][i]*1.5 if d["vma5"][i] > 0 else False
    rsiok = 50 <= d["rsi"][i] <= 70
    return bool(0 <= dist <= 15 and d["c"][i] > d["sma5"][i] and bull and (volx or rsiok))

# B) 진짜 눌림목: 정배열 + 직전고점대비 되돌림 3~15% + 20MA±5% 근접 + 양봉 + 아래꼬리(저가가 지지선 터치 후 회복) + 거래량 반등
def pullback_true(d, i):
    if not uptrend(d, i): return False
    c=d["c"]; o=d["o"]; h=d["h"]; l=d["l"]; sma20=d["sma20"][i]; sma5=d["sma5"][i]
    if sma20 <= 0: return False
    # ② 되돌림: 직전 10일 고점 대비 (고점-오늘저가) 되돌림 3~15%
    hi10 = np.max(h[i-10:i]) if i >= 10 else h[i]
    retr = (hi10 - l[i]) / hi10 * 100 if hi10 > 0 else 0
    pulled = 3 <= retr <= 18
    # ③ 지지선 근접: 종가가 20일선 ±0~6% (눌림 후 지지 부근)
    near = -2 <= (c[i]-sma20)/sma20*100 <= 6
    # ⑤ 양봉 + 아래꼬리(저가가 5MA 혹은 20MA 근처를 찍고 종가는 회복)
    bull = o[i] < c[i]
    rng = h[i]-l[i]
    lowtail = rng > 0 and (min(o[i], c[i]) - l[i])/rng >= 0.25      # 아래꼬리 25%+
    touched = l[i] <= max(sma5, sma20)*1.005                       # 당일 저가가 지지선 터치
    # ④ 거래량 반등(전일 대비 증가)
    volup = d["v"][i] > d["v"][i-1]
    return bool(pulled and near and bull and (lowtail or touched) and volup)

def sim(entry):
    trades = []
    for code, d in ST.items():
        c=d["c"]; h=d["h"]; l=d["l"]; idx=d["idx"]
        pos=False; bp=0; bd=None; bi=0
        for i in range(62, len(c)):
            if not pos:
                try: sig = entry(d, i)
                except Exception: sig = False
                if sig and S <= idx[i] <= E:
                    pos=True; bp=c[i]; bd=idx[i]; bi=i
            else:
                px=c[i]; sell=False; k=i-bi
                tgt = tk(int(bp*1.12), bp); slow = int(l[max(0, bi-10):bi].min()); stp = max(slow, int(bp*0.90))
                if l[i] <= stp: sell=True; px=stp
                elif h[i] >= tgt: sell=True; px=tgt
                elif k >= 8: sell=True
                if sell:
                    if bd in calidx:
                        ei=calidx[bd]; trades.append({"entry":ei,"exit":max(ei+1,nidx(idx[i])),"ret":(px-bp)/bp*100,"pri":(0,-d["trd"])})
                    pos=False
    by={}
    for t in trades: by.setdefault(t["entry"],[]).append(t)
    bal=1.0; op=[]; curve=[1.0]
    for di in range(len(cal)):
        rem=[]
        for p in op:
            if p["xi"]==di: bal*=(1+(p["ret"]-0.3)/100/3)
            else: rem.append(p)
        op=rem
        for sg in sorted(by.get(di,[]), key=lambda x:x["pri"]):
            if len(op)>=3: break
            op.append({"xi":sg["exit"],"ret":sg["ret"]})
        curve.append(bal)
    peak=curve[0]; mdd=0
    for vv in curve: peak=max(peak,vv); mdd=max(mdd,(peak-vv)/peak*100)
    n=len(trades); wr=sum(1 for t in trades if t["ret"]>0)/n*100 if n else 0
    avg=np.mean([t["ret"] for t in trades]) if n else 0
    return (bal-1)*100, n, wr, avg, mdd
log("\n전략                복리       건수  승률  평균  MDD")
for label, fn in [("A) 현재 HUNT(추세추종)", hunt_now), ("B) 진짜 눌림목(되돌림+지지+꼬리)", pullback_true)]:
    comp,n,wr,avg,mdd = sim(fn)
    log(f"  {label:26} {comp:+7.0f}%  n{n:<5} {wr:.0f}%  {avg:+.1f}  {mdd:.0f}%")
    log(f"    T+{time.time()-T0:.0f}s")
log(f"\n소요 {time.time()-T0:.0f}s")
