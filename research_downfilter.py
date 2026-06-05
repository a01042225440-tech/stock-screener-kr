"""
'파랑(어제대비 하락) 종목 매수 제외' 필터가 수익을 개선하는가.
모멘텀(신고가/돌파)·스윙(BB+RSI) 둘 다, 여러 최소등락률 임계값으로 1년 백테스트.
floor=0  → 오늘 양봉(+0% 이상)만 매수 / floor=-3 → -3%까지는 허용 / None → 필터없음
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
    sd = s.rolling(20).std(); bl = (s.rolling(20).mean()-2*sd).values; bu = (s.rolling(20).mean()+2*sd).values
    d = s.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean(); rsi = (100-100/(1+up/dn))
    rsig = rsi.rolling(9).mean().values; rsi = rsi.values
    vma20 = pd.Series(v).rolling(20).mean().values; hi20 = pd.Series(h).rolling(20).max().shift(1).values
    daychg = (s/s.shift(1)-1).values*100
    ST[code] = dict(trd=trd, idx=df.index, c=c, o=o, h=h, l=l, v=v, bl=bl, bu=bu, rsi=rsi, rsig=rsig, vma20=vma20, hi20=hi20, daychg=daychg)
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
def bull(d, i): return d["o"][i] < d["c"][i]
def swing_entry(d, i):
    return (not np.isnan(d["bl"][i-1])) and d["c"][i-1] <= d["bl"][i-1] and d["c"][i] > d["bl"][i] and bull(d, i) and (d["h"][i]-d["l"][i] > 0) and (d["c"][i]-d["l"][i])/(d["h"][i]-d["l"][i]) >= 0.5 and (not np.isnan(d["rsig"][i-1])) and d["rsi"][i] > d["rsig"][i] and d["rsi"][i-1] <= d["rsig"][i-1]
def mom_entry(d, i):
    return (not np.isnan(d["hi20"][i])) and d["c"][i] >= d["hi20"][i] and d["v"][i] >= d["vma20"][i]*1.5 and bull(d, i)
def sim(entry, exit_mode, floor):
    trades = []
    for code, d in ST.items():
        c = d["c"]; h = d["h"]; l = d["l"]; idx = d["idx"]; bl = d["bl"]; bu = d["bu"]; dc = d["daychg"]
        pos = False; bp = 0; bd = None; bi = 0
        for i in range(25, len(c)):
            if not pos:
                try: sig = entry(d, i)
                except Exception: sig = False
                if sig and (floor is None or (not np.isnan(dc[i]) and dc[i] >= floor)) and S <= idx[i] <= E:
                    pos = True; bp = c[i]; bd = idx[i]; bi = i
            else:
                px = c[i]; sell = False; k = i-bi
                if exit_mode == "MR":
                    if px <= bp*0.90: sell = True
                    elif (not np.isnan(bl[i])) and px < bl[i]: sell = True
                    elif (not np.isnan(bu[i])) and px >= bu[i]: sell = True
                    elif (idx[i]-bd).days >= 58: sell = True
                else:
                    tgt = tk(int(bp*1.12), bp); slow = int(l[max(0, bi-10):bi].min()); stp = max(slow, int(bp*0.90))
                    if l[i] <= stp: sell = True; px = stp
                    elif h[i] >= tgt: sell = True; px = tgt
                    elif k >= 8: sell = True
                if sell:
                    if bd in calidx:
                        ei = calidx[bd]; trades.append({"entry": ei, "exit": max(ei+1, nidx(idx[i])), "ret": (px-bp)/bp*100, "pri": (0, -d["trd"])})
                    pos = False
    by = {}
    for t in trades: by.setdefault(t["entry"], []).append(t)
    bal = 1.0; op = []
    for di in range(len(cal)):
        rem = []
        for p in op:
            if p["xi"] == di: bal *= (1+(p["ret"]-0.3)/100/3)
            else: rem.append(p)
        op = rem
        for sg in sorted(by.get(di, []), key=lambda x: x["pri"]):
            if len(op) >= 3: break
            op.append({"xi": sg["exit"], "ret": sg["ret"]})
    n = len(trades); wr = sum(1 for t in trades if t["ret"] > 0)/n*100 if n else 0
    avg = np.mean([t["ret"] for t in trades]) if n else 0
    return (bal-1)*100, n, wr, avg
log("\n하락제외필터(어제대비 N% 미만 매수제외)  모멘텀(신고가)복리       스윙(BB+RSI)복리")
for fl in [None, -5, -3, -1, 0, 1]:
    label = "없음(파랑허용)" if fl is None else (f"+{fl}% 이상만" if fl > 0 else (f"{fl}% 이상만" if fl < 0 else "0%이상(빨강만)"))
    mc, mn, mw, ma = sim(mom_entry, "TF", fl)
    sc, sn, sw_, sa = sim(swing_entry, "MR", fl)
    log(f"  {label:16} {mc:+8.0f}%(n{mn},승{mw:.0f},평{ma:+.1f})   {sc:+8.0f}%(n{sn},승{sw_:.0f},평{sa:+.1f})")
    log(f"    T+{time.time()-T0:.0f}s")
log(f"\n소요 {time.time()-T0:.0f}s")
