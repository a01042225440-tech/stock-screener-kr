"""풍산(103140) 코일스프링 패턴 검증
2026-04-06 폭발일과 그 직전 며칠을 각각 검사해본다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import naver_ohlcv_fast, screen_pro
import pandas as pd
from datetime import datetime, timedelta

CODE = "103140"
NAME = "풍산"
MCAP_HINT = 5000  # 대략 (실제값은 phase2에서 다른 경로로 옴 - 여기선 충족 가정)

print(f"\n{'='*70}")
print(f"  풍산(103140) 코일스프링 검증")
print(f"{'='*70}\n")

# 250일치 데이터 가져오기 (2026-04-10 기준으로 fetch)
df = naver_ohlcv_fast(CODE, days=320, target_date="2026-04-10")
if df is None:
    print("[FAIL] OHLCV fetch failed")
    sys.exit(1)

print(f"데이터 범위: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} rows)\n")

# 폭발일 전후 5일치 출력
print("최근 8영업일 가격/거래량:")
for d in df.tail(8).itertuples():
    print(f"  {d.Index.date()}  C={d.Close:>8,}  V={d.Volume:>12,}")
print()

# 각 날짜에 대해 screen_pro 실행
test_dates = ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-06", "2026-04-07", "2026-04-08"]
for td in test_dates:
    try:
        ts = pd.Timestamp(td)
    except:
        continue
    sub = df[df.index <= ts]
    if len(sub) < 201:
        print(f"[{td}] insufficient data ({len(sub)})")
        continue
    r = screen_pro(sub, NAME, CODE, MCAP_HINT)
    if r is None:
        print(f"[{td}] screen_pro=None (탈락)")
        continue
    grade = r.get("grade", "-")
    p1 = r.get("p1_score", r.get("p1Score", 0))
    p2 = r.get("phase2_pass", False)
    vm = r.get("volMult", 0)
    ct = r.get("chgToday", 0)
    bw = r.get("bbWidthPct", 0)
    box = r.get("boxRange", 0)
    av = r.get("annualVol", 0)
    vtr = r.get("volTrendRatio", 0)
    mn = r.get("minerviniCount", 0)
    sec = r.get("sector", "-")
    sb = r.get("sectorBonus", 0)
    fs = r.get("finalScore", 0)
    print(f"[{td}] grade={grade:<7} P1={p1}/10 P2={p2}  vol×{vm:>4.1f}  당일{ct:+.1f}%  BB폭{bw:.1f}%  박스{box:.1f}%  연변동{av:.0f}%  vol추세{vtr:.2f}  Minervini={mn}/5  Sec={sec}+{sb}  Final={fs:.0f}")
    if r.get("p1_details"):
        print(f"           P1획득: {', '.join(r['p1_details'])}")

print(f"\n{'='*70}")
