"""인바디(041830) 2026-05-11 매수 → 1주일 +10% 청산 검증"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import naver_ohlcv_fast, screen_pro, fetch_investor_data
import pandas as pd

CODE = "041830"
NAME = "인바디"
BUY_DATE = "2026-05-11"

print(f"\n{'='*72}")
print(f"  인바디({CODE}) {BUY_DATE} 매수 검증")
print(f"{'='*72}\n")

# 250일치 OHLCV (5/11 이후 7영업일까지 포함)
df = naver_ohlcv_fast(CODE, days=320, target_date="2026-06-01")
if df is None:
    print("[FAIL] OHLCV 가져오기 실패")
    sys.exit(1)

print(f"데이터 범위: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} 영업일)\n")

# 5/11 기준 screen_pro 실행
df_at_buy = df[df.index <= pd.Timestamp(BUY_DATE)]
if len(df_at_buy) < 201:
    print(f"[SKIP] 5/11 기준 데이터 부족: {len(df_at_buy)}")
    sys.exit(1)

# 시총/펀더 가져오기
inv = fetch_investor_data(CODE)
mcap_hint = 5000  # 추정 (인바디 시총 약 5천억대)
fundamental = None
if inv:
    fundamental = {
        "per": inv.get("per", 0),
        "pbr": inv.get("pbr", 0),
        "eps": inv.get("eps", 0),
        "target_price": inv.get("target_price", 0),
        "sector_ratio": inv.get("sector_ratio", 50),
    }
    print("펀더멘털:")
    print(f"  EPS={inv.get('eps',0):.0f}  PER={inv.get('per',0):.1f}  PBR={inv.get('pbr',0):.2f}  "
          f"목표가={inv.get('target_price',0):.0f}  업종강도={inv.get('sector_ratio',0):.0f}%\n")

r = screen_pro(df_at_buy, NAME, CODE, mcap_hint, fundamental=fundamental)
if r is None:
    print("[탈락] screen_pro에서 제외됨")
    sys.exit(0)

print(f"등급: {r.get('grade')}")
print(f"A 정배열: {r.get('aCount')}/4  (A1>200MA={r.get('a1_ma200')}, A2>60MA={r.get('a2_ma60')}, "
      f"A3정배열={r.get('a3_align')}, A4MA200상승={r.get('a4_ma200up')})")
print(f"B 펀더: {r.get('bCount')}/3  (B1EPS+={r.get('b1_eps')}, B2PER적정={r.get('b2_per')}, "
      f"B3목표가={r.get('b3_target')})")
print(f"K 캔들: K1={r.get('k1_up_close')} K2={r.get('k2_bullish')} K3={r.get('k3_short_wick')} "
      f"→ candlePass={r.get('candlePass')}")
print(f"섹터: {r.get('sector')}+{r.get('sectorBonus')}  업종강도={r.get('sectorRatio')}%")
print(f"HUNT 트리거={r.get('huntTrigger')}  BREAKOUT 트리거={r.get('breakoutTrigger')}")
print(f"20MA거리={r.get('ma20Dist')}%  거래량(5일)={r.get('volMult5')}x  RSI={r.get('rsi')}")
print(f"52주 위치={r.get('proximity52w')}%  목표가+{r.get('targetUpside')}%")
print(f"최종점수={r.get('finalScore')}")

if r.get('grade') not in ('HUNT', 'BREAKOUT', 'BB_BREAK'):
    print(f"\n→ 인바디는 {BUY_DATE}에 매수후보가 아니었음 (등급: {r.get('grade')})")
    print(f"  현재 시스템 기준으로 5/11 매수 대상은 아님")

# 5/11 종가 = 매수가
buy_row = df.loc[pd.Timestamp(BUY_DATE)] if pd.Timestamp(BUY_DATE) in df.index else None
if buy_row is None:
    # 가장 가까운 영업일 찾기
    nearest = df[df.index <= pd.Timestamp(BUY_DATE)].iloc[-1]
    print(f"\n⚠️ {BUY_DATE}는 비영업일. 가장 가까운 영업일: {df[df.index <= pd.Timestamp(BUY_DATE)].index[-1].date()}")
    buy_row = nearest
    actual_buy_date = df[df.index <= pd.Timestamp(BUY_DATE)].index[-1]
else:
    actual_buy_date = pd.Timestamp(BUY_DATE)

buy_price = int(buy_row["Close"])
target_price = int(buy_price * 1.10)
stop_price = int(buy_price * 0.93)

print(f"\n{'─'*72}")
print(f"매수 시뮬레이션")
print(f"{'─'*72}")
print(f"매수일: {actual_buy_date.date()}  매수가(당일 종가): {buy_price:,}원")
print(f"🎯 청산가 (+10%): {target_price:,}원")
print(f"🛑 손절가 (-7%):  {stop_price:,}원")
print(f"최대 보유: 5영업일")

# 매수일 다음날부터 5영업일 추적
df_after = df[df.index > actual_buy_date].head(5)
print(f"\n{'─'*72}")
print(f"매수 후 5영업일 추적")
print(f"{'─'*72}")
print(f"{'날짜':<12} {'시가':>9} {'고가':>9} {'저가':>9} {'종가':>9} {'고가도달':<10} {'종가수익':>9}")
print(f"{'─'*72}")

exit_day = None
exit_reason = None
exit_price = None

for i, (idx, row) in enumerate(df_after.iterrows(), 1):
    o, h, l, c = int(row["Open"]), int(row["High"]), int(row["Low"]), int(row["Close"])
    high_pct = (h - buy_price) / buy_price * 100
    close_pct = (c - buy_price) / buy_price * 100
    high_target = h >= target_price
    low_stop = l <= stop_price
    mark = ""
    if exit_day is None:
        if high_target:
            exit_day = idx
            exit_reason = "🎯 +10% 청산 도달"
            exit_price = target_price
            mark = " ← 🎯 청산!"
        elif low_stop:
            exit_day = idx
            exit_reason = "🛑 손절"
            exit_price = stop_price
            mark = " ← 🛑 손절"
        elif i == 5:
            exit_day = idx
            exit_reason = "⏰ 5영업일 만기 → 종가 청산"
            exit_price = c
            mark = " ← ⏰ 만기 청산"
    print(f"{str(idx.date()):<12} {o:>9,} {h:>9,} {l:>9,} {c:>9,} 고가{high_pct:>+5.1f}%  종가{close_pct:>+5.1f}%{mark}")

print(f"\n{'='*72}")
print(f"  📊 최종 결과")
print(f"{'='*72}")
if exit_day:
    profit = exit_price - buy_price
    profit_pct = profit / buy_price * 100
    print(f"청산일: {exit_day.date()}")
    print(f"청산 사유: {exit_reason}")
    print(f"매수가: {buy_price:,}원 → 청산가: {exit_price:,}원")
    print(f"수익률: {profit_pct:+.2f}% ({profit:+,}원)")
    if profit_pct >= 10:
        print(f"\n✅ 1주일 안 +10% 목표 달성")
    elif profit_pct >= 0:
        print(f"\n🟡 +10% 미달성, 손익분기 위 만기 청산")
    elif profit_pct >= -7:
        print(f"\n🟠 손실 만기 청산 (손절선 미터치)")
    else:
        print(f"\n🔴 손절")
else:
    print("(추적 불가)")

# 비교용: 1주일 후 최고가
if len(df_after) >= 5:
    max_high_pct = (df_after["High"].max() - buy_price) / buy_price * 100
    print(f"\n참고: 5영업일 내 최고가 = {int(df_after['High'].max()):,}원 (매수가 대비 +{max_high_pct:.2f}%)")
