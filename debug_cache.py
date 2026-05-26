"""OHLCV fetch 문제 진단"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app import naver_ohlcv_fast
import pandas as pd

CODE = "264450"  # 유비쿼스
print("\n=== 유비쿼스(264450) OHLCV 진단 ===\n")

# target_date 없이
print("[1] target_date 없이 (최신):")
df1 = naver_ohlcv_fast(CODE, 250)
print(f"   len={len(df1)}, 첫일={df1.index[0].date()}, 마지막={df1.index[-1].date()}")
print(f"   마지막 3일:")
for idx, row in df1.tail(3).iterrows():
    print(f"     {idx.date()}  O={row['Open']:.0f} H={row['High']:.0f} L={row['Low']:.0f} C={row['Close']:.0f} V={row['Volume']:.0f}")

# target_date="2026-05-22"
print("\n[2] target_date='2026-05-22':")
df2 = naver_ohlcv_fast(CODE, 250, target_date="2026-05-22")
print(f"   len={len(df2)}, 첫일={df2.index[0].date()}, 마지막={df2.index[-1].date()}")
print(f"   마지막 5일:")
for idx, row in df2.tail(5).iterrows():
    print(f"     {idx.date()}  O={row['Open']:.0f} C={row['Close']:.0f}")

# target_date="2026-05-25"
print("\n[3] target_date='2026-05-25':")
df3 = naver_ohlcv_fast(CODE, 250, target_date="2026-05-25")
print(f"   len={len(df3)}, 첫일={df3.index[0].date()}, 마지막={df3.index[-1].date()}")
print(f"   마지막 5일:")
for idx, row in df3.tail(5).iterrows():
    print(f"     {idx.date()}  O={row['Open']:.0f} C={row['Close']:.0f}")

print("\n[4] 슬라이싱 테스트 (5/22까지):")
df_sliced = df2[df2.index <= pd.Timestamp("2026-05-22")]
print(f"   슬라이싱 후 len={len(df_sliced)}, 마지막={df_sliced.index[-1].date()}")
print(f"   마지막 3일:")
for idx, row in df_sliced.tail(3).iterrows():
    print(f"     {idx.date()}  O={row['Open']:.0f} C={row['Close']:.0f}")

print("\n[5] 슬라이싱 테스트 (5/25까지):")
df_sliced2 = df3[df3.index <= pd.Timestamp("2026-05-25")]
print(f"   슬라이싱 후 len={len(df_sliced2)}, 마지막={df_sliced2.index[-1].date()}")
print(f"   마지막 3일:")
for idx, row in df_sliced2.tail(3).iterrows():
    print(f"     {idx.date()}  O={row['Open']:.0f} C={row['Close']:.0f}")
