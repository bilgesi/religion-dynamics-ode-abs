#!/usr/bin/env python3
"""
Find rise-and-fall candidates from Finland raw data.
Identifies rows that show a peak in the middle of the time series.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

RAW = "data/raw/finland/finland_religious_population_1990_2019.csv"
df = pd.read_csv(RAW, sep=None, engine="python")

cat = df.columns[0]
df[cat] = df[cat].astype(str).str.strip()

year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c).strip())]
for c in year_cols:
    df[c] = (df[c].astype(str)
             .str.replace("\u00a0", "", regex=False)
             .str.replace(" ", "", regex=False)
             .str.replace(",", "", regex=False))
    df[c] = pd.to_numeric(df[c], errors="coerce")

def row_contains(s):
    hit = df[df[cat].str.upper().str.contains(s.upper(), na=False)]
    if hit.empty:
        raise ValueError(f"Row not found: {s}")
    return hit.iloc[0][year_cols].to_numpy(dtype=float)

TOTAL = row_contains("TOTAL")
years = np.array([int(c) for c in year_cols])

cands = []
for i, name in enumerate(df[cat].astype(str)):
    name_u = name.upper()
    # Skip TOTAL and top-level categories
    if name_u in ["TOTAL", "CHRISTIANITY_TOTAL"] or name_u == "CHRISTIANITY":
        continue
    series = df.loc[i, year_cols].to_numpy(dtype=float)
    if np.any(np.isnan(series)): 
        continue
    share = series / TOTAL

    peak_idx = int(np.argmax(share))
    peak_year = years[peak_idx]

    # peak in the middle + end lower than peak = rise-and-fall
    if peak_idx < 3 or peak_idx > len(years)-4:
        continue
    if share[-1] >= share[peak_idx]:
        continue

    amplitude = share[peak_idx] - share[0]
    drop = share[peak_idx] - share[-1]
    score = amplitude + drop

    cands.append((score, peak_year, name, float(share[0]), float(share[peak_idx]), float(share[-1])))

cands.sort(reverse=True)
print("=" * 80)
print("Top 10 Rise-and-Fall Candidates (sorted by score = amplitude + drop)")
print("=" * 80)
print(f"{'Row Name':<45} {'Score':<10} {'Peak':<6} {'Start':<10} {'Peak':<10} {'End':<10}")
print("-" * 80)
for s, py, n, y0, yp, yend in cands[:10]:
    print(f"{n:<45} {s:.6f}  {py:<6} {y0:.6f}  {yp:.6f}  {yend:.6f}")

print("\n" + "=" * 80)
print(f"Total candidates found: {len(cands)}")
print("=" * 80)
