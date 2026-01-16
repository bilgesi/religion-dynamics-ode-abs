#!/usr/bin/env python3
"""
Build time series for Finland OTHER RELIGIOUS GROUPS (rise-and-fall pattern).
"""

import pandas as pd
import re
from pathlib import Path

RAW = "data/raw/finland/finland_religious_population_1990_2019.csv"
OUT = "data/raw/finland/finland_risefall_otherreliggroups_1990_2019.csv"

df = pd.read_csv(RAW, sep=None, engine="python")
cat = df.columns[0]
df[cat] = df[cat].astype(str).str.strip()

# year columns
year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c).strip())]
if not year_cols:
    raise ValueError("Year columns not found")

# numeric cleanup
for c in year_cols:
    df[c] = (df[c].astype(str)
             .str.replace("\u00a0", "", regex=False)
             .str.replace(" ", "", regex=False)
             .str.replace(",", "", regex=False))
    df[c] = pd.to_numeric(df[c], errors="coerce")

def get_row_contains(s):
    hit = df[df[cat].str.upper().str.contains(s.upper(), na=False)]
    if hit.empty:
        raise ValueError(f"Row not found containing: {s}")
    return hit.iloc[0][year_cols]

TOTAL = get_row_contains("TOTAL")
ORG = get_row_contains("OTHER_RELIGIOUS_GROUPS_TOTAL")  # This is the name in the CSV

out = pd.DataFrame({
    "year": [int(str(y).strip()) for y in year_cols],
    "obs_share": [float(ORG[y] / TOTAL[y]) for y in year_cols]
}).sort_values("year")

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(f"Wrote {OUT}, rows={len(out)}")
print(f"Peak year = {int(out.loc[out['obs_share'].idxmax(), 'year'])}")
print(f"Share range: {out['obs_share'].min():.6f} - {out['obs_share'].max():.6f}")
