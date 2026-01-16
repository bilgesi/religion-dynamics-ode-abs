"""
Build Finland 4-group time series from raw census data.

Processes raw CSV to create counts and shares for Christianity, Islam, None,
and Other religious groups. Data processing script.
"""
import pandas as pd
import re

INPUT = "finland_religious_population_1990_2019.csv"

OUT_COUNTS_WIDE = "finland_4groups_counts_wide.csv"
OUT_SHARES_LONG = "finland_4groups_shares_long.csv"

# 1) CSV oku
# sep=None => ayrac (virgul/; tab) otomatik tahmin etmeye calisir
df = pd.read_csv(INPUT, sep=None, engine="python")

# 2) Ilk sutun = kategori adi (din grubu)
cat_col = df.columns[0]
df[cat_col] = df[cat_col].astype(str).str.strip()

# 3) Yil sutunlarini bul (1990, 1991, ..., 2019)
year_cols = []
for c in df.columns:
    s = str(c).strip()
    if re.fullmatch(r"\d{4}", s):
        year_cols.append(c)

if not year_cols:
    raise ValueError("Year columns not found. Check CSV headers.")

print(f"Found year columns: {year_cols}")

# 4) Sayilari temizle: "4,998,478" -> 4998478 (int)
for c in year_cols:
    df[c] = (
        df[c].astype(str)
        .str.replace("\u00a0", "", regex=False)  # NBSP (bazen oluyor)
        .str.replace(" ", "", regex=False)
        .str.replace(",", "", regex=False)       # binlik virgul
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 5) Istedigimiz satirlari cekmek icin yardimci fonksiyon
def get_row(exact_label: str) -> pd.Series:
    # Tam eslesme dene
    hit = df[df[cat_col].str.upper() == exact_label.upper()]
    if hit.empty:
        # Bulamazsa "contains" ile arar (kucuk farkliliklara tolerans)
        hit = df[df[cat_col].str.upper().str.contains(exact_label.upper(), na=False)]
    if hit.empty:
        raise ValueError(f"Row not found: {exact_label}")
    return hit.iloc[0][year_cols]

print("\nSearching for rows...")
TOTAL = get_row("TOTAL")
print(f"  [OK] TOTAL found")

# CHRISTIANITY_TOTAL veya CHRISTIANITY'yi ara
try:
    CHRIST = get_row("CHRISTIANITY_TOTAL")
    print(f"  [OK] CHRISTIANITY_TOTAL found")
except:
    CHRIST = get_row("CHRISTIANITY")
    print(f"  [OK] CHRISTIANITY found")

ISLAM = get_row("ISLAM")
print(f"  [OK] ISLAM found")

# CSV'de alt cizgili olabilir, hem deneyelim
try:
    NONE = get_row("PERSONS_NOT_MEMBERS_OF_ANY_RELIGIOUS_COMMUNITY")
    print(f"  [OK] PERSONS_NOT_MEMBERS_OF_ANY_RELIGIOUS_COMMUNITY found")
except:
    NONE = get_row("PERSONS NOT MEMBERS OF ANY RELIGIOUS COMMUNITY")
    print(f"  [OK] PERSONS NOT MEMBERS found")

# 6) OTHER hesapla (residual)
OTHER = (TOTAL - (CHRIST + ISLAM + NONE)).clip(lower=0)

# 7) Wide counts ciktisi (satirlar=grup, sutunlar=yillar)
out_counts = pd.DataFrame({"group": ["christianity", "islam", "none", "other"]})
for y in year_cols:
    out_counts[y] = [CHRIST[y], ISLAM[y], NONE[y], OTHER[y]]

out_counts.to_csv(OUT_COUNTS_WIDE, index=False)

# 8) Long shares ciktisi (year, group, share)
counts_long = out_counts.melt(id_vars="group", var_name="year", value_name="count")
totals_long = pd.DataFrame({"year": year_cols, "total": [TOTAL[y] for y in year_cols]})

merged = counts_long.merge(totals_long, on="year", how="left")
merged["share"] = merged["count"] / merged["total"]

# 9) Kontrol: her yil paylar 1 ediyor mu?
check = merged.groupby("year")["share"].sum().reset_index()
max_dev = (check["share"] - 1.0).abs().max()
print(f"\nCheck: Max |sum(shares)-1| = {float(max_dev):.2e}")

if max_dev > 1e-6:
    print("  [WARNING] Shares sum is not close to 1!")
else:
    print("  [OK] Shares sum is close to 1 (OK)")

merged[["year", "group", "count", "share"]].sort_values(["year", "group"]).to_csv(
    OUT_SHARES_LONG, index=False
)

print(f"\n[OK] Created: {OUT_COUNTS_WIDE}")
print(f"[OK] Created: {OUT_SHARES_LONG}")
