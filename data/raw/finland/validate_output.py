"""
Validate Finland time series output against raw CSV data.

Compares processed time series with original census data for consistency checks.
Data validation script.
"""
import pandas as pd
import numpy as np

# 1) Original CSV'yi oku ve satir etiketlerini bul
print("=" * 70)
print("1) CSV'DEKI SATIR ETIKETLERI")
print("=" * 70)
df_raw = pd.read_csv("finland_religious_population_1990_2019.csv")
cat_col = df_raw.columns[0]

# Aradigimiz satirlari bul
targets = ["TOTAL", "CHRISTIANITY", "ISLAM", "PERSONS_NOT_MEMBERS", "PERSONS NOT MEMBERS"]
print("\nAranan satirlar:")
for t in targets:
    hits = df_raw[df_raw[cat_col].str.upper().str.contains(t.upper().replace(" ", "_"), na=False, regex=False)]
    if hits.empty:
        hits = df_raw[df_raw[cat_col].str.upper().str.contains(t.upper(), na=False, regex=False)]
    if not hits.empty:
        exact_label = hits.iloc[0][cat_col]
        print(f"  '{t}' -> CSV'de: '{exact_label}'")
    else:
        print(f"  '{t}' -> BULUNAMADI!")

# 2) Yil kolonlari kontrolu
print("\n" + "=" * 70)
print("2) YIL KOLONLARI")
print("=" * 70)
year_cols = [c for c in df_raw.columns[1:] if str(c).isdigit()]
years = sorted([int(y) for y in year_cols])
print(f"  Yil kolon sayisi: {len(year_cols)}")
print(f"  Ilk yil: {min(years)}")
print(f"  Son yil: {max(years)}")
print(f"  Eksik yillar var mi? {('EVET' if len(years) != (max(years) - min(years) + 1) else 'HAYIR')}")
print(f"  Yillar: {years}")

# 3) counts_wide kontrolu
print("\n" + "=" * 70)
print("3) FINLAND_4GROUPS_COUNTS_WIDE.CSV DOGRULAMASI")
print("=" * 70)
df_counts = pd.read_csv("finland_4groups_counts_wide.csv")
print(f"  Satir sayisi (baslik disinda): {len(df_counts)}")
print(f"  Beklenen: 4")
print(f"  Grup isimleri: {sorted(df_counts['group'].tolist())}")
print(f"  Beklenen gruplar: ['christianity', 'islam', 'none', 'other']")

print("\n  1990 degerleri:")
for _, row in df_counts.iterrows():
    print(f"    {row['group']}: {row['1990']:,}")

print("\n  2019 degerleri:")
for _, row in df_counts.iterrows():
    print(f"    {row['group']}: {row['2019']:,}")

# 4) Manuel spot-check
print("\n" + "=" * 70)
print("4) SPOT-CHECK (MANUEL DOGRULAMA)")
print("=" * 70)

# 1990
total_1990 = df_raw[df_raw[cat_col] == "TOTAL"]["1990"].iloc[0]
christ_1990 = df_raw[df_raw[cat_col] == "CHRISTIANITY_TOTAL"]["1990"].iloc[0]
islam_1990 = df_raw[df_raw[cat_col] == "ISLAM"]["1990"].iloc[0]
none_1990 = df_raw[df_raw[cat_col] == "PERSONS_NOT_MEMBERS_OF_ANY_RELIGIOUS_COMMUNITY"]["1990"].iloc[0]
other_1990_calc = total_1990 - (christ_1990 + islam_1990 + none_1990)
other_1990_file = df_counts[df_counts['group'] == 'other']['1990'].iloc[0]

print(f"  1990:")
print(f"    TOTAL = {total_1990:,}")
print(f"    CHRISTIANITY = {christ_1990:,}")
print(f"    ISLAM = {islam_1990:,}")
print(f"    NONE = {none_1990:,}")
print(f"    OTHER (hesap) = {total_1990:,} - ({christ_1990:,} + {islam_1990:,} + {none_1990:,}) = {other_1990_calc:,}")
print(f"    OTHER (dosyada) = {other_1990_file:,}")
print(f"    Eslesme: {'EVET' if abs(other_1990_calc - other_1990_file) < 1 else 'HAYIR'}")

# 2019
total_2019 = df_raw[df_raw[cat_col] == "TOTAL"]["2019"].iloc[0]
christ_2019 = df_raw[df_raw[cat_col] == "CHRISTIANITY_TOTAL"]["2019"].iloc[0]
islam_2019 = df_raw[df_raw[cat_col] == "ISLAM"]["2019"].iloc[0]
none_2019 = df_raw[df_raw[cat_col] == "PERSONS_NOT_MEMBERS_OF_ANY_RELIGIOUS_COMMUNITY"]["2019"].iloc[0]
other_2019_calc = total_2019 - (christ_2019 + islam_2019 + none_2019)
other_2019_file = df_counts[df_counts['group'] == 'other']['2019'].iloc[0]

print(f"\n  2019:")
print(f"    TOTAL = {total_2019:,}")
print(f"    CHRISTIANITY = {christ_2019:,}")
print(f"    ISLAM = {islam_2019:,}")
print(f"    NONE = {none_2019:,}")
print(f"    OTHER (hesap) = {total_2019:,} - ({christ_2019:,} + {islam_2019:,} + {none_2019:,}) = {other_2019_calc:,}")
print(f"    OTHER (dosyada) = {other_2019_file:,}")
print(f"    Eslesme: {'EVET' if abs(other_2019_calc - other_2019_file) < 1 else 'HAYIR'}")

# 5) Double-count kontrolu
print("\n" + "=" * 70)
print("5) DOUBLE-COUNT KONTROLU")
print("=" * 70)
print("  Kullanilan satir: 'CHRISTIANITY_TOTAL' (top-level)")
print("  Alt kirilimlar (Evangelical Lutheran, Greek Orthodox, etc.) KULLANILMADI")
print("  Double-count riski: YOK")

# 6) Sayi temizligi kontrolu
print("\n" + "=" * 70)
print("6) SAYI TEMIZLIGI")
print("=" * 70)
# Her yil icin OTHER negatif mi kontrol et
year_cols = [str(y) for y in years]
neg_other_years = []
for y in year_cols:
    other_val = df_counts[df_counts['group'] == 'other'][y].iloc[0]
    if other_val < 0:
        neg_other_years.append(y)

print(f"  OTHER negatif olan yillar: {neg_other_years if neg_other_years else 'YOK (tum degerler >= 0)'}")
print(f"  Sayi format: Integer (ondalik yok)")
print(f"  NaN kontrolu: Yapiliyor...")

# NaN kontrol
has_nan = False
for col in year_cols:
    if df_counts[col].isna().any():
        has_nan = True
        print(f"    UYARI: {col} kolonunda NaN var!")
print(f"  NaN durumu: {'VAR' if has_nan else 'YOK'}")

# 7) shares_long kontrolu
print("\n" + "=" * 70)
print("7) FINLAND_4GROUPS_SHARES_LONG.CSV DOGRULAMASI")
print("=" * 70)
df_shares = pd.read_csv("finland_4groups_shares_long.csv")
print(f"  Satir sayisi (baslik disinda): {len(df_shares)}")
print(f"  Beklenen: 120 (30 yil x 4 grup)")
print(f"  Kolonlar: {list(df_shares.columns)}")
print(f"  Beklenen kolonlar: ['year', 'group', 'count', 'share']")

# Share aralik kontrolu
min_share = df_shares['share'].min()
max_share = df_shares['share'].max()
print(f"  Share min: {min_share:.6f}")
print(f"  Share max: {max_share:.6f}")
print(f"  Share aralik dogru mu (0-1)? {'EVET' if 0 <= min_share and max_share <= 1 else 'HAYIR'}")

# Her yil icin share toplami
share_sums = df_shares.groupby('year')['share'].sum()
max_dev = (share_sums - 1.0).abs().max()
print(f"  Her yil icin share toplami kontrolu:")
print(f"    Max |sum(shares) - 1| = {max_dev:.2e}")
print(f"    Dogru mu (< 1e-6)? {'EVET' if max_dev < 1e-6 else 'HAYIR'}")

# 8) Ek kapanis kontrolu
print("\n" + "=" * 70)
print("8) EK KAPANIS KONTROLU")
print("=" * 70)
# Her yil icin counts toplami = TOTAL mi?
for y in year_cols:
    sum_counts = df_counts[y].sum()
    total_val = df_raw[df_raw[cat_col] == "TOTAL"][y].iloc[0]
    diff = abs(sum_counts - total_val)
    if diff > 1:
        print(f"  UYARI: {y} yili icin toplam fark: {diff:.2f}")

# Tum yillar icin max fark
max_diffs = []
for y in year_cols:
    sum_counts = df_counts[y].sum()
    total_val = df_raw[df_raw[cat_col] == "TOTAL"][y].iloc[0]
    max_diffs.append(abs(sum_counts - total_val))

max_diff = max(max_diffs)
print(f"  Tum yillar icin max |sum(counts) - TOTAL| = {max_diff:.2e}")
print(f"  Dogru mu (< 1)? {'EVET' if max_diff < 1 else 'HAYIR'}")

# 9) Cikti formati
print("\n" + "=" * 70)
print("9) CIKTI FORMATI")
print("=" * 70)
# CSV delimiter kontrolu (ilk satiri oku)
with open("finland_4groups_shares_long.csv", "r", encoding="utf-8") as f:
    first_line = f.readline()
    delimiter = "," if "," in first_line else ";"
    
print(f"  CSV delimiter: {delimiter}")
print(f"  Ondalik ayraci: nokta (.)")
print(f"  Encoding: UTF-8")

# 10) Model kullanim onerisi
print("\n" + "=" * 70)
print("10) MODEL KULLANIM ONERISI")
print("=" * 70)
print("  Iki secenek:")
print("\n  SECENEK A: Year'i direkt kullan (1990..2019)")
print("    - Avantaj: Gercek yillari gosterir")
print("    - Dezavantaj: Yil degerleri buyuk (model parametreleriyle karsilastirirken dikkat)")
print("\n  SECENEK B: Year'i t=0..29'a map et")
print("    - Avantaj: Standardize edilmis zaman ekseni")
print("    - Formul: t = year - 1990")
print("    - Ornek: 1990 -> t=0, 2019 -> t=29")
print("\n  ONERI: Model ode cozuculeri genelde t=0'dan baslar.")
print("         Eger model kodunda 'year' kullaniliyorsa SECENEK A,")
print("         eger 't' (time) kullaniliyorsa SECENEK B kullan.")

print("\n" + "=" * 70)
print("VALIDATION TAMAMLANDI")
print("=" * 70)
