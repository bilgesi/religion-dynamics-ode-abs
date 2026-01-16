#!/usr/bin/env python3
"""
Build New Zealand replacement time series from raw census data.
Creates 3 groups: none, christianity, other
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Input files
RAW_2001 = Path("data/raw/nz/nz_religion_2001.csv")
RAW_2006 = Path("data/raw/nz/nz_religion_2006.csv")
RAW_2013_2023 = Path("data/raw/nz/nz_religion_2013_2018_2023.csv")

# Output files
OUT_DIR = Path("data/processed")
OUT_COUNTS = OUT_DIR / "nz_3groups_counts_wide.csv"
OUT_SHARES = OUT_DIR / "nz_3groups_shares_long.csv"

# Christian keywords for 2013-2023 data
CHRISTIAN_KEYWORDS = [
    "christian", "anglican", "catholic", "presbyter", "methodist", "baptist", 
    "luther", "pentecost", "orthodox", "advent", "salvation", "brethren", 
    "evangelical", "united church", "assemblies of god", "apostolic", 
    "church of", "roman catholic", "seventh-day", "latter-day", "mormon"
]

EXCLUDE_KEYWORDS = [
    "total people", "no religion", "not stated", "object to answering", 
    "not elsewhere included", "not elsewhere classified"
]

def is_christian(affiliation: str) -> bool:
    """Check if affiliation is Christian."""
    aff_lower = affiliation.lower()
    # Exclude first
    for excl in EXCLUDE_KEYWORDS:
        if excl in aff_lower:
            return False
    # Include if matches Christian keyword
    for keyword in CHRISTIAN_KEYWORDS:
        if keyword in aff_lower:
            return True
    return False

def process_2001_2006(csv_path: Path, year: int) -> dict:
    """Process 2001 or 2006 CSV file."""
    df = pd.read_csv(csv_path)
    
    # Get max value per Religious affiliation (national level = max across regions)
    df_max = df.groupby("Religious affiliation")["Value"].max().reset_index()
    
    # Extract values
    total_stated = df_max[df_max["Religious affiliation"] == "Total people stated"]["Value"].iloc[0]
    none = df_max[df_max["Religious affiliation"] == "No Religion"]["Value"].iloc[0]
    
    # Christianity = Christian + Maori Christian
    christian = 0
    if "Christian" in df_max["Religious affiliation"].values:
        christian += df_max[df_max["Religious affiliation"] == "Christian"]["Value"].iloc[0]
    if "Maori Christian" in df_max["Religious affiliation"].values:
        christian += df_max[df_max["Religious affiliation"] == "Maori Christian"]["Value"].iloc[0]
    
    other = total_stated - none - christian
    
    return {
        "year": year,
        "none": int(none),
        "christianity": int(christian),
        "other": int(other),
        "total_stated": int(total_stated)
    }

def process_2013_2023(csv_path: Path) -> list:
    """Process 2013-2023 CSV file."""
    df = pd.read_csv(csv_path)
    
    # Filter for Count unit only
    df_count = df[df["Unit"] == "Count"].copy()
    
    results = []
    years = [2013, 2018, 2023]
    
    for year in years:
        df_year = df_count[df_count["Census Year"] == year].copy()
        
        # Total people stated
        total_row = df_year[df_year["Religious affiliation"].str.contains("Total people stated", case=False, na=False)]
        if len(total_row) == 0:
            raise ValueError(f"Total people stated not found for {year}")
        total_stated = int(total_row["Value"].iloc[0])
        
        # No religion
        none_row = df_year[df_year["Religious affiliation"].str.contains("No religion", case=False, na=False)]
        if len(none_row) == 0:
            raise ValueError(f"No religion not found for {year}")
        none = int(none_row["Value"].iloc[0])
        
        # Christianity: sum all Christian denominations
        christian = 0
        for _, row in df_year.iterrows():
            aff = str(row["Religious affiliation"])
            if is_christian(aff):
                val = row["Value"]
                if pd.notna(val):
                    christian += int(val)
        
        other = total_stated - none - christian
        
        results.append({
            "year": year,
            "none": int(none),
            "christianity": int(christian),
            "other": int(other),
            "total_stated": int(total_stated)
        })
    
    return results

def main():
    print("Building New Zealand replacement time series...")
    print("=" * 70)
    
    all_results = []
    
    # Process 2001
    print(f"\nProcessing {RAW_2001.name}...")
    result_2001 = process_2001_2006(RAW_2001, 2001)
    all_results.append(result_2001)
    print(f"  2001: none={result_2001['none']:,}, christianity={result_2001['christianity']:,}, other={result_2001['other']:,}, total={result_2001['total_stated']:,}")
    
    # Process 2006
    print(f"\nProcessing {RAW_2006.name}...")
    result_2006 = process_2001_2006(RAW_2006, 2006)
    all_results.append(result_2006)
    print(f"  2006: none={result_2006['none']:,}, christianity={result_2006['christianity']:,}, other={result_2006['other']:,}, total={result_2006['total_stated']:,}")
    
    # Process 2013-2023
    print(f"\nProcessing {RAW_2013_2023.name}...")
    results_2013_2023 = process_2013_2023(RAW_2013_2023)
    for r in results_2013_2023:
        all_results.append(r)
        print(f"  {r['year']}: none={r['none']:,}, christianity={r['christianity']:,}, other={r['other']:,}, total={r['total_stated']:,}")
    
    # Sort by year
    all_results.sort(key=lambda x: x["year"])
    
    # Create counts wide format
    counts_wide = pd.DataFrame({
        "year": [r["year"] for r in all_results],
        "none": [r["none"] for r in all_results],
        "christianity": [r["christianity"] for r in all_results],
        "other": [r["other"] for r in all_results],
        "total_stated": [r["total_stated"] for r in all_results]
    })
    
    # Create shares long format
    shares_long = []
    for r in all_results:
        total = r["total_stated"]
        for group in ["none", "christianity", "other"]:
            count = r[group]
            share = count / total
            shares_long.append({
                "year": r["year"],
                "group": group,
                "count": count,
                "share": share
            })
    
    shares_df = pd.DataFrame(shares_long)
    
    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    # Check: (none + christianity + other) == total_stated
    for r in all_results:
        sum_groups = r["none"] + r["christianity"] + r["other"]
        diff = abs(sum_groups - r["total_stated"])
        if diff > 1:
            print(f"WARNING: {r['year']}: sum(groups)={sum_groups}, total={r['total_stated']}, diff={diff}")
    
    max_diff = max([abs(r["none"] + r["christianity"] + r["other"] - r["total_stated"]) for r in all_results])
    print(f"Max |(none+christianity+other) - total_stated| = {max_diff}")
    
    # Check: sum(shares) == 1
    for year in [r["year"] for r in all_results]:
        year_shares = shares_df[shares_df["year"] == year]["share"]
        sum_shares = year_shares.sum()
        diff = abs(sum_shares - 1.0)
        if diff > 1e-6:
            print(f"WARNING: {year}: sum(shares)={sum_shares:.6f}, diff={diff:.2e}")
    
    max_share_diff = max([abs(shares_df[shares_df["year"] == y]["share"].sum() - 1.0) 
                          for y in [r["year"] for r in all_results]])
    print(f"Max |sum(shares) - 1| = {max_share_diff:.2e}")
    
    # Print replacement series
    print("\n" + "=" * 70)
    print("REPLACEMENT SERIES (none_share and christianity_share)")
    print("=" * 70)
    for r in all_results:
        none_share = r["none"] / r["total_stated"]
        christ_share = r["christianity"] / r["total_stated"]
        print(f"  {r['year']}: none={none_share:.4f}, christianity={christ_share:.4f}")
    
    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts_wide.to_csv(OUT_COUNTS, index=False)
    shares_df.to_csv(OUT_SHARES, index=False)
    
    print("\n" + "=" * 70)
    print(f"Saved: {OUT_COUNTS}")
    print(f"Saved: {OUT_SHARES}")
    print("=" * 70)

if __name__ == "__main__":
    main()
