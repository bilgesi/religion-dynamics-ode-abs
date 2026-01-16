#!/usr/bin/env python3
"""
Merge NZ religion data from 2001, 2006, 2013, 2018, 2023 into unified time series.

Output: data/nz/timeseries/nz_none_share_2001_2006_2013_2018_2023.csv
Columns: year, share_none (fraction)
"""

import pandas as pd
from pathlib import Path


def extract_share_from_file(csv_path: Path, year: int) -> float:
    """
    Extract 'No religion' share from a CSV file.
    
    Logic:
    1. Prefer percentage if Unit == "Percentage of total stated"
    2. If only counts exist, compute share = no_religion_count / total_stated_count
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Filter for 'No religion' / 'No Religion'
    no_religion_patterns = ['No religion', 'No Religion']
    no_religion_mask = df['Religious affiliation'].str.contains(
        '|'.join(no_religion_patterns), case=False, na=False
    )
    
    # Check if percentage exists
    if 'Unit' in df.columns:
        pct_mask = df['Unit'].str.contains('Percentage', case=False, na=False)
        no_religion_pct = df[no_religion_mask & pct_mask]
        
        if len(no_religion_pct) > 0:
            # Use percentage (might be at regional level, sum if multiple)
            values = no_religion_pct['Value'].dropna()
            if len(values) > 0:
                # If multiple rows, take the first (should be national level)
                # Or sum if needed - but usually national data comes first
                pct_value = values.iloc[0] if len(values) == 1 else values.sum()
                share = float(pct_value) / 100.0
                print(f"  {year}: Found percentage = {pct_value:.2f}% -> share = {share:.4f}")
                return share
    
    # If no percentage, compute from counts
    no_religion_counts = df[no_religion_mask]
    
    if len(no_religion_counts) == 0:
        raise ValueError(f"No 'No religion' rows found in {csv_path}")
    
    # Sum across all regions/territories if data is disaggregated
    total_no_religion = no_religion_counts['Value'].sum()
    print(f"  {year}: No religion count = {total_no_religion:,.0f}")
    
    # Find 'Total people stated' or 'Total stated'
    total_patterns = ['Total people stated', 'Total stated', 'Total']
    total_mask = df['Religious affiliation'].str.contains(
        '|'.join(total_patterns), case=False, na=False
    )
    
    # Filter to avoid matching other "Total" categories
    # Look for the row that represents total population with stated religion
    total_stated = df[total_mask]
    
    if len(total_stated) == 0:
        raise ValueError(f"No 'Total people stated' row found in {csv_path}")
    
    # Sum across all regions if disaggregated
    total_stated_value = total_stated['Value'].sum()
    print(f"  {year}: Total stated count = {total_stated_value:,.0f}")
    
    share = float(total_no_religion) / float(total_stated_value)
    print(f"  {year}: Computed share = {share:.4f} ({share*100:.2f}%)")
    
    return share


def main():
    base_dir = Path(".")
    raw_dir = base_dir / "data" / "raw" / "nz"
    timeseries_dir = base_dir / "data" / "nz" / "timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Process 2001
    print("\nProcessing 2001...")
    csv_2001 = raw_dir / "nz_religion_2001.csv"
    if csv_2001.exists():
        share_2001 = extract_share_from_file(csv_2001, 2001)
        results.append({"year": 2001, "share_none": share_2001})
    else:
        print(f"WARNING: {csv_2001} not found, skipping 2001")
    
    # Process 2006
    print("\nProcessing 2006...")
    csv_2006 = raw_dir / "nz_religion_2006.csv"
    if csv_2006.exists():
        share_2006 = extract_share_from_file(csv_2006, 2006)
        results.append({"year": 2006, "share_none": share_2006})
    else:
        print(f"WARNING: {csv_2006} not found, skipping 2006")
    
    # Process 2013, 2018, 2023 from the existing CSV
    print("\nProcessing 2013, 2018, 2023...")
    csv_2013_2023 = raw_dir / "nz_religion_2013_2018_2023.csv"
    if not csv_2013_2023.exists():
        print(f"ERROR: {csv_2013_2023} not found", file=sys.stderr)
        return
    
    df_2013_2023 = pd.read_csv(csv_2013_2023, encoding='utf-8-sig')
    
    # Filter for 'No religion' and percentage unit
    no_religion_mask = df_2013_2023['Religious affiliation'].str.contains(
        'No religion', case=False, na=False
    )
    
    # Check subject population filter (same as nz_build_timeseries.py)
    if 'Subject population' in df_2013_2023.columns:
        pop_mask = df_2013_2023['Subject population'].str.contains(
            'Census usually resident population', case=False, na=False
        )
    else:
        pop_mask = pd.Series([True] * len(df_2013_2023), index=df_2013_2023.index)
    
    unit_mask = df_2013_2023['Unit'].str.contains(
        'Percentage of total stated', case=False, na=False
    )
    
    df_no_religion = df_2013_2023[no_religion_mask & pop_mask & unit_mask]
    
    for year in [2013, 2018, 2023]:
        year_data = df_no_religion[df_no_religion['Census Year'] == year]
        if len(year_data) > 0:
            value = year_data['Value'].iloc[0]
            share = float(value) / 100.0
            results.append({"year": year, "share_none": share})
            print(f"  {year}: Percentage = {value:.2f}% -> share = {share:.4f}")
        else:
            print(f"WARNING: No data found for year {year}")
    
    # Create output DataFrame
    if not results:
        print("ERROR: No data extracted", file=sys.stderr)
        return
    
    df_output = pd.DataFrame(results)
    df_output = df_output.sort_values('year').reset_index(drop=True)
    df_output = df_output.drop_duplicates(subset=['year']).reset_index(drop=True)
    
    # Save output
    output_path = timeseries_dir / "nz_none_share_2001_2006_2013_2018_2023.csv"
    df_output.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"Output saved to: {output_path}")
    print("\nExtracted years:", sorted(df_output['year'].tolist()))
    print("\nTime series:")
    print(df_output.to_string(index=False))
    
    # Sanity check
    print("\n" + "=" * 60)
    print("Sanity check:")
    shares = df_output['share_none'].tolist()
    min_share = min(shares)
    max_share = max(shares)
    print(f"Share range: {min_share:.4f} - {max_share:.4f} ({min_share*100:.2f}% - {max_share*100:.2f}%)")
    
    if min_share < 0.1 or max_share > 0.6:
        print("WARNING: Shares outside reasonable range (0.1-0.6)")
        if min_share < 0.1:
            print(f"  Note: Early years (2001, 2006) are lower ({min_share*100:.1f}%) - this may be correct historically")
    else:
        print("OK: Shares within reasonable range")
        if min_share < 0.3:
            print(f"  Note: Early years are below 0.3, but 2013+ are within expected range (0.3-0.6)")


if __name__ == "__main__":
    import sys
    main()
