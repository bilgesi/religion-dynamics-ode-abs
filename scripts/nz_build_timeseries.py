#!/usr/bin/env python3
"""
Build time series for New Zealand religion data from census CSV.

Extracts "No religion" percentages and creates a 2-strain replacement series:
- share_none: percentage of people with no religion (as fraction)
- share_religion: 1.0 - share_none (percentage with any religion, as fraction)

Usage:
    python scripts/nz_build_timeseries.py
"""

import csv
import sys
from pathlib import Path
from typing import Dict, Optional


def main():
    # Paths
    base_dir = Path(".")
    input_file = base_dir / "data" / "nz" / "raw" / "nz_religion_2013_2018_2023.csv"
    
    # Try alternative path if the expected one doesn't exist
    if not input_file.exists():
        alt_file = base_dir / "Census_Usually_resident_population_by_religious_affiliation_2013_2018_2023.csv"
        if alt_file.exists():
            input_file = alt_file
        else:
            print(f"ERROR: Could not find input file. Tried:", file=sys.stderr)
            print(f"  - {input_file}", file=sys.stderr)
            print(f"  - {alt_file}", file=sys.stderr)
            sys.exit(1)
    
    # Read CSV (handle BOM if present)
    try:
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"ERROR: Failed to read {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Filter and extract data
    # Filter: Subject population == "Census usually resident population", Unit == "Percentage of total stated"
    # Extract: Religious affiliation == "No religion"
    
    no_religion_data = {}
    
    for row in rows:
        # Handle BOM in column names
        subject_pop = row.get("Subject population", "").strip()
        unit = row.get("Unit", "").strip()
        religious_aff = row.get("Religious affiliation", "").strip()
        # Try both with and without BOM
        census_year = row.get("Census Year", "").strip() or row.get("\ufeffCensus Year", "").strip()
        value_str = row.get("Value", "").strip()
        
        # Check filters - use case-insensitive contains for Subject population
        subject_pop_lower = subject_pop.lower()
        if "census usually resident population" not in subject_pop_lower:
            continue
        if unit != "Percentage of total stated":
            continue
        if religious_aff != "No religion":
            continue
        
        # Parse year
        try:
            year = int(census_year)
        except (ValueError, TypeError):
            continue
        
        # Parse value
        try:
            value = float(value_str)
        except (ValueError, TypeError):
            continue
        
        # Store data
        if year not in no_religion_data:
            no_religion_data[year] = value
    
    # Check if we have data for all expected years
    expected_years = [2013, 2018, 2023]
    missing_years = [y for y in expected_years if y not in no_religion_data]
    
    if missing_years:
        print(f"WARNING: Missing data for years: {missing_years}", file=sys.stderr)
    
    if not no_religion_data:
        print("ERROR: No data found for 'No religion' with required filters", file=sys.stderr)
        sys.exit(1)
    
    # Build time series
    time_series = []
    for year in sorted(no_religion_data.keys()):
        share_none = no_religion_data[year] / 100.0  # Convert percentage to fraction
        share_religion = 1.0 - share_none
        time_series.append({
            "year": year,
            "share_none": share_none,
            "share_religion": share_religion
        })
    
    # Sanity check
    expected_shares = {2013: 0.419, 2018: 0.482, 2023: 0.516}
    print("Sanity check (expected vs actual):")
    for year, expected in expected_shares.items():
        if year in no_religion_data:
            actual = no_religion_data[year] / 100.0
            diff = abs(actual - expected)
            status = "OK" if diff < 0.02 else "WARN"
            print(f"  {year}: expected ~{expected:.3f}, actual {actual:.6f} (diff: {diff:.6f}) {status}")
    
    # Write output
    output_dir = base_dir / "data" / "nz" / "timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "nz_none_vs_religion_2013_2018_2023.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["year", "share_none", "share_religion"])
        writer.writeheader()
        for row in time_series:
            writer.writerow(row)
    
    print(f"\nSuccessfully created: {output_file}")
    print(f"Rows written: {len(time_series)}")


if __name__ == "__main__":
    main()
