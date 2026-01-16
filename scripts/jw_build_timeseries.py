#!/usr/bin/env python3
"""
Build time series for a specific country from Jehovah's Witnesses raw CSV files.

Usage:
    python scripts/jw_build_timeseries.py --start_year 2014 --end_year 2024 --country Turkey
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List


def find_country_column(headers: List[str]) -> Optional[int]:
    """Find the country/territory column index."""
    candidates = ["Country", "Land", "Lands", "Country or Territory", "Country/Territory", "Territory"]
    for i, header in enumerate(headers):
        header_lower = header.lower()
        for candidate in candidates:
            if candidate.lower() in header_lower:
                return i
    return None


def find_country_row(rows: List[List], country_col_idx: int, country_name: str) -> Optional[List]:
    """Find the row for a country (case-insensitive exact match or substring)."""
    country_lower = country_name.lower()
    # Normalize Turkish characters for better matching
    country_normalized = country_lower.replace('ü', 'u').replace('ö', 'o').replace('ş', 's').replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i')
    
    for row in rows:
        if country_col_idx < len(row):
            row_country = row[country_col_idx].strip()
            row_country_lower = row_country.lower()
            row_country_normalized = row_country_lower.replace('ü', 'u').replace('ö', 'o').replace('ş', 's').replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i')
            
            # Try exact case-insensitive match
            if row_country_lower == country_lower:
                return row
            # Try normalized match
            if row_country_normalized == country_normalized:
                return row
            # Try substring match containing country name (check for "turk" in both)
            if "turk" in country_normalized and "turk" in row_country_normalized:
                return row
            if country_lower in row_country_lower or row_country_lower in country_lower:
                return row
            # Try normalized substring match
            if country_normalized in row_country_normalized or row_country_normalized in country_normalized:
                return row
    return None


def find_column(headers: List[str], candidates: List[str], year: Optional[int] = None) -> Optional[int]:
    """Find column index by matching candidates, optionally with year prefix."""
    for i, header in enumerate(headers):
        header_lower = header.lower()
        for candidate in candidates:
            candidate_lower = candidate.lower()
            # Try exact match or contains
            if candidate_lower == header_lower or candidate_lower in header_lower:
                # If year is specified, check if year is in header
                if year is None or str(year) in header:
                    return i
    return None


def clean_number(value: str) -> Optional[int]:
    """Clean and convert number string to int."""
    if not value or value.strip() == "":
        return None
    # Remove commas, spaces, non-breaking spaces
    cleaned = re.sub(r'[, \u00A0]', '', str(value).strip())
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_year_data(year: int, csv_path: Path, country_name: str) -> Optional[Dict]:
    """Parse data for a specific year and country."""
    if not csv_path.exists():
        print(f"WARNING: File not found: {csv_path}", file=sys.stderr)
        return None
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)
    except Exception as e:
        print(f"WARNING: Error reading {csv_path}: {e}", file=sys.stderr)
        return None
    
    # Find country column
    country_col_idx = find_country_column(headers)
    if country_col_idx is None:
        print(f"WARNING: Could not find country column in {csv_path}", file=sys.stderr)
        return None
    
    # Find country row
    country_row = find_country_row(rows, country_col_idx, country_name)
    if country_row is None:
        print(f"WARNING: {country_name} not found in {csv_path}", file=sys.stderr)
        return None
    
    # Find population column
    pop_candidates = ["Population", "Pop.", "Population (Estimated)"]
    pop_col_idx = find_column(headers, pop_candidates)
    if pop_col_idx is None:
        print(f"WARNING: Could not find population column in {csv_path}", file=sys.stderr)
        return None
    
    # Find average publishers column
    avg_pub_candidates = [
        f"{year} Av. Pubs.", f"{year} Average Publishers", f"{year} Avg. Publishers",
        "Average publishers", "Avg. publishers", "Publishers (average)", 
        "Av. Pubs.", "Average Publishers"
    ]
    avg_pub_col_idx = find_column(headers, avg_pub_candidates, year)
    # If average not found, try to use peak as fallback (for years like 2017-2018)
    if avg_pub_col_idx is None:
        # Try peak as fallback
        peak_candidates = [
            f"{year} Peak Pubs.", f"{year} Peak Publishers",
            "Peak publishers", "Peak", "Peak Pubs.", "Peak Publishers"
        ]
        peak_fallback = find_column(headers, peak_candidates, year)
        if peak_fallback is not None:
            avg_pub_col_idx = peak_fallback
            print(f"WARNING: Using peak publishers as average for {year} in {csv_path}", file=sys.stderr)
        else:
            print(f"WARNING: Could not find average publishers column in {csv_path}", file=sys.stderr)
            return None
    
    # Find peak publishers column
    peak_pub_candidates = [
        f"{year} Peak Pubs.", f"{year} Peak Publishers",
        "Peak publishers", "Peak", "Peak Pubs.", "Peak Publishers"
    ]
    peak_pub_col_idx = find_column(headers, peak_pub_candidates, year)
    
    # Extract values
    population = clean_number(country_row[pop_col_idx] if pop_col_idx < len(country_row) else "")
    avg_publishers = clean_number(country_row[avg_pub_col_idx] if avg_pub_col_idx < len(country_row) else "")
    peak_publishers = clean_number(country_row[peak_pub_col_idx] if peak_pub_col_idx is not None and peak_pub_col_idx < len(country_row) else "")
    
    if population is None or avg_publishers is None:
        print(f"WARNING: Missing required data for {country_name} in {csv_path}", file=sys.stderr)
        return None
    
    # Calculate shares
    share_avg = avg_publishers / population if population > 0 else None
    share_peak = peak_publishers / population if peak_publishers is not None and population > 0 else None
    
    return {
        "year": year,
        "country": country_name,
        "population": population,
        "avg_publishers": avg_publishers,
        "peak_publishers": peak_publishers,
        "share_avg": share_avg,
        "share_peak": share_peak
    }


def main():
    parser = argparse.ArgumentParser(description="Build time series from Jehovah's Witnesses CSV files")
    parser.add_argument("--start_year", type=int, required=True, help="Start year (e.g., 2014)")
    parser.add_argument("--end_year", type=int, required=True, help="End year (e.g., 2024)")
    parser.add_argument("--country", type=str, required=True, help="Country name (e.g., Turkey)")
    parser.add_argument("--data_dir", type=str, default=".", help="Base directory for CSV files")
    
    args = parser.parse_args()
    
    base_dir = Path(args.data_dir).resolve()
    
    # Try both file naming patterns
    pattern1 = base_dir / "data" / "jw" / "raw" / f"jw_country_{args.start_year}.csv"
    pattern2 = base_dir / f"Jehovah_Data_{args.start_year}.csv"
    
    # Determine which pattern to use
    use_pattern1 = pattern1.exists()
    use_pattern2 = pattern2.exists()
    
    if not use_pattern1 and not use_pattern2:
        print(f"ERROR: Could not find CSV files. Tried:", file=sys.stderr)
        print(f"  - {pattern1}", file=sys.stderr)
        print(f"  - {pattern2}", file=sys.stderr)
        sys.exit(1)
    
    # Collect data for all years
    results = []
    missing_years = []
    
    for year in range(args.start_year, args.end_year + 1):
        if use_pattern1:
            csv_path = base_dir / "data" / "jw" / "raw" / f"jw_country_{year}.csv"
        else:
            csv_path = base_dir / f"Jehovah_Data_{year}.csv"
        
        data = parse_year_data(year, csv_path, args.country)
        if data:
            results.append(data)
        else:
            missing_years.append(year)
    
    if not results:
        print(f"ERROR: No data found for {args.country} in any year", file=sys.stderr)
        sys.exit(1)
    
    # Sort by year
    results.sort(key=lambda x: x["year"])
    
    # Write output
    output_dir = base_dir / "data" / "jw" / "timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"jw_timeseries_{args.country.lower()}_{args.start_year}_{args.end_year}.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "year", "country", "population", "avg_publishers", 
            "peak_publishers", "share_avg", "share_peak"
        ])
        writer.writeheader()
        for row in results:
            # Convert None to empty string for CSV
            csv_row = {k: ("" if v is None else v) for k, v in row.items()}
            writer.writerow(csv_row)
    
    print(f"Successfully created: {output_file}")
    print(f"Rows written: {len(results)}")
    if missing_years:
        print(f"Missing years: {missing_years}", file=sys.stderr)


if __name__ == "__main__":
    main()
