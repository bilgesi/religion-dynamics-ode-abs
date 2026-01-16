#!/usr/bin/env python3
"""
Check which countries have complete data (no missing or invalid data) across all years 2014-2024.
"""

import csv
import re
from pathlib import Path
from typing import Set, Dict, List, Optional


def clean_number(value: str) -> Optional[int]:
    """Clean and convert number string to int."""
    if not value or value.strip() == "":
        return None
    cleaned = re.sub(r'[, \u00A0]', '', str(value).strip())
    try:
        return int(cleaned)
    except ValueError:
        return None


def find_country_column(headers: List[str]) -> Optional[int]:
    """Find the country/territory column index."""
    candidates = ["Country", "Land", "Lands", "Country or Territory", "Country/Territory", "Territory"]
    for i, header in enumerate(headers):
        header_lower = header.lower()
        for candidate in candidates:
            if candidate.lower() in header_lower:
                return i
    return None


def find_column(headers: List[str], candidates: List[str], year: Optional[int] = None) -> Optional[int]:
    """Find column index by matching candidates, optionally with year prefix."""
    for i, header in enumerate(headers):
        header_lower = header.lower()
        for candidate in candidates:
            candidate_lower = candidate.lower()
            if candidate_lower == header_lower or candidate_lower in header_lower:
                if year is None or str(year) in header:
                    return i
    return None


def check_country_complete(country_name: str, year: int, csv_path: Path) -> bool:
    """Check if a country has complete data for a given year."""
    if not csv_path.exists():
        return False
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)
    except Exception:
        return False
    
    # Find country column
    country_col_idx = find_country_column(headers)
    if country_col_idx is None:
        return False
    
    # Find country row
    country_row = None
    country_lower = country_name.lower()
    country_normalized = country_lower.replace('ü', 'u').replace('ö', 'o').replace('ş', 's').replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i')
    
    for row in rows:
        if country_col_idx < len(row):
            row_country = row[country_col_idx].strip()
            row_country_lower = row_country.lower()
            row_country_normalized = row_country_lower.replace('ü', 'u').replace('ö', 'o').replace('ş', 's').replace('ç', 'c').replace('ğ', 'g').replace('ı', 'i')
            
            if row_country_lower == country_lower or row_country_normalized == country_normalized:
                country_row = row
                break
            if "turk" in country_normalized and "turk" in row_country_normalized:
                country_row = row
                break
    
    if country_row is None:
        return False
    
    # Check required columns: Population, Avg. Pubs., Peak Pubs.
    pop_candidates = ["Population", "Pop.", "Population (Estimated)"]
    pop_col_idx = find_column(headers, pop_candidates)
    if pop_col_idx is None:
        return False
    
    avg_pub_candidates = [
        f"{year} Av. Pubs.", f"{year} Average Publishers", f"{year} Avg. Publishers",
        "Average publishers", "Avg. publishers", "Publishers (average)", 
        "Av. Pubs.", "Average Publishers"
    ]
    avg_pub_col_idx = find_column(headers, avg_pub_candidates, year)
    
    # If average not found, try peak as fallback (for years like 2017-2018)
    if avg_pub_col_idx is None:
        peak_candidates = [
            f"{year} Peak Pubs.", f"{year} Peak Publishers",
            "Peak publishers", "Peak", "Peak Pubs.", "Peak Publishers"
        ]
        avg_pub_col_idx = find_column(headers, peak_candidates, year)
    
    if avg_pub_col_idx is None:
        return False
    
    peak_pub_candidates = [
        f"{year} Peak Pubs.", f"{year} Peak Publishers",
        "Peak publishers", "Peak", "Peak Pubs.", "Peak Publishers"
    ]
    peak_pub_col_idx = find_column(headers, peak_pub_candidates, year)
    
    # Check if values exist and are valid
    population = clean_number(country_row[pop_col_idx] if pop_col_idx < len(country_row) else "")
    avg_publishers = clean_number(country_row[avg_pub_col_idx] if avg_pub_col_idx < len(country_row) else "")
    
    if population is None or population <= 0:
        return False
    if avg_publishers is None or avg_publishers < 0:
        return False
    
    # Peak is optional, but if column exists, value should be valid
    if peak_pub_col_idx is not None and peak_pub_col_idx < len(country_row):
        peak_publishers = clean_number(country_row[peak_pub_col_idx])
        if peak_publishers is not None and peak_publishers < 0:
            return False
    
    return True


def get_all_countries(base_dir: Path, start_year: int, end_year: int) -> Set[str]:
    """Get all unique country names from all CSV files."""
    all_countries = set()
    
    for year in range(start_year, end_year + 1):
        csv_path = base_dir / f"Jehovah_Data_{year}.csv"
        if not csv_path.exists():
            continue
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
            
            country_col_idx = find_country_column(headers)
            if country_col_idx is None:
                continue
            
            for row in rows:
                if country_col_idx < len(row):
                    country_name = row[country_col_idx].strip()
                    # Skip "Other Lands" and empty rows
                    if country_name and "Other Lands" not in country_name:
                        all_countries.add(country_name)
        except Exception:
            continue
    
    return all_countries


def main():
    base_dir = Path(".")
    start_year = 2014
    end_year = 2024
    
    print(f"Checking countries for complete data from {start_year} to {end_year}...")
    print()
    
    # Get all countries
    all_countries = get_all_countries(base_dir, start_year, end_year)
    print(f"Total unique countries found: {len(all_countries)}")
    print()
    
    # Check each country
    complete_countries = []
    incomplete_countries = []
    
    for country in sorted(all_countries):
        complete = True
        missing_years = []
        
        for year in range(start_year, end_year + 1):
            csv_path = base_dir / f"Jehovah_Data_{year}.csv"
            if not check_country_complete(country, year, csv_path):
                complete = False
                missing_years.append(year)
        
        if complete:
            complete_countries.append(country)
        else:
            incomplete_countries.append((country, missing_years))
    
    # Print results
    print(f"Countries with COMPLETE data ({len(complete_countries)}):")
    print("=" * 60)
    for country in complete_countries:
        print(f"  - {country}")
    
    print()
    print(f"Countries with INCOMPLETE data ({len(incomplete_countries)}):")
    print("=" * 60)
    # Show only first 20 incomplete countries
    for country, missing_years in incomplete_countries[:20]:
        print(f"  - {country}: missing in {missing_years}")
    
    if len(incomplete_countries) > 20:
        print(f"  ... and {len(incomplete_countries) - 20} more")
    
    print()
    print(f"Summary:")
    print(f"  Complete: {len(complete_countries)}")
    print(f"  Incomplete: {len(incomplete_countries)}")
    print(f"  Total: {len(all_countries)}")


if __name__ == "__main__":
    main()
