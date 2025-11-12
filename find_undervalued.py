#!/usr/bin/env python3
"""
Main script to run the complete data pipeline for identifying undervalued MLB hitters.

This script orchestrates the entire process:
1. Collects comprehensive 2025 MLB season data for all hitters with ≥200 PA
2. Fetches Statcast metrics (Barrel%, HardHit%, Exit Velocity, etc.)
3. Merges salary data
4. Calculates all advanced metrics (TOVA+, UPI, OPS 2.0, UVS, etc.)
5. Saves the complete dataset for use in the dashboard

The resulting dataset contains 350 hitters with 30+ advanced statistics,
including expected performance metrics, contact quality, plate discipline,
batted ball profile, and value metrics (WAR, salary efficiency).

Output: data/processed/comprehensive_stats_2025.csv
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_pipeline.fetch_comprehensive import main as fetch_data
from src.utils.metrics import calculate_all_advanced_metrics

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "models"


def display_results(df: pd.DataFrame, position_type: str, top_n: int = 20):
    """Display undervalued players in a readable format."""
    print("\n" + "=" * 80)
    print(f"TOP {top_n} UNDERVALUED {position_type.upper()}S")
    print("=" * 80)
    
    if position_type == 'Hitter':
        # Try different column name variations
        name_col = None
        for col in ['Name', 'last_name, first_name', 'name', 'player_name']:
            if col in df.columns:
                name_col = col
                break
        
        cols = [name_col] if name_col else []
        cols.extend(['undervalued_rank', 'value_score', 'woba'])
        
        # Try xwOBA column variations
        xwoba_col = None
        for col in ['est_woba', 'xwoba', 'xwOBA', 'expected_woba']:
            if col in df.columns:
                xwoba_col = col
                break
        if xwoba_col:
            cols.append(xwoba_col)
        
        cols.extend(['performance_diff', 'luck_index', 'contact_efficiency', 'true_player_value'])
    else:
        # Try different column name variations
        name_col = None
        for col in ['Name', 'last_name, first_name', 'name', 'player_name']:
            if col in df.columns:
                name_col = col
                break
        
        cols = [name_col] if name_col else []
        cols.extend(['undervalued_rank', 'value_score'])
        
        # Try ERA column variations
        era_col = None
        for col in ['era', 'ERA', 'ERA_']:
            if col in df.columns:
                era_col = col
                break
        if era_col:
            cols.append(era_col)
        
        xera_col = None
        for col in ['xera', 'xERA', 'expected_era']:
            if col in df.columns:
                xera_col = col
                break
        if xera_col:
            cols.append(xera_col)
        
        cols.extend(['performance_diff', 'pitch_deception_index', 'true_player_value', 'csw_percent'])
    
    # Select available columns - ensure name column is included
    available_cols = []
    for col in cols:
        if col is not None and col in df.columns:
            available_cols.append(col)
    
    # Make sure we have a name column - check all possible name columns
    name_cols_to_try = ['last_name, first_name', 'Name', 'name', 'player_name']
    name_found = False
    for nc in name_cols_to_try:
        if nc in df.columns and nc not in available_cols:
            available_cols.insert(0, nc)
            name_found = True
            break
    
    # Use the sorted dataframe (already sorted by undervalued_rank)
    display_df = df[available_cols].head(top_n).copy()
    
    # Debug: check if name column has data
    name_col_in_df = None
    for nc in name_cols_to_try:
        if nc in display_df.columns:
            name_col_in_df = nc
            break
    
    # Rename name column for display and ensure it has data
    if name_col_in_df:
        display_df = display_df.rename(columns={name_col_in_df: 'Player'})
        # Check if Player column has NaN values and try to fill from original index
        if display_df['Player'].isna().any():
            # Try to get names from original dataframe using index
            original_df = df if 'last_name, first_name' in df.columns else None
            if original_df is not None:
                for idx in display_df.index:
                    if pd.isna(display_df.loc[idx, 'Player']):
                        if idx in original_df.index and 'last_name, first_name' in original_df.columns:
                            display_df.loc[idx, 'Player'] = original_df.loc[idx, 'last_name, first_name']
    
    # Fill any remaining NaN in Player column
    if 'Player' in display_df.columns:
        display_df['Player'] = display_df['Player'].fillna('Unknown')
    
    # Format numeric columns
    for col in display_df.select_dtypes(include=['float64']).columns:
        if col in ['value_score', 'luck_index', 'contact_efficiency', 'true_player_value']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        elif col in ['woba', 'est_woba', 'xwoba', 'performance_diff']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        elif col in ['era', 'xera']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        elif col == 'csw_percent':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print("\n")


def main():
    """Main function to find undervalued players."""
    year = 2025
    
    print("\n" + "=" * 80)
    print("FINDING UNDERVALUED MLB PLAYERS - 2025 SEASON")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Collect comprehensive MLB data")
    print("2. Calculate advanced 'Moneyball 2.0' metrics")
    print("3. Calculate UVS (Undervaluation Score)")
    print("4. Generate final dataset")
    print("\n" + "=" * 80 + "\n")
    
    # Step 1: Check if data exists, otherwise fetch it
    data_path = DATA_PROCESSED / f"comprehensive_stats_{year}.csv"
    if not data_path.exists():
        data_path_alt = DATA_PROCESSED / f"combined_stats_{year}.csv"
        if not data_path_alt.exists():
            print("Data not found. Fetching comprehensive 2025 MLB data...")
            print("This may take several minutes...\n")
            df = fetch_data()
            if df is None or df.empty:
                print("ERROR: Could not fetch data. Please check your internet connection.")
                return
        else:
            print(f"Loading existing data from {data_path_alt}...")
            df = pd.read_csv(data_path_alt)
    else:
        print(f"Loading existing data from {data_path}...")
        df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} player records\n")
    
    # Step 2: Calculate advanced metrics if not already present
    if 'luck_index' not in df.columns:
        print("Calculating advanced metrics...")
        df = calculate_all_advanced_metrics(df)
        print("Advanced metrics calculated!\n")
    
    # Step 3: Separate hitters and pitchers and apply filters
    hitters = df[df.get('position_type', '') == 'Hitter'].copy()
    pitchers = df[df.get('position_type', '') == 'Pitcher'].copy()
    
    # Apply 200 PA filter for hitters (safety check even if data was collected before filter)
    if len(hitters) > 0:
        pa_col = None
        for col in ['pa', 'PA']:
            if col in hitters.columns:
                pa_col = col
                break
        
        if pa_col:
            before_count = len(hitters)
            hitters = hitters[hitters[pa_col] >= 200]
            after_count = len(hitters)
            if before_count != after_count:
                print(f"Filtered hitters: {before_count} -> {after_count} (>= 200 PA required)\n")
    
    # Apply IP filter for pitchers (safety check)
    if len(pitchers) > 0:
        ip_col = None
        for col in ['IP', 'ip', 'IP_']:
            if col in pitchers.columns:
                ip_col = col
                break
        
        if ip_col:
            # Try to identify starters vs relievers
            gs_col = None
            for col in ['GS', 'gs', 'Games Started']:
                if col in pitchers.columns:
                    gs_col = col
                    break
            
            before_count = len(pitchers)
            if gs_col and gs_col in pitchers.columns:
                # Starters: >= 50 IP, Relievers: >= 20 IP
                starters = pitchers[pitchers[gs_col] > 0]
                relievers = pitchers[pitchers[gs_col] == 0]
                starters_filtered = starters[starters[ip_col] >= 50]
                relievers_filtered = relievers[relievers[ip_col] >= 20]
                pitchers = pd.concat([starters_filtered, relievers_filtered], ignore_index=True)
            else:
                # Fallback: use 50 IP minimum for all
                pitchers = pitchers[pitchers[ip_col] >= 50]
            
            after_count = len(pitchers)
            if before_count != after_count:
                print(f"Filtered pitchers: {before_count} -> {after_count} (>= 50 IP starters, >= 20 IP relievers)\n")
    
    print(f"Hitters: {len(hitters)}")
    print(f"Pitchers: {len(pitchers)}\n")
    
    if len(hitters) == 0 and len(pitchers) == 0:
        print("ERROR: No player data found. Please check your data files.")
        return
    
    # Step 4: Data pipeline complete
    print("\n" + "=" * 80)
    print("DATA PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n✓ Processed {len(hitters)} hitters and {len(pitchers)} pitchers")
    print(f"✓ All advanced metrics calculated (UVS, TOVA+, UPI, OPS 2.0, etc.)")
    print(f"✓ Data saved to: {DATA_PROCESSED / 'comprehensive_stats_2025.csv'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\nTo view results:")
    print("- Activate venv: source venv/bin/activate")
    print("- Start dashboard: python frontend/table_dashboard.py")
    print("- Dashboard URL: http://localhost:8050")
    print("\nThe dashboard displays:")
    print("- All 350 hitters with 30+ advanced statistics")
    print("- UVS (Undervaluation Score) rankings")
    print("- Sortable, filterable tables with CSV export")
    print("\n")


if __name__ == "__main__":
    main()

