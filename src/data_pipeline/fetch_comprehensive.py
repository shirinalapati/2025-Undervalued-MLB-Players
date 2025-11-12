"""
Main comprehensive data pipeline for the Undervalued MLB Players project.

This module orchestrates the complete data collection and processing pipeline:
1. Fetches comprehensive batting statistics for all hitters with ≥200 PA
2. Collects Statcast metrics (Barrel%, HardHit%, Exit Velocity, Sweet Spot%)
3. Merges salary data from CSV files
4. Calculates all advanced composite metrics (TOVA+, UPI, OPS 2.0, UVS)
5. Saves the final dataset for use in the dashboard

The pipeline focuses exclusively on hitters (350 players from 2025 season)
and produces a comprehensive dataset with 30+ advanced statistics per player.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.fetch_advanced_metrics import combine_all_data
from src.utils.metrics import calculate_all_advanced_metrics
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def main():
    """Run the comprehensive data pipeline."""
    year = 2025
    
    logger.info("=" * 60)
    logger.info("Comprehensive MLB Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"Collecting data for {year} season...")
    
    # Fetch comprehensive data
    df = combine_all_data(year)
    
    if df.empty:
        logger.error("No data collected. Please check your connection and pybaseball installation.")
        return
    
    # Calculate all advanced metrics
    logger.info("Calculating advanced 'Moneyball 2.0' metrics...")
    df = calculate_all_advanced_metrics(df)
    
    # Save final dataset
    output_path = DATA_PROCESSED / f"comprehensive_stats_{year}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved comprehensive dataset: {output_path}")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Total columns: {len(df.columns)}")
    
    # Show summary
    if 'position_type' in df.columns:
        hitters = df[df['position_type'] == 'Hitter']
        pitchers = df[df['position_type'] == 'Pitcher']
        logger.info(f"Hitters: {len(hitters)}")
        logger.info(f"Pitchers: {len(pitchers)}")
    
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    
    return df


if __name__ == "__main__":
    main()

