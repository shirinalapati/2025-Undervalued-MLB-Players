"""
Fetch Statcast pitch-level data to calculate advanced contact quality metrics for hitters.

This module fetches pitch-by-pitch Statcast data for individual players to calculate
contact quality metrics that are not available in summary statistics:
- Average Exit Velocity: Mean speed of all batted balls
- Sweet Spot %: Percentage of batted balls with launch angle 8-32 degrees

These metrics are essential components of the Contact Quality Index (CQI) in the UVS formula.
The module processes players one at a time with delays to respect API rate limits.

Note: Some players may already have these metrics from summary data (Barrel%, HardHit%),
but this module provides more granular calculations when needed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Optional, List
from datetime import datetime
import time

sys.path.append(str(Path(__file__).parent.parent))

try:
    import pybaseball as pyb
    from pybaseball import statcast_batter, playerid_lookup
except ImportError:
    print("Warning: pybaseball not installed. Install with: pip install pybaseball")
    pyb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def calculate_contact_metrics(pitch_data: pd.DataFrame) -> dict:
    """
    Calculate contact quality metrics from pitch-level Statcast data.
    
    Args:
        pitch_data: DataFrame with Statcast pitch-level data
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {
        'avg_exit_velocity': np.nan,
        'sweet_spot_percent': np.nan,
        'barrel_percent': np.nan,
        'hard_hit_percent': np.nan,
        'total_batted_balls': 0
    }
    
    if pitch_data is None or pitch_data.empty:
        return metrics
    
    # Filter to only batted balls (where launch_speed and launch_angle exist)
    batted_balls = pitch_data[
        pitch_data['launch_speed'].notna() & 
        pitch_data['launch_angle'].notna()
    ].copy()
    
    if len(batted_balls) == 0:
        return metrics
    
    metrics['total_batted_balls'] = len(batted_balls)
    
    # Average Exit Velocity
    metrics['avg_exit_velocity'] = batted_balls['launch_speed'].mean()
    
    # Sweet Spot % (launch angle between 8° and 32°)
    sweet_spot = ((batted_balls['launch_angle'] >= 8) & 
                  (batted_balls['launch_angle'] <= 32))
    metrics['sweet_spot_percent'] = sweet_spot.mean() * 100
    
    # Hard Hit % (exit velocity >= 95 mph)
    hard_hit = batted_balls['launch_speed'] >= 95
    metrics['hard_hit_percent'] = hard_hit.mean() * 100
    
    # Barrel % (based on Statcast definition)
    # Barrel: launch_angle between 8-50° AND exit_velocity thresholds:
    #   - 8-12°: >= 98 mph
    #   - 12-14°: >= 97 mph
    #   - 14-16°: >= 95 mph
    #   - 16-24°: >= 93 mph
    #   - 24-26°: >= 92 mph
    #   - 26-50°: >= 91 mph
    def is_barrel(la, ev):
        if pd.isna(la) or pd.isna(ev):
            return False
        if 8 <= la <= 12:
            return ev >= 98
        elif 12 < la <= 14:
            return ev >= 97
        elif 14 < la <= 16:
            return ev >= 95
        elif 16 < la <= 24:
            return ev >= 93
        elif 24 < la <= 26:
            return ev >= 92
        elif 26 < la <= 50:
            return ev >= 91
        return False
    
    barrels = batted_balls.apply(
        lambda row: is_barrel(row['launch_angle'], row['launch_speed']), 
        axis=1
    )
    metrics['barrel_percent'] = barrels.mean() * 100
    
    return metrics


def fetch_player_statcast_data(player_id: int, start_date: str, end_date: str, 
                                delay: float = 0.5) -> Optional[pd.DataFrame]:
    """
    Fetch Statcast pitch-level data for a specific player.
    
    Args:
        player_id: MLB player ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        delay: Delay between requests (seconds) to avoid rate limiting
        
    Returns:
        DataFrame with pitch-level data or None if error
    """
    if pyb is None:
        return None
    
    try:
        time.sleep(delay)  # Rate limiting
        data = statcast_batter(start_date, end_date, player_id=player_id)
        return data if data is not None and not data.empty else None
    except Exception as e:
        logger.warning(f"Error fetching Statcast data for player {player_id}: {e}")
        return None


def calculate_statcast_metrics_for_players(
    player_ids: List[int], 
    year: int = 2025,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate Statcast contact metrics for a list of players.
    
    Args:
        player_ids: List of MLB player IDs
        year: Season year
        sample_size: If provided, only process first N players (for testing)
        
    Returns:
        DataFrame with player_id and calculated metrics
    """
    if pyb is None:
        logger.error("pybaseball not available")
        return pd.DataFrame()
    
    # Date range for the season
    start_date = f"{year}-03-01"  # Spring training start
    end_date = f"{year}-10-31"    # Regular season end
    
    results = []
    total_players = len(player_ids) if sample_size is None else min(sample_size, len(player_ids))
    
    logger.info(f"Calculating Statcast metrics for {total_players} players...")
    logger.info("This may take a while due to API rate limiting...")
    
    for i, player_id in enumerate(player_ids[:total_players], 1):
        if i % 10 == 0:
            logger.info(f"Processing player {i}/{total_players}...")
        
        pitch_data = fetch_player_statcast_data(player_id, start_date, end_date)
        
        if pitch_data is not None and not pitch_data.empty:
            metrics = calculate_contact_metrics(pitch_data)
            metrics['player_id'] = player_id
            results.append(metrics)
        else:
            # Add empty metrics
            results.append({
                'player_id': player_id,
                'avg_exit_velocity': np.nan,
                'sweet_spot_percent': np.nan,
                'barrel_percent': np.nan,
                'hard_hit_percent': np.nan,
                'total_batted_balls': 0
            })
    
    df = pd.DataFrame(results)
    logger.info(f"Calculated metrics for {len(df)} players")
    logger.info(f"Players with data: {df['total_batted_balls'].gt(0).sum()}")
    
    return df


def enhance_batting_data_with_statcast(
    batting_df: pd.DataFrame,
    year: int = 2025,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Enhance existing batting data with Statcast pitch-level metrics.
    
    Args:
        batting_df: DataFrame with batting statistics
        year: Season year
        sample_size: If provided, only process first N players (for testing)
        
    Returns:
        Enhanced DataFrame with Statcast metrics added
    """
    if 'player_id' not in batting_df.columns:
        logger.error("batting_df must have 'player_id' column")
        return batting_df
    
    # Get unique player IDs
    player_ids = batting_df['player_id'].dropna().unique().tolist()
    
    if len(player_ids) == 0:
        logger.warning("No player IDs found in batting data")
        return batting_df
    
    # Calculate Statcast metrics
    statcast_metrics = calculate_statcast_metrics_for_players(
        player_ids, 
        year=year,
        sample_size=sample_size
    )
    
    if statcast_metrics.empty:
        logger.warning("No Statcast metrics calculated")
        return batting_df
    
    # Merge with existing data
    enhanced_df = pd.merge(
        batting_df,
        statcast_metrics[['player_id', 'avg_exit_velocity', 'sweet_spot_percent', 
                         'barrel_percent', 'hard_hit_percent']],
        on='player_id',
        how='left',
        suffixes=('', '_statcast')
    )
    
    # Update existing columns if Statcast data is better
    if 'avg_exit_velocity' in enhanced_df.columns:
        # Use Statcast value if original is NaN
        mask = enhanced_df['avg_exit_velocity'].isna()
        if 'avg_exit_velocity_statcast' in enhanced_df.columns:
            enhanced_df.loc[mask, 'avg_exit_velocity'] = enhanced_df.loc[mask, 'avg_exit_velocity_statcast']
        enhanced_df = enhanced_df.drop(columns=['avg_exit_velocity_statcast'], errors='ignore')
    
    # Update barrel and hard hit percentages
    for col in ['barrel_batted_rate', 'Barrel%']:
        if col in enhanced_df.columns:
            mask = enhanced_df[col].isna()
            if 'barrel_percent' in enhanced_df.columns:
                enhanced_df.loc[mask, col] = enhanced_df.loc[mask, 'barrel_percent']
    
    for col in ['hard_hit_percent', 'HardHit%']:
        if col in enhanced_df.columns:
            mask = enhanced_df[col].isna()
            if 'hard_hit_percent' in enhanced_df.columns:
                enhanced_df.loc[mask, col] = enhanced_df.loc[mask, 'hard_hit_percent']
    
    logger.info(f"Enhanced {enhanced_df['avg_exit_velocity'].notna().sum()} players with Statcast metrics")
    
    return enhanced_df


if __name__ == '__main__':
    # Example usage
    logger.info("Statcast pitch-level data fetcher")
    logger.info("Use enhance_batting_data_with_statcast() to add metrics to existing data")

