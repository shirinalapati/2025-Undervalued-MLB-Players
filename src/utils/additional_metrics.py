"""
Additional advanced metrics for hitters.

This module provides placeholder functions for advanced metrics that require
more granular data than what's available in summary statistics:
- Run Value per Pitch/Swing: Requires pitch-level data
- WAR per Plate Appearance: Can be calculated from WAR and PA
- Salary vs xWAR Gap: Requires contract and projection data
- WPA (Win Probability Added): Requires play-by-play data
- Park-adjusted metrics: ERA- and FIP- (for pitchers, not used in this project)

Most of these are placeholders that return NaN or default values, as the full
implementation would require pitch-level or play-by-play data that's not
currently collected. The main project focuses on metrics available from
summary statistics (Statcast, FanGraphs).
"""

import pandas as pd
import numpy as np
from src.utils.metrics import normalize_metric


def calculate_run_value_per_pitch(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Run Value per Pitch.
    
    Args:
        df: DataFrame with run value and pitch data
        
    Returns:
        Series with run value per pitch
    """
    # Try to find run value and pitch columns
    rv_col = None
    for col in ['run_value', 'Run Value', 'RV', 'rv']:
        if col in df.columns:
            rv_col = col
            break
    
    pitch_col = None
    for col in ['pitches', 'Pitches', 'P', 'p']:
        if col in df.columns:
            pitch_col = col
            break
    
    if rv_col is None or pitch_col is None:
        return pd.Series(np.nan, index=df.index)
    
    rv = df[rv_col].fillna(0)
    pitches = df[pitch_col].fillna(1)
    
    return rv / pitches


def calculate_run_value_per_swing(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Run Value per Swing.
    
    Args:
        df: DataFrame with run value and swing data
        
    Returns:
        Series with run value per swing
    """
    rv_col = None
    for col in ['run_value', 'Run Value', 'RV', 'rv']:
        if col in df.columns:
            rv_col = col
            break
    
    swing_col = None
    for col in ['swings', 'Swings', 'swing_percent', 'swings_total']:
        if col in df.columns:
            swing_col = col
            break
    
    if rv_col is None or swing_col is None:
        return pd.Series(np.nan, index=df.index)
    
    rv = df[rv_col].fillna(0)
    swings = df[swing_col].fillna(1)
    
    return rv / swings


def calculate_war_per_pa(df: pd.DataFrame) -> pd.Series:
    """
    Calculate WAR per Plate Appearance.
    
    Args:
        df: DataFrame with WAR and PA columns
        
    Returns:
        Series with WAR per PA
    """
    war_col = None
    for col in ['war', 'WAR', 'WAR_']:
        if col in df.columns:
            war_col = col
            break
    
    pa_col = None
    for col in ['pa', 'PA', 'PA_']:
        if col in df.columns:
            pa_col = col
            break
    
    if war_col is None or pa_col is None:
        return pd.Series(np.nan, index=df.index)
    
    war = df[war_col].fillna(0)
    pa = df[pa_col].fillna(1)
    
    return war / pa


def calculate_salary_vs_xwar_gap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Salary vs xWAR Gap.
    
    Measures the difference between actual salary and expected salary based on xWAR.
    
    Args:
        df: DataFrame with salary, xWAR, and WAR columns
        
    Returns:
        Series with salary vs xWAR gap
    """
    salary_col = None
    for col in ['salary', 'Salary', 'salary_2025']:
        if col in df.columns:
            salary_col = col
            break
    
    xwar_col = None
    for col in ['xwar', 'xWAR', 'expected_war', 'est_war']:
        if col in df.columns:
            xwar_col = col
            break
    
    if salary_col is None or xwar_col is None:
        return pd.Series(np.nan, index=df.index)
    
    # Estimate expected salary based on xWAR (roughly $8M per WAR)
    # This is a simplified calculation
    salary_per_war = 8_000_000  # $8M per WAR (approximate market rate)
    expected_salary = df[xwar_col].fillna(0) * salary_per_war
    actual_salary = df[salary_col].fillna(0)
    
    # Gap = Expected - Actual (positive = player is undervalued)
    gap = expected_salary - actual_salary
    
    return gap


def calculate_wpa(df: pd.DataFrame) -> pd.Series:
    """
    Get Win Probability Added (WPA) if available.
    
    Args:
        df: DataFrame with WPA data
        
    Returns:
        Series with WPA values
    """
    wpa_col = None
    for col in ['wpa', 'WPA', 'Win Probability Added']:
        if col in df.columns:
            wpa_col = col
            break
    
    if wpa_col is None:
        return pd.Series(np.nan, index=df.index)
    
    return df[wpa_col].fillna(0)


def calculate_era_minus(df: pd.DataFrame) -> pd.Series:
    """
    Calculate ERA- (park-adjusted ERA, where 100 is league average).
    
    Lower is better. ERA- < 100 means better than league average.
    
    Args:
        df: DataFrame with ERA data
        
    Returns:
        Series with ERA- values
    """
    era_col = None
    for col in ['era', 'ERA', 'ERA_']:
        if col in df.columns:
            era_col = col
            break
    
    if era_col is None:
        return pd.Series(np.nan, index=df.index)
    
    # ERA- = (ERA / League ERA) * 100
    # For simplicity, we'll use the mean ERA as league average
    league_era = df[era_col].mean()
    era_minus = (df[era_col].fillna(0) / (league_era + 1e-10)) * 100
    
    return era_minus


def calculate_fip_minus(df: pd.DataFrame) -> pd.Series:
    """
    Calculate FIP- (park-adjusted FIP, where 100 is league average).
    
    Lower is better. FIP- < 100 means better than league average.
    
    Args:
        df: DataFrame with FIP data
        
    Returns:
        Series with FIP- values
    """
    fip_col = None
    for col in ['fip', 'FIP', 'FIP_']:
        if col in df.columns:
            fip_col = col
            break
    
    if fip_col is None:
        return pd.Series(np.nan, index=df.index)
    
    # FIP- = (FIP / League FIP) * 100
    league_fip = df[fip_col].mean()
    fip_minus = (df[fip_col].fillna(0) / (league_fip + 1e-10)) * 100
    
    return fip_minus


def calculate_all_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all additional metrics that were missing.
    
    Args:
        df: DataFrame with player statistics
        
    Returns:
        DataFrame with additional metrics added
    """
    df = df.copy()
    
    # Separate hitters and pitchers
    hitters = df[df.get('position_type', '') == 'Hitter'].copy()
    pitchers = df[df.get('position_type', '') == 'Pitcher'].copy()
    
    # Calculate metrics for hitters
    if len(hitters) > 0:
        hitters['run_value_per_pitch'] = calculate_run_value_per_pitch(hitters)
        hitters['run_value_per_swing'] = calculate_run_value_per_swing(hitters)
        hitters['war_per_pa'] = calculate_war_per_pa(hitters)
        hitters['salary_vs_xwar_gap'] = calculate_salary_vs_xwar_gap(hitters)
        hitters['wpa'] = calculate_wpa(hitters)
    
    # Calculate metrics for pitchers
    if len(pitchers) > 0:
        pitchers['era_minus'] = calculate_era_minus(pitchers)
        pitchers['fip_minus'] = calculate_fip_minus(pitchers)
        pitchers['wpa'] = calculate_wpa(pitchers)
    
    # Combine back
    if len(hitters) > 0 and len(pitchers) > 0:
        combined = pd.concat([hitters, pitchers], ignore_index=True)
    elif len(hitters) > 0:
        combined = hitters
    elif len(pitchers) > 0:
        combined = pitchers
    else:
        combined = df
    
    return combined

