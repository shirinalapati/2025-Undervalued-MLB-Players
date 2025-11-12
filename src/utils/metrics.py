"""
Utility functions for calculating advanced baseball metrics for hitters.

This module provides functions to calculate various advanced statistics used
in player valuation and the UVS (Undervaluation Score) calculation:
- wOBA (Weighted On-Base Average)
- Luck Index: (xwOBA - wOBA) + (xBA - BA)
- Contact Efficiency: wOBA / HardHit%
- True Player Value: Weighted combination of xwOBA, OAA, BsR, WAR/Salary
- WAR per Salary: Efficiency metric

This module also orchestrates the calculation of all advanced metrics by calling
the composite metric calculators (TOVA+, UPI, OPS 2.0, UVS) from other modules.
"""

import pandas as pd
import numpy as np


def calculate_wOBA(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Weighted On-Base Average (wOBA).
    
    wOBA weights each offensive outcome based on its run value.
    Uses 2023 weights as standard (adjust if needed for 2025).
    
    Args:
        df: DataFrame with columns: BB, HBP, 1B, 2B, 3B, HR, AB, SF
        
    Returns:
        Series with wOBA values
    """
    # 2023 wOBA weights (adjust for 2025 if available)
    wBB = 0.69
    wHBP = 0.72
    w1B = 0.89
    w2B = 1.27
    w3B = 1.62
    wHR = 2.10
    
    # Calculate components
    numerator = (
        wBB * df.get('BB', 0).fillna(0) +
        wHBP * df.get('HBP', 0).fillna(0) +
        w1B * (df.get('1B', 0).fillna(0)) +
        w2B * (df.get('2B', 0).fillna(0)) +
        w3B * (df.get('3B', 0).fillna(0)) +
        wHR * df.get('HR', 0).fillna(0)
    )
    
    denominator = (
        df.get('AB', 0).fillna(0) +
        df.get('BB', 0).fillna(0) -
        df.get('IBB', 0).fillna(0) +
        df.get('SF', 0).fillna(0) +
        df.get('HBP', 0).fillna(0)
    )
    
    wOBA = numerator / denominator.replace(0, np.nan)
    return wOBA


def calculate_value_score(
    df: pd.DataFrame,
    actual_col: str,
    expected_col: str,
    salary_col: str = None,
    salary_weight: float = 0.3
) -> pd.Series:
    """
    Calculate a composite value score for players.
    
    Combines performance over expectation with salary efficiency.
    
    Args:
        df: DataFrame with player data
        actual_col: Column name for actual performance metric
        expected_col: Column name for expected performance metric
        salary_col: Column name for salary (optional)
        salary_weight: Weight for salary component (0-1)
        
    Returns:
        Series with value scores
    """
    # Performance over expectation
    performance_diff = df[actual_col] - df[expected_col]
    performance_score = (performance_diff - performance_diff.min()) / (
        performance_diff.max() - performance_diff.min() + 1e-10
    )
    
    # If salary data available, incorporate it
    if salary_col and salary_col in df.columns:
        # Inverse salary (lower is better for value)
        salary_normalized = 1 - (
            (df[salary_col] - df[salary_col].min()) / 
            (df[salary_col].max() - df[salary_col].min() + 1e-10)
        )
        value_score = (
            (1 - salary_weight) * performance_score +
            salary_weight * salary_normalized
        )
    else:
        value_score = performance_score
    
    return value_score


def calculate_undervalued_rank(
    df: pd.DataFrame,
    value_score_col: str = 'value_score'
) -> pd.Series:
    """
    Rank players by value score (higher = more undervalued).
    
    Args:
        df: DataFrame with value scores
        value_score_col: Column name for value score
        
    Returns:
        Series with ranks (1 = most undervalued)
    """
    return df[value_score_col].rank(ascending=False, method='min').astype(int)


def normalize_metric(series: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize a metric to 0-1 scale.
    
    Args:
        series: Series to normalize
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized series
    """
    if method == 'minmax':
        return (series - series.min()) / (series.max() - series.min() + 1e-10)
    elif method == 'zscore':
        return (series - series.mean()) / (series.std() + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_luck_index(df: pd.DataFrame) -> pd.Series:
    """
    Calculate "Luck Index" = (xwOBA - wOBA) + (xBA - BA)
    
    Positive values indicate "unlucky" players (underperforming expected stats).
    These players are often undervalued.
    
    Args:
        df: DataFrame with xwOBA, wOBA, xBA, BA columns
        
    Returns:
        Series with luck index values
    """
    luck_index = pd.Series(0.0, index=df.index)
    
    # For hitters
    if 'xwoba' in df.columns and 'woba' in df.columns:
        woba_diff = df['xwoba'].fillna(0) - df['woba'].fillna(0)
        luck_index += woba_diff
    
    if 'xba' in df.columns and 'ba' in df.columns:
        ba_diff = df['xba'].fillna(0) - df['ba'].fillna(0)
        luck_index += ba_diff
    
    return luck_index


def calculate_contact_efficiency(df: pd.DataFrame) -> pd.Series:
    """
    Calculate "Contact Efficiency" = wOBA / HardHit%
    
    Measures how well a player converts hard contact into production.
    Higher values indicate efficient hitters.
    
    Args:
        df: DataFrame with wOBA and hard_hit_percent columns
        
    Returns:
        Series with contact efficiency values
    """
    # Try to find wOBA column
    woba_col = None
    for col in ['woba', 'wOBA', 'WOBA']:
        if col in df.columns:
            woba_col = col
            break
    
    # Try to find hard hit percent column
    hard_hit_col = None
    for col in ['hard_hit_percent', 'hard_hit%', 'HardHit%', 'hardhit_percent']:
        if col in df.columns:
            hard_hit_col = col
            break
    
    if woba_col is None or hard_hit_col is None:
        return pd.Series(np.nan, index=df.index)
    
    hard_hit = df[hard_hit_col].fillna(0)
    # Check if it's already a percentage (0-100) or decimal (0-1)
    if hard_hit.max() > 1:
        hard_hit = hard_hit / 100  # Convert percentage to decimal
    
    woba = df[woba_col].fillna(0)
    
    # Avoid division by zero
    efficiency = woba / (hard_hit + 1e-10)
    
    return efficiency


def calculate_pitch_deception_index(df: pd.DataFrame) -> pd.Series:
    """
    Calculate "Pitch Deception Index" for pitchers.
    
    Measures how unique a pitcher's pitch movement is compared to average.
    Higher values indicate more deceptive pitches.
    
    Note: This is a simplified version. Full implementation would require
    pitch-level data with horizontal/vertical break measurements.
    
    Args:
        df: DataFrame with pitch movement data (if available)
        
    Returns:
        Series with pitch deception index values
    """
    # Placeholder - would need actual pitch movement data
    # For now, use spin rate and velocity as proxies
    deception = pd.Series(0.0, index=df.index)
    
    if 'spin_rate' in df.columns:
        spin_norm = normalize_metric(df['spin_rate'].fillna(0))
        deception += spin_norm
    
    if 'release_speed' in df.columns:
        velo_norm = normalize_metric(df['release_speed'].fillna(0))
        deception += velo_norm
    
    return deception


def calculate_true_player_value(
    df: pd.DataFrame,
    position_type: str = 'hitter'
) -> pd.Series:
    """
    Calculate "True Player Value (TPV)" = Weighted combination of:
    - xwOBA (or xERA for pitchers)
    - OAA (Outs Above Average) for defense
    - BsR (Base Running Runs) for baserunning
    - WAR/Salary for cost efficiency
    
    Args:
        df: DataFrame with player metrics
        position_type: 'hitter' or 'pitcher'
        
    Returns:
        Series with True Player Value scores
    """
    tpv = pd.Series(0.0, index=df.index)
    
    if position_type == 'hitter':
        # Weight: xwOBA (50%), OAA (20%), BsR (20%), WAR/Salary (10%)
        if 'xwoba' in df.columns:
            xwoba_norm = normalize_metric(df['xwoba'].fillna(0))
            tpv += 0.5 * xwoba_norm
        
        if 'oaa' in df.columns:
            oaa_norm = normalize_metric(df['oaa'].fillna(0))
            tpv += 0.2 * oaa_norm
        
        if 'bsr' in df.columns:
            bsr_norm = normalize_metric(df['bsr'].fillna(0))
            tpv += 0.2 * bsr_norm
        
        # WAR per salary (if available)
        if 'war' in df.columns and 'salary' in df.columns:
            war_per_salary = df['war'].fillna(0) / (df['salary'].fillna(1) + 1e-10)
            war_per_salary_norm = normalize_metric(war_per_salary)
            tpv += 0.1 * war_per_salary_norm
    
    else:  # pitcher
        # Weight: xERA (inverted, 50%), CSW% (30%), WAR/Salary (20%)
        if 'xera' in df.columns:
            # Lower ERA is better, so invert
            era_inv = -df['xera'].fillna(10)
            era_norm = normalize_metric(era_inv)
            tpv += 0.5 * era_norm
        
        if 'csw_percent' in df.columns:
            csw_norm = normalize_metric(df['csw_percent'].fillna(0))
            tpv += 0.3 * csw_norm
        
        if 'war' in df.columns and 'salary' in df.columns:
            war_per_salary = df['war'].fillna(0) / (df['salary'].fillna(1) + 1e-10)
            war_per_salary_norm = normalize_metric(war_per_salary)
            tpv += 0.2 * war_per_salary_norm
    
    return tpv


def calculate_war_per_salary(df: pd.DataFrame) -> pd.Series:
    """
    Calculate WAR per $1M salary.
    
    Higher values indicate better cost efficiency.
    
    Args:
        df: DataFrame with WAR and salary columns
        
    Returns:
        Series with WAR per $1M salary
    """
    if 'war' not in df.columns or 'salary' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    salary_millions = df['salary'].fillna(1) / 1_000_000  # Convert to millions
    war = df['war'].fillna(0)
    
    war_per_mil = war / (salary_millions + 1e-10)
    
    return war_per_mil


def calculate_market_inefficiency(
    df: pd.DataFrame,
    projected_col: str = 'steamer_war',
    actual_col: str = 'war'
) -> pd.Series:
    """
    Calculate market inefficiency = Actual WAR - Projected WAR.
    
    Positive values indicate players outperforming projections (undervalued by market).
    
    Args:
        df: DataFrame with projected and actual WAR
        projected_col: Column name for projected WAR
        actual_col: Column name for actual WAR
        
    Returns:
        Series with market inefficiency scores
    """
    if projected_col not in df.columns or actual_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    inefficiency = df[actual_col].fillna(0) - df[projected_col].fillna(0)
    
    return inefficiency


def calculate_all_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all advanced "Moneyball 2.0" metrics for a dataset.
    Includes TOVA+, UPI, and OPS 2.0 for hitters.
    
    Args:
        df: DataFrame with player statistics
        
    Returns:
        DataFrame with all advanced metrics added
    """
    df = df.copy()
    
    # Separate hitters and pitchers
    hitters = df[df.get('position_type', '') == 'Hitter'].copy()
    pitchers = df[df.get('position_type', '') == 'Pitcher'].copy()
    
    # Calculate metrics for hitters
    if len(hitters) > 0:
        hitters['luck_index'] = calculate_luck_index(hitters)
        hitters['contact_efficiency'] = calculate_contact_efficiency(hitters)
        hitters['true_player_value'] = calculate_true_player_value(hitters, 'hitter')
        hitters['war_per_salary'] = calculate_war_per_salary(hitters)
    
    # Calculate metrics for pitchers
    if len(pitchers) > 0:
        pitchers['pitch_deception_index'] = calculate_pitch_deception_index(pitchers)
        pitchers['true_player_value'] = calculate_true_player_value(pitchers, 'pitcher')
        pitchers['war_per_salary'] = calculate_war_per_salary(pitchers)
    
    # Combine back
    if len(hitters) > 0 and len(pitchers) > 0:
        combined = pd.concat([hitters, pitchers], ignore_index=True)
    elif len(hitters) > 0:
        combined = hitters
    elif len(pitchers) > 0:
        combined = pitchers
    else:
        combined = df
    
    # Add additional missing metrics
    try:
        from src.utils.additional_metrics import calculate_all_additional_metrics
        combined = calculate_all_additional_metrics(combined)
    except Exception as e:
        import logging
        logging.warning(f"Could not calculate additional metrics: {e}")
    
    # Calculate composite metrics (TOVA+, UPI, OPS 2.0) for hitters
    try:
        from src.utils.tova_metrics import calculate_all_composite_metrics
        combined = calculate_all_composite_metrics(combined)
    except Exception as e:
        import logging
        logging.warning(f"Could not calculate composite metrics (TOVA+, UPI, OPS 2.0): {e}")
    
    # Calculate UVS (Undervaluation Score) metrics for hitters
    try:
        from src.utils.uvs_metrics import calculate_all_uvs_metrics
        hitters_with_uvs = combined[combined.get('position_type', '') == 'Hitter'].copy()
        if len(hitters_with_uvs) > 0:
            hitters_with_uvs = calculate_all_uvs_metrics(hitters_with_uvs)
            # Update the combined dataframe
            combined.loc[combined.get('position_type', '') == 'Hitter', hitters_with_uvs.columns] = hitters_with_uvs
    except Exception as e:
        import logging
        logging.warning(f"Could not calculate UVS metrics: {e}")
    
    return combined

