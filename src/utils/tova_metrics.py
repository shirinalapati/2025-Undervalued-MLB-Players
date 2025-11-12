"""
Calculate TOVA+, UPI, OPS 2.0, TOVA$, BOV, and BOV_power composite metrics for hitters.

These are advanced composite metrics that combine multiple statistics using z-scores
to provide holistic views of player performance. While the main project uses UVS
(Undervaluation Score) as the primary metric, these additional metrics provide
alternative perspectives on player value and performance.

Metrics:
- TOVA+ (True Offensive Value Added Plus): Comprehensive offensive value metric
- UPI (Ultimate Performance Index): Performance potential metric
- OPS 2.0 (Machine-Readable OPS): Enhanced version of traditional OPS
- TOVA$: Cost-efficiency version of TOVA+ (TOVA+ / Salary in millions)
- BOV (Best Overall Value): Balanced value metric
- BOV_power: Power-focused version of BOV (weighted toward expected production)
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_z_score(series: pd.Series) -> pd.Series:
    """Calculate z-scores for a series (standardize to mean=0, std=1)."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def calculate_contact_quality(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Contact Quality composite score.
    
    Contact Quality = (z(Barrel%) + z(HardHit%) + z(Exit Velo) + z(Sweet Spot%)) / 4
    """
    metrics = []
    
    # Barrel%
    if 'barrel_batted_rate' in df.columns:
        barrel = df['barrel_batted_rate']
    elif 'Barrel%' in df.columns:
        barrel = df['Barrel%']
    else:
        barrel = pd.Series(0.0, index=df.index)
    
    # HardHit%
    if 'hard_hit_percent' in df.columns:
        hardhit = df['hard_hit_percent']
    elif 'HardHit%' in df.columns:
        hardhit = df['HardHit%']
    else:
        hardhit = pd.Series(0.0, index=df.index)
    
    # Exit Velo
    if 'avg_exit_velocity' in df.columns:
        exit_velo = df['avg_exit_velocity']
    else:
        exit_velo = pd.Series(0.0, index=df.index)
    
    # Sweet Spot%
    if 'sweet_spot_percent' in df.columns:
        sweet_spot = df['sweet_spot_percent']
    else:
        sweet_spot = pd.Series(0.0, index=df.index)
    
    # Calculate z-scores and average
    z_barrel = calculate_z_score(barrel.fillna(0))
    z_hardhit = calculate_z_score(hardhit.fillna(0))
    z_exit_velo = calculate_z_score(exit_velo.fillna(0))
    z_sweet_spot = calculate_z_score(sweet_spot.fillna(0))
    
    contact_quality = (z_barrel + z_hardhit + z_exit_velo + z_sweet_spot) / 4
    
    return contact_quality


def calculate_plate_discipline(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Plate Discipline composite score.
    
    Plate Discipline = z(BB%) – z(K%) – z(Chase%) + z(Z-Contact%)
    """
    # BB%
    if 'bb_percent' in df.columns:
        bb_pct = df['bb_percent']
    elif 'BB%' in df.columns:
        bb_pct = df['BB%']
    else:
        bb_pct = pd.Series(0.0, index=df.index)
    
    # K%
    if 'k_percent' in df.columns:
        k_pct = df['k_percent']
    elif 'K%' in df.columns:
        k_pct = df['K%']
    else:
        k_pct = pd.Series(0.0, index=df.index)
    
    # Chase% (O-Swing%)
    if 'o_swing_percent' in df.columns:
        chase = df['o_swing_percent']
    elif 'O-Swing%' in df.columns:
        chase = df['O-Swing%']
    else:
        chase = pd.Series(0.0, index=df.index)
    
    # Z-Contact%
    if 'z_contact_percent' in df.columns:
        z_contact = df['z_contact_percent']
    elif 'Z-Contact%' in df.columns:
        z_contact = df['Z-Contact%']
    else:
        z_contact = pd.Series(0.0, index=df.index)
    
    # Calculate z-scores
    z_bb = calculate_z_score(bb_pct.fillna(0))
    z_k = calculate_z_score(k_pct.fillna(0))
    z_chase = calculate_z_score(chase.fillna(0))
    z_z_contact = calculate_z_score(z_contact.fillna(0))
    
    plate_discipline = z_bb - z_k - z_chase + z_z_contact
    
    return plate_discipline


def calculate_luck_index_component(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Luck Index component.
    
    Luck Index = (xwOBA – wOBA) + (xBA – BA)
    """
    # xwOBA - wOBA
    if 'xwoba' in df.columns or 'est_woba' in df.columns:
        xwoba = df.get('xwoba', df.get('est_woba', pd.Series(0.0, index=df.index)))
    else:
        xwoba = pd.Series(0.0, index=df.index)
    
    if 'woba' in df.columns or 'wOBA' in df.columns:
        woba = df.get('woba', df.get('wOBA', pd.Series(0.0, index=df.index)))
    else:
        woba = pd.Series(0.0, index=df.index)
    
    # xBA - BA
    if 'xba' in df.columns or 'est_ba' in df.columns:
        xba = df.get('xba', df.get('est_ba', pd.Series(0.0, index=df.index)))
    else:
        xba = pd.Series(0.0, index=df.index)
    
    if 'ba' in df.columns or 'BA' in df.columns:
        ba = df.get('ba', df.get('BA', pd.Series(0.0, index=df.index)))
    else:
        ba = pd.Series(0.0, index=df.index)
    
    luck_index = (xwoba.fillna(0) - woba.fillna(0)) + (xba.fillna(0) - ba.fillna(0))
    
    return luck_index


def calculate_run_value_component(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Run Value composite score.
    
    Run Value = (z(Run Value / Pitch) + z(Run Value / Swing)) / 2
    
    Note: If Run Value per Pitch/Swing data is not available,
    we'll use a placeholder or calculate from available metrics.
    """
    # Try to find run value columns
    run_value_pitch = None
    run_value_swing = None
    
    for col in df.columns:
        if 'run' in col.lower() and 'value' in col.lower() and 'pitch' in col.lower():
            run_value_pitch = df[col]
        elif 'run' in col.lower() and 'value' in col.lower() and 'swing' in col.lower():
            run_value_swing = df[col]
    
    # If not available, use placeholder (all zeros)
    if run_value_pitch is None:
        run_value_pitch = pd.Series(0.0, index=df.index)
    if run_value_swing is None:
        run_value_swing = pd.Series(0.0, index=df.index)
    
    z_pitch = calculate_z_score(run_value_pitch.fillna(0))
    z_swing = calculate_z_score(run_value_swing.fillna(0))
    
    run_value = (z_pitch + z_swing) / 2
    
    return run_value


def calculate_balance_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Balance Score.
    
    Balance Score = -| z(GB%) – z(LD%) | – | z(Pull%) – z(Oppo%) |
    """
    # GB%
    if 'gb_percent' in df.columns:
        gb = df['gb_percent']
    elif 'GB%' in df.columns:
        gb = df['GB%']
    else:
        gb = pd.Series(0.0, index=df.index)
    
    # LD%
    if 'ld_percent' in df.columns:
        ld = df['ld_percent']
    elif 'LD%' in df.columns:
        ld = df['LD%']
    else:
        ld = pd.Series(0.0, index=df.index)
    
    # Pull%
    if 'pull_percent' in df.columns:
        pull = df['pull_percent']
    elif 'Pull%' in df.columns:
        pull = df['Pull%']
    else:
        pull = pd.Series(0.0, index=df.index)
    
    # Oppo%
    if 'oppo_percent' in df.columns:
        oppo = df['oppo_percent']
    elif 'Oppo%' in df.columns:
        oppo = df['Oppo%']
    else:
        oppo = pd.Series(0.0, index=df.index)
    
    z_gb = calculate_z_score(gb.fillna(0))
    z_ld = calculate_z_score(ld.fillna(0))
    z_pull = calculate_z_score(pull.fillna(0))
    z_oppo = calculate_z_score(oppo.fillna(0))
    
    balance_score = -abs(z_gb - z_ld) - abs(z_pull - z_oppo)
    
    return balance_score


def calculate_tova_plus(df: pd.DataFrame) -> pd.Series:
    """
    Calculate TOVA+ (True Offensive Value Added Plus).
    
    TOVA+ = 0.15 z(xwOBA) + 0.10 z(xSLG) + 0.10 z(xISO) + 0.10 z(Contact Quality)
          + 0.10 z(Plate Discipline) + 0.10 z(Luck Index) + 0.10 z(Run Value)
          + 0.10 z(Balance Score) + 0.05 z(wRC+) + 0.10 z(xBA)
    """
    # Calculate sub-components
    contact_quality = calculate_contact_quality(df)
    plate_discipline = calculate_plate_discipline(df)
    luck_index = calculate_luck_index_component(df)
    run_value = calculate_run_value_component(df)
    balance_score = calculate_balance_score(df)
    
    # Get individual metrics
    # xwOBA
    if 'xwoba' in df.columns or 'est_woba' in df.columns:
        xwoba = df.get('xwoba', df.get('est_woba', pd.Series(0.0, index=df.index)))
    else:
        xwoba = pd.Series(0.0, index=df.index)
    
    # xSLG
    if 'xslg' in df.columns or 'est_slg' in df.columns:
        xslg = df.get('xslg', df.get('est_slg', pd.Series(0.0, index=df.index)))
    else:
        xslg = pd.Series(0.0, index=df.index)
    
    # xISO
    if 'xiso' in df.columns or 'xISO' in df.columns:
        xiso = df.get('xiso', df.get('xISO', pd.Series(0.0, index=df.index)))
    else:
        xiso = pd.Series(0.0, index=df.index)
    
    # xBA
    if 'xba' in df.columns or 'est_ba' in df.columns:
        xba = df.get('xba', df.get('est_ba', pd.Series(0.0, index=df.index)))
    else:
        xba = pd.Series(0.0, index=df.index)
    
    # wRC+
    if 'wrc_plus' in df.columns or 'wRC+' in df.columns:
        wrc_plus = df.get('wrc_plus', df.get('wRC+', pd.Series(0.0, index=df.index)))
    else:
        wrc_plus = pd.Series(0.0, index=df.index)
    
    # Calculate z-scores
    z_xwoba = calculate_z_score(xwoba.fillna(0))
    z_xslg = calculate_z_score(xslg.fillna(0))
    z_xiso = calculate_z_score(xiso.fillna(0))
    z_xba = calculate_z_score(xba.fillna(0))
    z_wrc_plus = calculate_z_score(wrc_plus.fillna(0))
    z_contact_quality = calculate_z_score(contact_quality)
    z_plate_discipline = calculate_z_score(plate_discipline)
    z_luck_index = calculate_z_score(luck_index)
    z_run_value = calculate_z_score(run_value)
    z_balance_score = calculate_z_score(balance_score)
    
    # Calculate TOVA+
    tova_plus = (
        0.15 * z_xwoba +
        0.10 * z_xslg +
        0.10 * z_xiso +
        0.10 * z_contact_quality +
        0.10 * z_plate_discipline +
        0.10 * z_luck_index +
        0.10 * z_run_value +
        0.10 * z_balance_score +
        0.05 * z_wrc_plus +
        0.10 * z_xba
    )
    
    return tova_plus


def calculate_ups(df: pd.DataFrame) -> pd.Series:
    """
    Calculate UPI (Ultimate Performance Index).
    
    UPI = 0.25 z(xwOBA – wOBA) + 0.10 z(xBA – BA) + 0.10 z(xSLG – SLG)
        + 0.15 z(Contact Quality) + 0.15 z(Plate Discipline) + 0.10 z(Run Value)
        + 0.05 z(Balance Score) + 0.10 z(wRC+)
    """
    # Calculate sub-components
    contact_quality = calculate_contact_quality(df)
    plate_discipline = calculate_plate_discipline(df)
    run_value = calculate_run_value_component(df)
    balance_score = calculate_balance_score(df)
    
    # Get metrics for differences
    # xwOBA - wOBA
    if 'xwoba' in df.columns or 'est_woba' in df.columns:
        xwoba = df.get('xwoba', df.get('est_woba', pd.Series(0.0, index=df.index)))
    else:
        xwoba = pd.Series(0.0, index=df.index)
    
    if 'woba' in df.columns or 'wOBA' in df.columns:
        woba = df.get('woba', df.get('wOBA', pd.Series(0.0, index=df.index)))
    else:
        woba = pd.Series(0.0, index=df.index)
    
    # xBA - BA
    if 'xba' in df.columns or 'est_ba' in df.columns:
        xba = df.get('xba', df.get('est_ba', pd.Series(0.0, index=df.index)))
    else:
        xba = pd.Series(0.0, index=df.index)
    
    if 'ba' in df.columns or 'BA' in df.columns:
        ba = df.get('ba', df.get('BA', pd.Series(0.0, index=df.index)))
    else:
        ba = pd.Series(0.0, index=df.index)
    
    # xSLG - SLG
    if 'xslg' in df.columns or 'est_slg' in df.columns:
        xslg = df.get('xslg', df.get('est_slg', pd.Series(0.0, index=df.index)))
    else:
        xslg = pd.Series(0.0, index=df.index)
    
    if 'slg' in df.columns or 'SLG' in df.columns:
        slg = df.get('slg', df.get('SLG', pd.Series(0.0, index=df.index)))
    else:
        slg = pd.Series(0.0, index=df.index)
    
    # wRC+
    if 'wrc_plus' in df.columns or 'wRC+' in df.columns:
        wrc_plus = df.get('wrc_plus', df.get('wRC+', pd.Series(0.0, index=df.index)))
    else:
        wrc_plus = pd.Series(0.0, index=df.index)
    
    # Calculate differences
    xwoba_diff = (xwoba.fillna(0) - woba.fillna(0))
    xba_diff = (xba.fillna(0) - ba.fillna(0))
    xslg_diff = (xslg.fillna(0) - slg.fillna(0))
    
    # Calculate z-scores
    z_xwoba_diff = calculate_z_score(xwoba_diff)
    z_xba_diff = calculate_z_score(xba_diff)
    z_xslg_diff = calculate_z_score(xslg_diff)
    z_contact_quality = calculate_z_score(contact_quality)
    z_plate_discipline = calculate_z_score(plate_discipline)
    z_run_value = calculate_z_score(run_value)
    z_balance_score = calculate_z_score(balance_score)
    z_wrc_plus = calculate_z_score(wrc_plus.fillna(0))
    
    # Calculate UPI
    upi = (
        0.25 * z_xwoba_diff +
        0.10 * z_xba_diff +
        0.10 * z_xslg_diff +
        0.15 * z_contact_quality +
        0.15 * z_plate_discipline +
        0.10 * z_run_value +
        0.05 * z_balance_score +
        0.10 * z_wrc_plus
    )
    
    return upi


def calculate_ops_2_0(df: pd.DataFrame) -> pd.Series:
    """
    Calculate OPS 2.0 (Machine-Readable OPS).
    
    OPS 2.0 = 0.4 z(xwOBA) + 0.2 z(xSLG) + 0.15 z(Barrel%) + 0.1 z(BB%)
            - 0.1 z(K%) + 0.05 z(Sweet Spot%)
    """
    # xwOBA
    if 'xwoba' in df.columns or 'est_woba' in df.columns:
        xwoba = df.get('xwoba', df.get('est_woba', pd.Series(0.0, index=df.index)))
    else:
        xwoba = pd.Series(0.0, index=df.index)
    
    # xSLG
    if 'xslg' in df.columns or 'est_slg' in df.columns:
        xslg = df.get('xslg', df.get('est_slg', pd.Series(0.0, index=df.index)))
    else:
        xslg = pd.Series(0.0, index=df.index)
    
    # Barrel%
    if 'barrel_batted_rate' in df.columns:
        barrel = df['barrel_batted_rate']
    elif 'Barrel%' in df.columns:
        barrel = df['Barrel%']
    else:
        barrel = pd.Series(0.0, index=df.index)
    
    # BB%
    if 'bb_percent' in df.columns:
        bb_pct = df['bb_percent']
    elif 'BB%' in df.columns:
        bb_pct = df['BB%']
    else:
        bb_pct = pd.Series(0.0, index=df.index)
    
    # K%
    if 'k_percent' in df.columns:
        k_pct = df['k_percent']
    elif 'K%' in df.columns:
        k_pct = df['K%']
    else:
        k_pct = pd.Series(0.0, index=df.index)
    
    # Sweet Spot%
    if 'sweet_spot_percent' in df.columns:
        sweet_spot = df['sweet_spot_percent']
    else:
        sweet_spot = pd.Series(0.0, index=df.index)
    
    # Calculate z-scores
    z_xwoba = calculate_z_score(xwoba.fillna(0))
    z_xslg = calculate_z_score(xslg.fillna(0))
    z_barrel = calculate_z_score(barrel.fillna(0))
    z_bb = calculate_z_score(bb_pct.fillna(0))
    z_k = calculate_z_score(k_pct.fillna(0))
    z_sweet_spot = calculate_z_score(sweet_spot.fillna(0))
    
    # Calculate OPS 2.0
    ops_2_0 = (
        0.4 * z_xwoba +
        0.2 * z_xslg +
        0.15 * z_barrel +
        0.1 * z_bb -
        0.1 * z_k +
        0.05 * z_sweet_spot
    )
    
    return ops_2_0


def calculate_expected_production(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Expected Production composite score.
    
    Expected Production = (z(xwOBA) + z(xSLG) + z(xISO)) / 3
    """
    # xwOBA - prioritize columns with actual data
    if 'est_woba' in df.columns and df['est_woba'].notna().any():
        xwoba = df['est_woba']
    elif 'xwoba' in df.columns and df['xwoba'].notna().any():
        xwoba = df['xwoba']
    else:
        xwoba = pd.Series(0.0, index=df.index)
    
    # xSLG - prioritize columns with actual data
    if 'est_slg' in df.columns and df['est_slg'].notna().any():
        xslg = df['est_slg']
    elif 'xslg' in df.columns and df['xslg'].notna().any():
        xslg = df['xslg']
    else:
        xslg = pd.Series(0.0, index=df.index)
    
    # xISO - check if exists, otherwise calculate from xSLG - xBA
    if 'xiso' in df.columns and df['xiso'].notna().any():
        xiso = df['xiso']
    elif 'xISO' in df.columns and df['xISO'].notna().any():
        xiso = df['xISO']
    else:
        # Calculate xISO = xSLG - xBA
        if 'est_ba' in df.columns and df['est_ba'].notna().any():
            xba = df['est_ba']
        elif 'xba' in df.columns and df['xba'].notna().any():
            xba = df['xba']
        else:
            xba = pd.Series(0.0, index=df.index)
        xiso = xslg - xba
    
    # Calculate z-scores and average
    z_xwoba = calculate_z_score(xwoba.fillna(0))
    z_xslg = calculate_z_score(xslg.fillna(0))
    z_xiso = calculate_z_score(xiso.fillna(0))
    
    expected_production = (z_xwoba + z_xslg + z_xiso) / 3
    
    return expected_production


def calculate_run_value_component_v2(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Run Value composite score (version 2 for BOV).
    
    Run Value = (z(Run Value / Pitch) + z(Run Value / Swing) + z(wRC+)) / 3
    """
    # Try to find run value columns
    run_value_pitch = None
    run_value_swing = None
    
    for col in df.columns:
        if 'run' in col.lower() and 'value' in col.lower() and 'pitch' in col.lower():
            run_value_pitch = df[col]
        elif 'run' in col.lower() and 'value' in col.lower() and 'swing' in col.lower():
            run_value_swing = df[col]
    
    # If not available, use placeholder (all zeros)
    if run_value_pitch is None:
        run_value_pitch = pd.Series(0.0, index=df.index)
    if run_value_swing is None:
        run_value_swing = pd.Series(0.0, index=df.index)
    
    # wRC+
    if 'wrc_plus' in df.columns or 'wRC+' in df.columns:
        wrc_plus = df.get('wrc_plus', df.get('wRC+', pd.Series(0.0, index=df.index)))
    else:
        wrc_plus = pd.Series(0.0, index=df.index)
    
    z_pitch = calculate_z_score(run_value_pitch.fillna(0))
    z_swing = calculate_z_score(run_value_swing.fillna(0))
    z_wrc_plus = calculate_z_score(wrc_plus.fillna(0))
    
    run_value = (z_pitch + z_swing + z_wrc_plus) / 3
    
    return run_value


def calculate_bov(df: pd.DataFrame) -> pd.Series:
    """
    Calculate BOV (Best Overall Value).
    
    BOV = 0.40 z(Expected Production)
        + 0.25 z(Contact Quality)
        + 0.10 z(Plate Discipline)
        + 0.15 z(Run Value)
        + 0.05 z(Balance Score)
        + 0.05 z(Luck Index)
    """
    # Calculate sub-components
    expected_production = calculate_expected_production(df)
    contact_quality = calculate_contact_quality(df)
    plate_discipline = calculate_plate_discipline(df)
    run_value = calculate_run_value_component_v2(df)
    balance_score = calculate_balance_score(df)
    luck_index = calculate_luck_index_component(df)
    
    # Calculate z-scores
    z_expected_prod = calculate_z_score(expected_production)
    z_contact_quality = calculate_z_score(contact_quality)
    z_plate_discipline = calculate_z_score(plate_discipline)
    z_run_value = calculate_z_score(run_value)
    z_balance_score = calculate_z_score(balance_score)
    z_luck_index = calculate_z_score(luck_index)
    
    # Calculate BOV
    bov = (
        0.40 * z_expected_prod +
        0.25 * z_contact_quality +
        0.10 * z_plate_discipline +
        0.15 * z_run_value +
        0.05 * z_balance_score +
        0.05 * z_luck_index
    )
    
    return bov


def calculate_bov_power(df: pd.DataFrame) -> pd.Series:
    """
    Calculate BOV_power (Best Overall Value - Power focused).
    
    BOV_power = 0.50 z(Expected Production)
              + 0.25 z(Contact Quality)
              + 0.10 z(Plate Discipline)
              + 0.10 z(Run Value)
              + 0.03 z(Balance Score)
              + 0.02 z(Luck Index)
    """
    # Calculate sub-components
    expected_production = calculate_expected_production(df)
    contact_quality = calculate_contact_quality(df)
    plate_discipline = calculate_plate_discipline(df)
    run_value = calculate_run_value_component_v2(df)
    balance_score = calculate_balance_score(df)
    luck_index = calculate_luck_index_component(df)
    
    # Calculate z-scores
    z_expected_prod = calculate_z_score(expected_production)
    z_contact_quality = calculate_z_score(contact_quality)
    z_plate_discipline = calculate_z_score(plate_discipline)
    z_run_value = calculate_z_score(run_value)
    z_balance_score = calculate_z_score(balance_score)
    z_luck_index = calculate_z_score(luck_index)
    
    # Calculate BOV_power
    bov_power = (
        0.50 * z_expected_prod +
        0.25 * z_contact_quality +
        0.10 * z_plate_discipline +
        0.10 * z_run_value +
        0.03 * z_balance_score +
        0.02 * z_luck_index
    )
    
    return bov_power


def calculate_tova_dollar(df: pd.DataFrame) -> pd.Series:
    """
    Calculate TOVA$ (cost-efficiency version of TOVA+).
    
    TOVA$ = TOVA+ / (Salary / 1M$)
    
    This measures TOVA+ per million dollars of salary.
    Higher values indicate better value (more production per dollar).
    """
    # Get TOVA+ (calculate if not present)
    if 'tova_plus' not in df.columns:
        tova_plus = calculate_tova_plus(df)
    else:
        tova_plus = df['tova_plus']
    
    # Get salary in millions
    if 'salary_2025' in df.columns:
        salary_millions = df['salary_2025']
    else:
        salary_millions = pd.Series(np.nan, index=df.index)
    
    # Calculate TOVA$ = TOVA+ / (Salary / 1M)
    # Since salary is already in millions, this is just TOVA+ / salary_millions
    # But we need to handle division by zero and missing salaries
    tova_dollar = tova_plus / (salary_millions + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Set to NaN where salary is missing or zero
    tova_dollar = tova_dollar.where(salary_millions.notna() & (salary_millions > 0), np.nan)
    
    return tova_dollar


def calculate_all_composite_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all composite metrics (TOVA+, UPI, OPS 2.0, TOVA$, BOV, BOV_power) for hitters.
    
    Args:
        df: DataFrame with player statistics
        
    Returns:
        DataFrame with added columns: tova_plus, upi, ops_2_0, tova_dollar, bov, bov_power
    """
    df = df.copy()
    
    # Only calculate for hitters
    hitters_mask = df.get('position_type', '') == 'Hitter'
    
    if hitters_mask.any():
        hitters = df[hitters_mask].copy()
        
        # Calculate metrics
        hitters['tova_plus'] = calculate_tova_plus(hitters)
        hitters['upi'] = calculate_ups(hitters)
        hitters['ops_2_0'] = calculate_ops_2_0(hitters)
        hitters['tova_dollar'] = calculate_tova_dollar(hitters)
        hitters['bov'] = calculate_bov(hitters)
        hitters['bov_power'] = calculate_bov_power(hitters)
        
        # Update main dataframe
        df.loc[hitters_mask, 'tova_plus'] = hitters['tova_plus']
        df.loc[hitters_mask, 'upi'] = hitters['upi']
        df.loc[hitters_mask, 'ops_2_0'] = hitters['ops_2_0']
        df.loc[hitters_mask, 'tova_dollar'] = hitters['tova_dollar']
        df.loc[hitters_mask, 'bov'] = hitters['bov']
        df.loc[hitters_mask, 'bov_power'] = hitters['bov_power']
    else:
        # No hitters, set to NaN
        df['tova_plus'] = np.nan
        df['upi'] = np.nan
        df['ops_2_0'] = np.nan
        df['tova_dollar'] = np.nan
        df['bov'] = np.nan
        df['bov_power'] = np.nan
    
    return df

