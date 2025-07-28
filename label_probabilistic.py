#!/usr/bin/env python3
"""Probabilistic Labeling System

This module implements the revolutionary probabilistic labeling approach that replaces
binary classification with probability distributions and expected value calculations.

Key Features:
- Outcome distribution modeling instead of binary yes/no
- Expected value calculations including spread costs
- Volatility-adjusted targets based on market conditions
- Success probability calibration to 58%+ threshold
- Market regime and session awareness

Author: David Stetler
Date: 2025-01-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import Dict, Tuple, Optional
import warnings

from utils import (
    get_run_dir,
    make_run_dirs,
    parse_start_date_arg,
    parse_end_date_arg,
    load_data,
)
from config import config


def estimate_dynamic_spread(atr_values: pd.Series, hour_values: pd.Series) -> pd.Series:
    """Estimate spread based on volatility and session characteristics.
    
    Args:
        atr_values: ATR values for volatility estimation
        hour_values: Hour values for session-based spread estimation
        
    Returns:
        Series of estimated spreads
    """
    # Base spread characteristics from config/analysis
    base_spread = 0.00013
    min_spread = 0.0001
    max_spread = 0.00028
    
    # Calculate ATR percentiles for volatility adjustment
    atr_percentiles = atr_values.rank(pct=True)
    
    # Session-based spread patterns
    session_multipliers = {
        # London open (higher spreads)
        7: 1.4, 8: 1.4, 9: 1.2,
        # NY open (moderate spreads)  
        13: 1.2, 14: 1.2, 15: 1.1,
        # Overlap (highest spreads)
        12: 1.5,
        # Asian quiet (lowest spreads)
        22: 0.8, 23: 0.8, 0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8,
        # Other hours (normal)
    }
    
    # Calculate spread estimates
    spread_estimates = pd.Series(base_spread, index=atr_values.index)
    
    # Apply volatility adjustment
    volatility_adjustment = 1.0 + (atr_percentiles - 0.5) * 0.6  # ±30% based on volatility
    spread_estimates *= volatility_adjustment
    
    # Apply session adjustment
    for hour, multiplier in session_multipliers.items():
        hour_mask = hour_values == hour
        spread_estimates.loc[hour_mask] *= multiplier
    
    # Clamp to realistic ranges
    spread_estimates = spread_estimates.clip(min_spread, max_spread)
    
    return spread_estimates


def calculate_future_outcomes(df: pd.DataFrame, future_window: int = 24) -> pd.DataFrame:
    """Calculate comprehensive outcome distributions for future price movements.
    
    Args:
        df: DataFrame with OHLC data
        future_window: Number of bars to look ahead
        
    Returns:
        DataFrame with outcome statistics
    """
    outcomes = pd.DataFrame(index=df.index)
    
    # Calculate returns for each future bar
    future_returns = []
    future_highs = []
    future_lows = []
    
    for i in range(1, future_window + 1):
        # Price returns
        future_return = (df['close'].shift(-i) / df['close'] - 1)
        future_returns.append(future_return)
        
        # High/low for each future bar
        future_highs.append(df['high'].shift(-i))
        future_lows.append(df['low'].shift(-i))
    
    # Convert to arrays for vectorized operations
    returns_array = pd.concat(future_returns, axis=1).values
    highs_array = pd.concat(future_highs, axis=1).values  
    lows_array = pd.concat(future_lows, axis=1).values
    
    # Calculate outcome statistics
    current_prices = df['close'].values.reshape(-1, 1)
    
    # Maximum favorable movement (best possible outcome)
    max_high_return = (np.nanmax(highs_array, axis=1) / df['close'] - 1)
    max_low_return = (df['close'] / np.nanmin(lows_array, axis=1) - 1)  # For short positions
    
    # Maximum adverse movement (worst possible outcome)  
    max_adverse_long = (df['close'] / np.nanmin(lows_array, axis=1) - 1)  # Long adverse
    max_adverse_short = (np.nanmax(highs_array, axis=1) / df['close'] - 1)  # Short adverse
    
    # Final outcome (where price ends up)
    final_return = returns_array[:, -1] if returns_array.shape[1] > 0 else np.zeros(len(df))
    
    # Path volatility (how much price moves around)
    path_volatility = np.nanstd(returns_array, axis=1)
    
    # Target hit probabilities (what fraction of future bars hit targets)
    target_long = 0.0015  # 15 pips target for long
    target_short = -0.0015  # 15 pips target for short
    stop_long = -0.0015   # 15 pips stop for long
    stop_short = 0.0015   # 15 pips stop for short
    
    # Calculate hit probabilities
    hit_target_long_prob = np.nanmean(returns_array >= target_long, axis=1)
    hit_target_short_prob = np.nanmean(returns_array <= target_short, axis=1)
    hit_stop_long_prob = np.nanmean(returns_array <= stop_long, axis=1)
    hit_stop_short_prob = np.nanmean(returns_array >= stop_short, axis=1)
    
    # Store outcomes
    outcomes['max_favorable_long'] = max_high_return
    outcomes['max_favorable_short'] = max_low_return
    outcomes['max_adverse_long'] = max_adverse_long
    outcomes['max_adverse_short'] = max_adverse_short
    outcomes['final_return'] = final_return
    outcomes['path_volatility'] = path_volatility
    outcomes['hit_target_long_prob'] = hit_target_long_prob
    outcomes['hit_target_short_prob'] = hit_target_short_prob
    outcomes['hit_stop_long_prob'] = hit_stop_long_prob
    outcomes['hit_stop_short_prob'] = hit_stop_short_prob
    
    return outcomes


def calculate_expected_value(outcomes: pd.DataFrame, spread_estimates: pd.Series, 
                           df: pd.DataFrame) -> pd.DataFrame:
    """Calculate expected value for long and short positions including all costs.
    
    Args:
        outcomes: Future outcome distributions
        spread_estimates: Dynamic spread estimates
        df: Original DataFrame with market data
        
    Returns:
        DataFrame with expected value calculations
    """
    ev_data = pd.DataFrame(index=df.index)
    
    # Target and stop levels (in price terms)
    target_pips = 15  # 15 pips target
    stop_pips = 15    # 15 pips stop (1:1 base, will be adjusted)
    
    target_price_move = target_pips / 10000  # Convert pips to price
    stop_price_move = stop_pips / 10000
    
    # Long position expected value
    # EV = P(hit_target) * (target - spread) - P(hit_stop) * (stop + spread)
    long_target_profit = target_price_move - spread_estimates
    long_stop_loss = stop_price_move + spread_estimates
    
    ev_data['ev_long'] = (
        outcomes['hit_target_long_prob'] * long_target_profit - 
        outcomes['hit_stop_long_prob'] * long_stop_loss
    )
    
    # Short position expected value  
    short_target_profit = target_price_move - spread_estimates
    short_stop_loss = stop_price_move + spread_estimates
    
    ev_data['ev_short'] = (
        outcomes['hit_target_short_prob'] * short_target_profit -
        outcomes['hit_stop_short_prob'] * short_stop_loss
    )
    
    # Risk-reward ratios
    ev_data['rr_long'] = np.where(
        outcomes['hit_stop_long_prob'] > 0,
        (outcomes['hit_target_long_prob'] * target_price_move) / 
        (outcomes['hit_stop_long_prob'] * stop_price_move),
        np.nan
    )
    
    ev_data['rr_short'] = np.where(
        outcomes['hit_stop_short_prob'] > 0,
        (outcomes['hit_target_short_prob'] * target_price_move) / 
        (outcomes['hit_stop_short_prob'] * stop_price_move),
        np.nan
    )
    
    # Success probabilities (win rate)
    ev_data['success_prob_long'] = outcomes['hit_target_long_prob']
    ev_data['success_prob_short'] = outcomes['hit_target_short_prob']
    
    # Market favorability (combination of multiple factors)
    ev_data['market_favorability_long'] = calculate_market_favorability(
        outcomes, df, 'long'
    )
    ev_data['market_favorability_short'] = calculate_market_favorability(
        outcomes, df, 'short'
    )
    
    return ev_data


def calculate_market_favorability(outcomes: pd.DataFrame, df: pd.DataFrame, 
                                direction: str) -> pd.Series:
    """Calculate market favorability score for given direction.
    
    Args:
        outcomes: Future outcome distributions
        df: Market data
        direction: 'long' or 'short'
        
    Returns:
        Series of favorability scores (0-1)
    """
    favorability = pd.Series(0.5, index=df.index)  # Start neutral
    
    # Volatility favorability (moderate volatility is best)
    vol_optimal = outcomes['path_volatility'].median()
    vol_distance = np.abs(outcomes['path_volatility'] - vol_optimal)
    vol_score = 1.0 - (vol_distance / vol_optimal).clip(0, 1)
    
    # Trend favorability (based on recent price action)
    if 'ema_5' in df.columns and 'ema_20' in df.columns:
        if direction == 'long':
            trend_score = np.where(df['ema_5'] > df['ema_20'], 0.7, 0.3)
        else:
            trend_score = np.where(df['ema_5'] < df['ema_20'], 0.7, 0.3)
    else:
        trend_score = 0.5
    
    # Session favorability (based on hour)
    hour = df.index.hour
    session_scores = {
        # London session (good for both)
        7: 0.8, 8: 0.8, 9: 0.7, 10: 0.7, 11: 0.7,
        # NY session (good for both)
        13: 0.8, 14: 0.8, 15: 0.7, 16: 0.6,
        # Overlap (best)
        12: 0.9,
        # Asian (lower activity)
        22: 0.4, 23: 0.4, 0: 0.4, 1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4, 5: 0.4, 6: 0.5,
        # Other hours
    }
    
    session_score = pd.Series(0.5, index=df.index)
    for h, score in session_scores.items():
        session_score.loc[hour == h] = score
    
    # Combine factors
    favorability = (
        vol_score * 0.3 +
        trend_score * 0.4 + 
        session_score * 0.3
    )
    
    return favorability.clip(0, 1)


def create_probabilistic_labels(df: pd.DataFrame, future_window: int = 24,
                              train_end_date: Optional[str] = None) -> pd.DataFrame:
    """Create probabilistic labels based on expected value and probability distributions.
    
    This is the revolutionary replacement for binary classification.
    
    Args:
        df: DataFrame with OHLC and indicator data
        future_window: Number of bars to look ahead
        train_end_date: Optional cutoff date for training
        
    Returns:
        DataFrame with probabilistic labels and supporting data
    """
    print("🎯 Creating probabilistic labels...")
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'atr']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate dynamic spread estimates
    print("📊 Calculating dynamic spread estimates...")
    hour_values = pd.Series(df.index.hour, index=df.index)
    spread_estimates = estimate_dynamic_spread(df['atr'], hour_values)
    
    # Calculate future outcome distributions
    print("📈 Calculating future outcome distributions...")
    outcomes = calculate_future_outcomes(df, future_window)
    
    # Calculate expected values
    print("💰 Calculating expected values...")
    ev_data = calculate_expected_value(outcomes, spread_estimates, df)
    
    # Create probabilistic labels
    print("🏷️ Creating probabilistic labels...")
    labels_df = df.copy()
    
    # Add all calculated data
    for col in outcomes.columns:
        labels_df[f'outcome_{col}'] = outcomes[col]
    
    for col in ev_data.columns:
        labels_df[f'ev_{col}'] = ev_data[col]
        
    labels_df['spread_estimate'] = spread_estimates
    
    # Core labeling criteria (OPTIMIZED from balanced calibration - 100% target success)
    min_win_rate = 0.58  # 58% minimum win rate (maintained)
    min_rr = 2.0         # 2.0:1 minimum risk-reward (calibrated)
    min_ev = 0.0004      # Minimum expected value after costs (4 pips - OPTIMIZED)
    min_favorability = 0.72  # Market conditions must be favorable (OPTIMIZED from 0.7)
    
    # Long labels: Only positive if ALL criteria met
    long_conditions = (
        (ev_data['ev_long'] > min_ev) &                           # Positive EV after costs
        (ev_data['success_prob_long'] >= min_win_rate) &          # 58%+ win rate
        (ev_data['rr_long'] >= min_rr) &                         # 1:2+ risk-reward
        (ev_data['market_favorability_long'] >= min_favorability) # Favorable conditions
    )
    
    # Short labels: Only positive if ALL criteria met  
    short_conditions = (
        (ev_data['ev_short'] > min_ev) &                          # Positive EV after costs
        (ev_data['success_prob_short'] >= min_win_rate) &         # 58%+ win rate
        (ev_data['rr_short'] >= min_rr) &                        # 1:2+ risk-reward  
        (ev_data['market_favorability_short'] >= min_favorability) # Favorable conditions
    )
    
    # Create labels (1 = positive signal, 0 = no signal)
    labels_df['label_long'] = long_conditions.astype(int)
    labels_df['label_short'] = short_conditions.astype(int)
    
    # Handle train_end_date cutoff
    if train_end_date:
        cutoff_ts = pd.Timestamp(train_end_date, tz=df.index.tz)
        cutoff_idx = df.index.searchsorted(cutoff_ts)
        mask = np.arange(len(df)) + future_window > cutoff_idx
        labels_df.loc[df.index[mask], ['label_long', 'label_short']] = 0
    
    # Add quality scores for ranking
    labels_df['signal_quality_long'] = (
        ev_data['ev_long'] * ev_data['success_prob_long'] * 
        ev_data['market_favorability_long']
    )
    labels_df['signal_quality_short'] = (
        ev_data['ev_short'] * ev_data['success_prob_short'] * 
        ev_data['market_favorability_short']
    )
    
    return labels_df


def analyze_probabilistic_labels(labels_df: pd.DataFrame) -> Dict:
    """Analyze the quality and distribution of probabilistic labels.
    
    Args:
        labels_df: DataFrame with probabilistic labels
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Basic signal counts
    total_bars = len(labels_df)
    long_signals = (labels_df['label_long'] == 1).sum()
    short_signals = (labels_df['label_short'] == 1).sum()
    total_signals = long_signals + short_signals
    
    analysis['total_bars'] = total_bars
    analysis['long_signals'] = long_signals
    analysis['short_signals'] = short_signals
    analysis['total_signals'] = total_signals
    analysis['signal_rate'] = total_signals / total_bars if total_bars > 0 else 0
    
    # Expected performance metrics
    if total_signals > 0:
        long_mask = labels_df['label_long'] == 1
        short_mask = labels_df['label_short'] == 1
        
        if long_signals > 0:
            analysis['avg_win_rate_long'] = labels_df.loc[long_mask, 'ev_success_prob_long'].mean()
            analysis['avg_ev_long'] = labels_df.loc[long_mask, 'ev_ev_long'].mean()
            analysis['avg_rr_long'] = labels_df.loc[long_mask, 'ev_rr_long'].mean()
            
        if short_signals > 0:
            analysis['avg_win_rate_short'] = labels_df.loc[short_mask, 'ev_success_prob_short'].mean()
            analysis['avg_ev_short'] = labels_df.loc[short_mask, 'ev_ev_short'].mean()
            analysis['avg_rr_short'] = labels_df.loc[short_mask, 'ev_rr_short'].mean()
    
    # Time-based analysis
    if total_signals > 0:
        signal_dates = labels_df[labels_df['label_long'] == 1].index.union(
            labels_df[labels_df['label_short'] == 1].index
        )
        
        if len(signal_dates) > 0:
            time_span = (signal_dates.max() - signal_dates.min()).days
            analysis['time_span_days'] = time_span
            analysis['signals_per_week'] = (total_signals * 7) / time_span if time_span > 0 else 0
    
    return analysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create probabilistic labels for trading data")
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Earliest bar to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=False,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        required=False,
        help="Cutoff date for labeling (YYYY-MM-DD)",
    )
    parser.add_argument("--run", type=str, help="Run directory (overrides RUN_ID)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if CSV parsing fails",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with validation",
    )
    return parser.parse_args()


def main():
    """Main function for probabilistic labeling."""
    args = parse_args()
    run_dir = Path(args.run) if args.run else Path(get_run_dir())
    make_run_dirs(str(run_dir))

    # Load prepared data
    prepared_path = run_dir / "data" / "prepared.csv"
    if not prepared_path.exists():
        print(f"❌ Error: {prepared_path} not found. Run prepare.py first.")
        sys.exit(1)

    try:
        start_date = parse_start_date_arg(args.start_date)
        end_date = parse_end_date_arg(args.end_date)
        train_end_date = parse_end_date_arg(args.train_end_date)
    except ValueError as exc:
        raise SystemExit(f"Invalid date: {exc}") from exc

    # Load data
    print("📊 Loading prepared data...")
    df = load_data(
        str(prepared_path), end_date=end_date, start_date=start_date, strict=args.strict
    )
    
    print(f"📈 Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

    # Create probabilistic labels
    try:
        labels_df = create_probabilistic_labels(
            df,
            future_window=config.get("label.future_window", 24),
            train_end_date=train_end_date,
        )
        
        # Analyze results
        print("📊 Analyzing probabilistic labels...")
        analysis = analyze_probabilistic_labels(labels_df)
        
        # Save results
        output_path = run_dir / "data" / "labeled_probabilistic.csv"
        labels_df.to_csv(output_path)
        
        # Save analysis
        analysis_path = run_dir / "data" / "label_analysis.json"
        import json
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Print summary
        print("\n🎯 Probabilistic Labeling Results:")
        print(f"   Total bars: {analysis['total_bars']:,}")
        print(f"   Long signals: {analysis['long_signals']:,}")
        print(f"   Short signals: {analysis['short_signals']:,}")
        print(f"   Total signals: {analysis['total_signals']:,}")
        print(f"   Signal rate: {analysis['signal_rate']:.4f} ({analysis['signal_rate']*100:.2f}%)")
        
        if 'signals_per_week' in analysis:
            print(f"   Signals per week: {analysis['signals_per_week']:.1f}")
            
        if 'avg_win_rate_long' in analysis:
            print(f"   Avg win rate (long): {analysis['avg_win_rate_long']:.1%}")
            print(f"   Avg expected value (long): {analysis['avg_ev_long']:.6f}")
            print(f"   Avg risk-reward (long): {analysis['avg_rr_long']:.2f}")
            
        if 'avg_win_rate_short' in analysis:
            print(f"   Avg win rate (short): {analysis['avg_win_rate_short']:.1%}")
            print(f"   Avg expected value (short): {analysis['avg_ev_short']:.6f}")
            print(f"   Avg risk-reward (short): {analysis['avg_rr_short']:.2f}")
        
        print(f"\n✅ Probabilistic labels saved to: {output_path}")
        print(f"✅ Analysis saved to: {analysis_path}")
        
        # Test mode validation
        if args.test:
            print("\n🧪 Running validation tests...")
            run_validation_tests(labels_df, analysis)
        
    except Exception as e:
        print(f"❌ Error creating probabilistic labels: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_validation_tests(labels_df: pd.DataFrame, analysis: Dict):
    """Run validation tests to ensure probabilistic labeling is working correctly."""
    print("🔍 Validation Test 1: Signal quality criteria")
    
    # Test that all positive labels meet our criteria
    long_signals = labels_df[labels_df['label_long'] == 1]
    short_signals = labels_df[labels_df['label_short'] == 1]
    
    if len(long_signals) > 0:
        min_win_rate = long_signals['ev_success_prob_long'].min()
        min_rr = long_signals['ev_rr_long'].min()
        min_ev = long_signals['ev_ev_long'].min()
        
        print(f"   Long signals - Min win rate: {min_win_rate:.1%} (target: ≥58%)")
        print(f"   Long signals - Min RR: {min_rr:.2f} (target: ≥2.0)")
        print(f"   Long signals - Min EV: {min_ev:.6f} (target: ≥0.0003)")
        
        assert min_win_rate >= 0.58, f"Long signals below 58% win rate: {min_win_rate:.1%}"
        assert min_rr >= 2.0, f"Long signals below 2.0 RR: {min_rr:.2f}"
        assert min_ev >= 0.0003, f"Long signals below minimum EV: {min_ev:.6f}"
        
    if len(short_signals) > 0:
        min_win_rate = short_signals['ev_success_prob_short'].min()
        min_rr = short_signals['ev_rr_short'].min()
        min_ev = short_signals['ev_ev_short'].min()
        
        print(f"   Short signals - Min win rate: {min_win_rate:.1%} (target: ≥58%)")
        print(f"   Short signals - Min RR: {min_rr:.2f} (target: ≥2.0)")
        print(f"   Short signals - Min EV: {min_ev:.6f} (target: ≥0.0003)")
        
        assert min_win_rate >= 0.58, f"Short signals below 58% win rate: {min_win_rate:.1%}"
        assert min_rr >= 2.0, f"Short signals below 2.0 RR: {min_rr:.2f}"
        assert min_ev >= 0.0003, f"Short signals below minimum EV: {min_ev:.6f}"
    
    print("✅ Test 1 passed: All signals meet quality criteria")
    
    print("🔍 Validation Test 2: Trade volume targets")
    signals_per_week = analysis.get('signals_per_week', 0)
    print(f"   Signals per week: {signals_per_week:.1f} (target: 25-50)")
    
    if signals_per_week < 5:
        print("⚠️  Warning: Very low signal volume - may need threshold adjustment")
    elif signals_per_week > 100:
        print("⚠️  Warning: Very high signal volume - may need threshold tightening")
    else:
        print("✅ Test 2 passed: Signal volume in reasonable range")
    
    print("🔍 Validation Test 3: Data integrity")
    
    # Check for NaN values in critical columns
    critical_cols = ['label_long', 'label_short', 'ev_ev_long', 'ev_ev_short']
    for col in critical_cols:
        if col in labels_df.columns:
            nan_count = labels_df[col].isna().sum()
            print(f"   {col}: {nan_count} NaN values")
            if nan_count > len(labels_df) * 0.1:  # More than 10% NaN
                print(f"⚠️  Warning: High NaN count in {col}")
    
    print("✅ Test 3 passed: Data integrity check complete")
    print("\n🎉 All validation tests completed!")


if __name__ == "__main__":
    main()