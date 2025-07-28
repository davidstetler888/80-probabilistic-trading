#!/usr/bin/env python3
"""Improved SLTP with Quality Focus

This script replaces sltp.py with quality-focused SL/TP logic.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from config import config

# Generate SL_TP_PAIRS from config like original sltp.py
SPREAD = config.get('sl_tp_grid.spread')
SL_MULTIPLIERS = config.get('sl_tp_grid.sl_multipliers')
TP_MULTIPLIERS = config.get('sl_tp_grid.tp_multipliers')

# Create valid SL/TP combinations with quality focus
SL_TP_PAIRS = []
for sl_mult in SL_MULTIPLIERS:
    for tp_mult in TP_MULTIPLIERS:
        sl_pips = (SPREAD * sl_mult) / 0.0001  # Convert to pips
        tp_pips = sl_pips * tp_mult
        if tp_pips >= 2 * sl_pips:  # Only keep RR >= 2.0
            SL_TP_PAIRS.append((sl_pips, tp_pips))

# For quality focus, prefer tighter SL and better RR
# Add our preferred quality-focused pair at the beginning
QUALITY_PAIR = (10.0, 30.0)  # 10 pips SL, 30 pips TP (3:1 RR)
if QUALITY_PAIR not in SL_TP_PAIRS:
    SL_TP_PAIRS.insert(0, QUALITY_PAIR)

def apply_quality_sltp():
    """Apply quality-focused SL/TP logic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--train_end_date", required=True)
    parser.add_argument("--end_date", required=True)
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    
    # Load signals
    signals_path = run_dir / "data" / "signals.csv"
    if not signals_path.exists():
        print("❌ No signals found!")
        return False
    
    signals_df = pd.read_csv(signals_path, index_col=0)
    signals_df.index = pd.to_datetime(signals_df.index, utc=True).tz_localize(None)
    
    print(f"Processing {len(signals_df)} signals")
    
    # Apply quality-focused SL/TP
    # Use tighter SL and better RR ratios
    signals_df['sl_pips'] = 10.0  # Tighter SL
    signals_df['tp_pips'] = 30.0  # 3:1 RR
    signals_df['sl_points'] = 1.0
    signals_df['tp_points'] = 3.0
    
    # Save improved signals
    signals_df.to_csv(run_dir / "data" / "signals.csv")
    
    print("✅ Applied quality-focused SL/TP")
    print(f"  SL: {signals_df['sl_pips'].iloc[0]} pips")
    print(f"  TP: {signals_df['tp_pips'].iloc[0]} pips")
    print(f"  RR: {signals_df['tp_pips'].iloc[0]/signals_df['sl_pips'].iloc[0]:.1f}:1")
    
    return True


def main():
    """Main function for compatibility with optimization framework."""
    return apply_quality_sltp()


if __name__ == "__main__":
    success = apply_quality_sltp()
    sys.exit(0 if success else 1) 