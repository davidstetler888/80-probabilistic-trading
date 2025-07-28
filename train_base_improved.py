#!/usr/bin/env python3
"""Improved Base Trainer with Quality Focus

This script replaces train_base.py with quality-focused training.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import argparse
import sys
from pathlib import Path

def train_with_quality_focus():
    """Train models with focus on high-quality signals."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--train_end_date", required=True)
    parser.add_argument("--end_date", required=True)
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    
    # Load data
    prepared_df = pd.read_csv(run_dir / "data" / "prepared.csv", index_col=0)
    prepared_df.index = pd.to_datetime(prepared_df.index, utc=True).tz_localize(None)
    
    # Load labels
    labels_df = pd.read_csv(run_dir / "data" / "labeled.csv", index_col=0)
    labels_df.index = pd.to_datetime(labels_df.index, utc=True).tz_localize(None)
    
    # Align data
    common_index = prepared_df.index.intersection(labels_df.index)
    prepared_aligned = prepared_df.loc[common_index]
    labels_aligned = labels_df.loc[common_index]
    
    print(f"Training with {len(prepared_aligned)} samples")
    
    # Prepare features and labels
    feature_cols = [col for col in prepared_aligned.columns if col not in ['label_long', 'label_short']]
    X = prepared_aligned[feature_cols]
    y_long = labels_aligned['label_long']
    y_short = labels_aligned['label_short']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Long labels: {y_long.sum()}")
    print(f"Short labels: {y_short.sum()}")
    
    # Train models with quality focus
    print("\n=== Training Quality-Focused Models ===")
    
    # Train long model
    print("Training long model...")
    X_train_long, X_test_long, y_train_long, y_test_long = train_test_split(
        X, y_long, test_size=0.2, random_state=42, stratify=y_long
    )
    
    train_data_long = lgb.Dataset(X_train_long, label=y_train_long)
    val_data_long = lgb.Dataset(X_test_long, label=y_test_long, reference=train_data_long)
    
    long_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': -1,
        'early_stopping_rounds': 100,
        'num_boost_round': 1000,
        'min_data_in_leaf': 50,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'scale_pos_weight': 5
    }
    
    long_model = lgb.train(
        long_params,
        train_data_long,
        valid_sets=[val_data_long],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(False)]
    )
    
    # Train short model
    print("Training short model...")
    X_train_short, X_test_short, y_train_short, y_test_short = train_test_split(
        X, y_short, test_size=0.2, random_state=42, stratify=y_short
    )
    
    train_data_short = lgb.Dataset(X_train_short, label=y_train_short)
    val_data_short = lgb.Dataset(X_test_short, label=y_test_short, reference=train_data_short)
    
    short_model = lgb.train(
        long_params,
        train_data_short,
        valid_sets=[val_data_short],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(False)]
    )
    
    # Find optimal thresholds for quality
    long_pred = long_model.predict(X_test_long)
    short_pred = short_model.predict(X_test_short)
    
    best_long_threshold = find_quality_threshold(y_test_long, long_pred)
    best_short_threshold = find_quality_threshold(y_test_short, short_pred)
    
    print(f"Quality thresholds:")
    print(f"  Long: {best_long_threshold}")
    print(f"  Short: {best_short_threshold}")
    
    # Generate high-quality signals
    print("\n=== Generating High-Quality Signals ===")
    
    long_predictions = long_model.predict(X)
    short_predictions = short_model.predict(X)
    
    signals = []
    for i, (idx, long_pred, short_pred) in enumerate(zip(X.index, long_predictions, short_predictions)):
        # Only generate signals if confidence is high enough
        if long_pred > best_long_threshold and long_pred > short_pred + 0.05:
            signals.append({
                'timestamp': idx,
                'side': 'long',
                'confidence': long_pred,
                'sl_pips': 10.0,  # Tighter SL
                'tp_pips': 30.0   # 3:1 RR
            })
        elif short_pred > best_short_threshold and short_pred > long_pred + 0.05:
            signals.append({
                'timestamp': idx,
                'side': 'short',
                'confidence': short_pred,
                'sl_pips': 10.0,  # Tighter SL
                'tp_pips': 30.0   # 3:1 RR
            })
    
    signals_df = pd.DataFrame(signals)
    if len(signals_df) > 0:
        signals_df.set_index('timestamp', inplace=True)
        signals_df.to_csv(run_dir / "data" / "signals.csv")
        
        print(f"Generated {len(signals_df)} high-quality signals")
        print(f"Direction distribution: {signals_df['side'].value_counts().to_dict()}")
        print(f"Mean confidence: {signals_df['confidence'].mean():.3f}")
        
        # Save model info
        model_info = {
            'long_threshold': float(best_long_threshold),
            'short_threshold': float(best_short_threshold),
            'total_signals': len(signals_df),
            'mean_confidence': float(signals_df['confidence'].mean())
        }
        
        (run_dir / "artifacts").mkdir(exist_ok=True)
        with open(run_dir / "artifacts" / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✅ High-quality signals generated successfully!")
        return True
    else:
        print("❌ No signals generated!")
        return False


def find_quality_threshold(y_true, y_pred):
    """Find threshold that maximizes precision for quality."""
    best_precision = 0
    best_threshold = 0.5
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_binary = (y_pred > threshold).astype(int)
        if y_pred_binary.sum() > 0:
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            if precision > best_precision:
                best_precision = precision
                best_threshold = threshold
    
    return best_threshold


if __name__ == "__main__":
    success = train_with_quality_focus()
    sys.exit(0 if success else 1) 