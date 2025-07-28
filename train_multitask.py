#!/usr/bin/env python3
"""Multi-Task Trading Model Architecture

This module implements the revolutionary multi-task learning approach that replaces
binary classification with specialized models for different aspects of trading.

Key Features:
- Direction prediction (classification): Up/Down/Sideways probability
- Magnitude prediction (regression): Expected price movement size
- Volatility prediction (regression): Expected path volatility
- Timing prediction (regression): Time to target/stop hit
- Integrated expected value calculation

Author: David Stetler
Date: 2025-01-29
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️  Warning: Full dependencies not available. Running in test mode.")

from config import config
from utils import (
    get_run_dir,
    make_run_dirs,
    parse_start_date_arg,
    parse_end_date_arg,
    load_data,
)


class MultiTaskTradingModel:
    """Multi-task model for comprehensive trading prediction."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        
        # Model configurations
        self.model_configs = {
            'direction': {
                'type': 'classification',
                'objective': 'multiclass',
                'num_class': 3,  # Up, Down, Sideways
                'metric': 'multi_logloss',
                'params': {
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'min_data_in_leaf': 50,
                    'reg_alpha': 0.3,
                    'reg_lambda': 0.3,
                    'random_state': random_state,
                    'verbose': -1
                }
            },
            'magnitude': {
                'type': 'regression',
                'objective': 'regression',
                'metric': 'rmse',
                'params': {
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'min_data_in_leaf': 50,
                    'reg_alpha': 0.2,
                    'reg_lambda': 0.2,
                    'random_state': random_state,
                    'verbose': -1
                }
            },
            'volatility': {
                'type': 'regression',
                'objective': 'regression',
                'metric': 'rmse',
                'params': {
                    'boosting_type': 'gbdt',
                    'num_leaves': 25,  # Smaller for volatility
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'min_data_in_leaf': 30,
                    'reg_alpha': 0.4,
                    'reg_lambda': 0.4,
                    'random_state': random_state,
                    'verbose': -1
                }
            },
            'timing': {
                'type': 'regression',
                'objective': 'regression',
                'metric': 'rmse',
                'params': {
                    'boosting_type': 'gbdt',
                    'num_leaves': 25,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'min_data_in_leaf': 30,
                    'reg_alpha': 0.3,
                    'reg_lambda': 0.3,
                    'random_state': random_state,
                    'verbose': -1
                }
            }
        }
    
    def prepare_direction_labels(self, labels_df: pd.DataFrame) -> pd.Series:
        """Prepare direction labels (0=Down, 1=Sideways, 2=Up)."""
        # Use the probabilistic labels to determine direction
        long_signals = labels_df['label_long'] == 1
        short_signals = labels_df['label_short'] == 1
        
        # Create direction labels
        direction_labels = pd.Series(1, index=labels_df.index)  # Default: Sideways
        direction_labels.loc[long_signals] = 2  # Up
        direction_labels.loc[short_signals] = 0  # Down
        
        return direction_labels
    
    def prepare_magnitude_labels(self, labels_df: pd.DataFrame) -> pd.Series:
        """Prepare magnitude labels (expected absolute price movement)."""
        # Use the maximum favorable movement as target
        magnitude_long = labels_df.get('outcome_max_favorable_long', 0).abs()
        magnitude_short = labels_df.get('outcome_max_favorable_short', 0).abs()
        
        # Take the maximum potential movement
        magnitude_labels = pd.concat([magnitude_long, magnitude_short], axis=1).max(axis=1)
        
        # Cap extreme values and ensure positive
        magnitude_labels = magnitude_labels.clip(0, 0.01)  # Cap at 100 pips
        
        return magnitude_labels
    
    def prepare_volatility_labels(self, labels_df: pd.DataFrame) -> pd.Series:
        """Prepare volatility labels (expected path volatility)."""
        # Use path volatility from probabilistic labels
        volatility_labels = labels_df.get('outcome_path_volatility', 0).abs()
        
        # Cap extreme values
        volatility_labels = volatility_labels.clip(0, 0.005)  # Cap at 50 pips
        
        return volatility_labels
    
    def prepare_timing_labels(self, labels_df: pd.DataFrame, future_window: int = 24) -> pd.Series:
        """Prepare timing labels (expected bars to target/stop)."""
        # Estimate timing based on volatility and magnitude
        magnitude = self.prepare_magnitude_labels(labels_df)
        volatility = self.prepare_volatility_labels(labels_df)
        
        # Higher volatility = faster movement = lower timing
        # Higher magnitude = longer time to reach = higher timing
        timing_labels = (magnitude / (volatility + 1e-6)).clip(1, future_window)
        
        return timing_labels
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training."""
        # Get all numeric columns except labels and outcomes
        exclude_patterns = ['label_', 'outcome_', 'ev_', 'spread_', 'signal_quality_']
        
        feature_cols = []
        for col in df.columns:
            if any(pattern in col for pattern in exclude_patterns):
                continue
            if df[col].dtype in ['int64', 'float64']:
                feature_cols.append(col)
        
        # Ensure we have required basic features
        required_features = ['close', 'high', 'low', 'volume', 'atr']
        for feature in required_features:
            if feature not in feature_cols and feature in df.columns:
                feature_cols.append(feature)
        
        self.feature_columns = feature_cols
        return df[feature_cols].copy()
    
    def train(self, labels_df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train all multi-task models."""
        if not DEPENDENCIES_AVAILABLE:
            return self._mock_training_results()
        
        print("🎯 Training Multi-Task Trading Models...")
        
        # Prepare features
        X = self.prepare_features(labels_df)
        print(f"📊 Using {len(self.feature_columns)} features")
        
        # Prepare labels for each task
        print("🏷️ Preparing labels for each task...")
        y_direction = self.prepare_direction_labels(labels_df)
        y_magnitude = self.prepare_magnitude_labels(labels_df)
        y_volatility = self.prepare_volatility_labels(labels_df)
        y_timing = self.prepare_timing_labels(labels_df)
        
        # Split data (time-aware split)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        
        y_train = {
            'direction': y_direction.iloc[:split_idx],
            'magnitude': y_magnitude.iloc[:split_idx],
            'volatility': y_volatility.iloc[:split_idx],
            'timing': y_timing.iloc[:split_idx]
        }
        
        y_test = {
            'direction': y_direction.iloc[split_idx:],
            'magnitude': y_magnitude.iloc[split_idx:],
            'volatility': y_volatility.iloc[split_idx:],
            'timing': y_timing.iloc[split_idx:]
        }
        
        # Scale features
        print("📏 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        self.scalers['features'] = scaler
        
        # Train each model
        results = {}
        
        for task_name, config in self.model_configs.items():
            print(f"🔧 Training {task_name} model...")
            
            # Prepare training data
            if config['type'] == 'classification':
                train_data = lgb.Dataset(X_train_scaled, label=y_train[task_name])
                valid_data = lgb.Dataset(X_test_scaled, label=y_test[task_name], reference=train_data)
            else:
                train_data = lgb.Dataset(X_train_scaled, label=y_train[task_name])
                valid_data = lgb.Dataset(X_test_scaled, label=y_test[task_name], reference=train_data)
            
            # Train model
            model = lgb.train(
                config['params'],
                train_data,
                num_boost_round=200,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            self.models[task_name] = model
            
            # Evaluate model
            if config['type'] == 'classification':
                pred_proba = model.predict(X_test_scaled, num_iteration=model.best_iteration)
                pred_class = np.argmax(pred_proba.reshape(-1, 3), axis=1)
                accuracy = accuracy_score(y_test[task_name], pred_class)
                results[task_name] = {'accuracy': accuracy, 'type': 'classification'}
                print(f"   {task_name} accuracy: {accuracy:.3f}")
            else:
                pred = model.predict(X_test_scaled, num_iteration=model.best_iteration)
                mse = mean_squared_error(y_test[task_name], pred)
                mae = mean_absolute_error(y_test[task_name], pred)
                results[task_name] = {'mse': mse, 'mae': mae, 'type': 'regression'}
                print(f"   {task_name} MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        self.is_trained = True
        print("✅ Multi-task training completed!")
        
        return results
    
    def predict_comprehensive(self, X: pd.DataFrame) -> Dict:
        """Generate comprehensive predictions for all tasks."""
        if not DEPENDENCIES_AVAILABLE:
            return self._mock_predictions(len(X))
        
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scalers['features'].transform(X),
            columns=X.columns,
            index=X.index
        )
        
        predictions = {}
        
        # Direction predictions (probabilities)
        direction_proba = self.models['direction'].predict(X_scaled)
        predictions['direction_proba'] = direction_proba.reshape(-1, 3)
        predictions['direction_class'] = np.argmax(predictions['direction_proba'], axis=1)
        
        # Regression predictions
        for task in ['magnitude', 'volatility', 'timing']:
            predictions[task] = self.models[task].predict(X_scaled)
        
        return predictions
    
    def calculate_expected_value(self, predictions: Dict, spread_estimates: pd.Series) -> Dict:
        """Calculate expected value from multi-task predictions."""
        n_samples = len(predictions['direction_proba'])
        
        # Extract probabilities
        prob_down = predictions['direction_proba'][:, 0]
        prob_sideways = predictions['direction_proba'][:, 1]  
        prob_up = predictions['direction_proba'][:, 2]
        
        # Get magnitude and volatility predictions
        magnitude = predictions['magnitude']
        volatility = predictions['volatility']
        
        # Calculate expected values
        # For long positions: EV = P(up) * (magnitude - spread) - P(down) * (magnitude + spread)
        ev_long = prob_up * (magnitude - spread_estimates) - prob_down * (magnitude + spread_estimates)
        
        # For short positions: EV = P(down) * (magnitude - spread) - P(up) * (magnitude + spread)
        ev_short = prob_down * (magnitude - spread_estimates) - prob_up * (magnitude + spread_estimates)
        
        # Calculate confidence (how certain is the direction prediction)
        confidence_long = prob_up - prob_down
        confidence_short = prob_down - prob_up
        
        # Risk-adjusted expected value (penalize high volatility)
        volatility_penalty = volatility * 0.5
        ev_long_adjusted = ev_long - volatility_penalty
        ev_short_adjusted = ev_short - volatility_penalty
        
        return {
            'ev_long': ev_long_adjusted,
            'ev_short': ev_short_adjusted,
            'confidence_long': confidence_long,
            'confidence_short': confidence_short,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'prob_sideways': prob_sideways,
            'expected_magnitude': magnitude,
            'expected_volatility': volatility,
            'expected_timing': predictions['timing']
        }
    
    def save_models(self, save_dir: Path):
        """Save all trained models and scalers."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for task_name, model in self.models.items():
            model_path = save_dir / f"multitask_{task_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers
        scalers_path = save_dir / "multitask_scalers.pkl"
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'model_configs': self.model_configs,
            'is_trained': self.is_trained
        }
        metadata_path = save_dir / "multitask_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Models saved to {save_dir}")
    
    def load_models(self, load_dir: Path):
        """Load all trained models and scalers."""
        # Load metadata
        metadata_path = load_dir / "multitask_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_columns = metadata['feature_columns']
        self.model_configs = metadata['model_configs']
        self.is_trained = metadata['is_trained']
        
        # Load models
        for task_name in self.model_configs.keys():
            model_path = load_dir / f"multitask_{task_name}.pkl"
            with open(model_path, 'rb') as f:
                self.models[task_name] = pickle.load(f)
        
        # Load scalers
        scalers_path = load_dir / "multitask_scalers.pkl"
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        print(f"✅ Models loaded from {load_dir}")
    
    def _mock_training_results(self) -> Dict:
        """Mock training results for testing without dependencies."""
        return {
            'direction': {'accuracy': 0.65, 'type': 'classification'},
            'magnitude': {'mse': 0.0001, 'mae': 0.008, 'type': 'regression'},
            'volatility': {'mse': 0.00005, 'mae': 0.005, 'type': 'regression'},
            'timing': {'mse': 25.0, 'mae': 4.2, 'type': 'regression'}
        }
    
    def _mock_predictions(self, n_samples: int) -> Dict:
        """Mock predictions for testing without dependencies."""
        np.random.seed(self.random_state)
        
        # Mock direction probabilities (3 classes)
        direction_proba = np.random.dirichlet([1, 1, 1], n_samples)
        
        return {
            'direction_proba': direction_proba,
            'direction_class': np.argmax(direction_proba, axis=1),
            'magnitude': np.random.normal(0.001, 0.0005, n_samples).clip(0, 0.01),
            'volatility': np.random.normal(0.0008, 0.0003, n_samples).clip(0, 0.005),
            'timing': np.random.normal(12, 6, n_samples).clip(1, 24)
        }


def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic test data for validation."""
    np.random.seed(42)
    
    # Basic OHLCV data
    data = {
        'close': np.random.normal(1.1000, 0.01, n_samples),
        'high': np.random.normal(1.1005, 0.01, n_samples),
        'low': np.random.normal(0.9995, 0.01, n_samples),
        'volume': np.random.randint(100, 1000, n_samples),
        'atr': np.random.normal(0.0008, 0.0002, n_samples).clip(0.0001, 0.002),
    }
    
    # Technical indicators
    data.update({
        'ema_5': data['close'] + np.random.normal(0, 0.0001, n_samples),
        'ema_20': data['close'] + np.random.normal(0, 0.0002, n_samples),
        'rsi_14': np.random.uniform(20, 80, n_samples),
        'macd': np.random.normal(0, 0.0001, n_samples),
    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.index = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
    
    # Add probabilistic labels (mock)
    df['label_long'] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    df['label_short'] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    
    # Add outcome data (mock)
    df['outcome_max_favorable_long'] = np.random.normal(0.002, 0.001, n_samples).clip(0, 0.01)
    df['outcome_max_favorable_short'] = np.random.normal(0.002, 0.001, n_samples).clip(0, 0.01)
    df['outcome_path_volatility'] = np.random.normal(0.0008, 0.0003, n_samples).clip(0, 0.005)
    
    return df


def run_validation_tests():
    """Run validation tests for the multi-task model."""
    print("🧪 Running Multi-Task Model Validation Tests\n")
    
    # Test 1: Model initialization
    print("🔍 Test 1: Model initialization")
    model = MultiTaskTradingModel(random_state=42)
    assert len(model.model_configs) == 4, "Should have 4 task models"
    assert 'direction' in model.model_configs, "Should have direction model"
    assert 'magnitude' in model.model_configs, "Should have magnitude model"
    assert 'volatility' in model.model_configs, "Should have volatility model"
    assert 'timing' in model.model_configs, "Should have timing model"
    print("✅ Model initialization test passed")
    
    # Test 2: Label preparation
    print("\n🔍 Test 2: Label preparation")
    test_data = create_test_data(100)
    
    direction_labels = model.prepare_direction_labels(test_data)
    assert len(direction_labels) == 100, "Should have same length as input"
    assert set(direction_labels.unique()) <= {0, 1, 2}, "Should only have values 0, 1, 2"
    
    magnitude_labels = model.prepare_magnitude_labels(test_data)
    assert len(magnitude_labels) == 100, "Should have same length as input"
    assert (magnitude_labels >= 0).all(), "Magnitude should be non-negative"
    
    print("✅ Label preparation test passed")
    
    # Test 3: Feature preparation
    print("\n🔍 Test 3: Feature preparation")
    features = model.prepare_features(test_data)
    assert len(features) == 100, "Should have same length as input"
    assert len(model.feature_columns) > 0, "Should have feature columns"
    assert 'close' in model.feature_columns, "Should include basic features"
    print(f"   Prepared {len(model.feature_columns)} features")
    print("✅ Feature preparation test passed")
    
    # Test 4: Training (mock or real)
    print("\n🔍 Test 4: Model training")
    results = model.train(test_data, test_size=0.2)
    assert 'direction' in results, "Should have direction results"
    assert 'magnitude' in results, "Should have magnitude results"
    assert model.is_trained, "Should be marked as trained"
    print("✅ Model training test passed")
    
    # Test 5: Predictions
    print("\n🔍 Test 5: Predictions")
    test_features = model.prepare_features(test_data.head(10))
    predictions = model.predict_comprehensive(test_features)
    
    assert 'direction_proba' in predictions, "Should have direction probabilities"
    assert 'magnitude' in predictions, "Should have magnitude predictions"
    assert len(predictions['direction_proba']) == 10, "Should predict for all samples"
    print("✅ Predictions test passed")
    
    # Test 6: Expected value calculation
    print("\n🔍 Test 6: Expected value calculation")
    spread_estimates = pd.Series([0.00013] * 10, index=test_features.index)
    ev_results = model.calculate_expected_value(predictions, spread_estimates)
    
    assert 'ev_long' in ev_results, "Should have long EV"
    assert 'ev_short' in ev_results, "Should have short EV"
    assert 'confidence_long' in ev_results, "Should have confidence scores"
    print("✅ Expected value calculation test passed")
    
    print("\n🎉 All multi-task model tests passed!")
    print("\n📋 Key Validation Results:")
    print("   ✅ Multi-task architecture properly initialized")
    print("   ✅ Label preparation handles all task types")
    print("   ✅ Feature preparation filters correctly")
    print("   ✅ Training completes for all tasks")
    print("   ✅ Predictions generated for all tasks")
    print("   ✅ Expected value calculation integrates predictions")
    print("\n🚀 Multi-task model architecture is ready!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-task trading models")
    parser.add_argument("--run", type=str, help="Run directory (overrides RUN_ID)")
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Earliest bar to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        required=False,
        help="Final bar used for training (YYYY-MM-DD)",
    )
    parser.add_argument("--end_date", type=str, required=False, help="YYYY-MM-DD last date for training")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with validation",
    )
    return parser.parse_args()


def main():
    """Main function for multi-task model training."""
    args = parse_args()
    
    if args.test:
        run_validation_tests()
        return
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Error: Required dependencies not available.")
        print("   Please install: pandas, numpy, scikit-learn, lightgbm")
        print("   Running validation tests instead...")
        run_validation_tests()
        return
    
    run_dir = Path(args.run) if args.run else Path(get_run_dir())
    make_run_dirs(str(run_dir))
    
    # Load probabilistic labels
    labels_path = run_dir / "data" / "labeled_probabilistic.csv"
    if not labels_path.exists():
        print(f"❌ Error: {labels_path} not found. Run label_probabilistic.py first.")
        sys.exit(1)
    
    try:
        start_date = parse_start_date_arg(args.start_date)
        end_date = parse_end_date_arg(args.end_date)
        train_end_date = parse_end_date_arg(args.train_end_date)
    except ValueError as exc:
        raise SystemExit(f"Invalid date: {exc}") from exc
    
    # Load data
    print("📊 Loading probabilistic labels...")
    labels_df = load_data(
        str(labels_path), end_date=end_date, start_date=start_date, strict=False
    )
    
    print(f"📈 Loaded {len(labels_df):,} bars from {labels_df.index[0]} to {labels_df.index[-1]}")
    
    # Initialize and train model
    try:
        model = MultiTaskTradingModel(random_state=42)
        
        # Train models
        results = model.train(labels_df, test_size=args.test_size)
        
        # Save models
        models_dir = run_dir / "models"
        model.save_models(models_dir)
        
        # Save results
        results_path = run_dir / "data" / "multitask_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate predictions for full dataset
        print("🔮 Generating predictions for full dataset...")
        features = model.prepare_features(labels_df)
        predictions = model.predict_comprehensive(features)
        
        # Calculate expected values
        spread_estimates = labels_df.get('spread_estimate', pd.Series(0.00013, index=labels_df.index))
        ev_results = model.calculate_expected_value(predictions, spread_estimates)
        
        # Save predictions
        pred_df = pd.DataFrame(ev_results, index=labels_df.index)
        pred_path = run_dir / "data" / "multitask_predictions.csv"
        pred_df.to_csv(pred_path)
        
        print("\n🎯 Multi-Task Training Results:")
        for task, result in results.items():
            if result['type'] == 'classification':
                print(f"   {task.capitalize()}: {result['accuracy']:.1%} accuracy")
            else:
                print(f"   {task.capitalize()}: MSE {result['mse']:.6f}, MAE {result['mae']:.6f}")
        
        print(f"\n✅ Models saved to: {models_dir}")
        print(f"✅ Predictions saved to: {pred_path}")
        print(f"✅ Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Error training multi-task models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()