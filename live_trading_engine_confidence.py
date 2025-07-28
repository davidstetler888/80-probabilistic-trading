#!/usr/bin/env python3
"""Live Trading Engine with Confidence-Based Position Sizing

This engine implements confidence-based position sizing:
- 2% position size at 72% confidence (minimum)
- 5% position size at 100% confidence (maximum)
- Linear scaling between confidence levels

Author: David Stetler
Date: 2025-01-29
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import time
import schedule
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading_confidence.log'),
        logging.StreamHandler()
    ]
)

class ConfidenceBasedTradingEngine:
    """Live trading engine with confidence-based position sizing."""
    
    def __init__(self, config_path="mt5_config_confidence.py"):
        self.config = self.load_config(config_path)
        self.models = self.load_models()
        self.positions = {}
        self.performance = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'daily_risk_used': 0.0,
            'last_reset_date': datetime.now().date()
        }
        
        # Initialize MT5
        self.initialize_mt5()
        
        logging.info("🚀 Confidence-Based Trading Engine Initialized")
        logging.info(f"📊 Position Sizing: {self.config['min_position_percent']:.0%} - {self.config['max_position_percent']:.0%}")
        logging.info(f"🎯 Confidence Range: {self.config['min_confidence']:.0%} - {self.config['max_confidence']:.0%}")
        
    def load_config(self, config_path):
        """Load MT5 configuration."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.MT5_CONFIG
    
    def load_models(self):
        """Load all trained models."""
        models = {}
        models_dir = "trained_models"
        
        try:
            models['feature_eng'] = joblib.load(f"{models_dir}/feature_engineering.pkl")
            models['labeler'] = joblib.load(f"{models_dir}/probabilistic_labeler.pkl")
            models['multitask'] = joblib.load(f"{models_dir}/multitask_model.pkl")
            models['ensemble'] = joblib.load(f"{models_dir}/ensemble_system.pkl")
            
            logging.info("✅ All models loaded successfully")
            return models
            
        except Exception as e:
            logging.error(f"❌ Failed to load models: {e}")
            return None
    
    def initialize_mt5(self):
        """Initialize MT5 connection."""
        if not mt5.initialize():
            logging.error("❌ MT5 initialization failed")
            return False
        
        # Login to account
        login_result = mt5.login(
            login=self.config['login'],
            password=self.config['password'],
            server=self.config['server']
        )
        
        if not login_result:
            logging.error("❌ MT5 login failed")
            return False
        
        logging.info("✅ MT5 connection established")
        return True
    
    def calculate_confidence_based_position_size(self, signal, account_balance):
        """Calculate position size based on model confidence (2-5% range)."""
        
        confidence = signal.get('confidence', 0.72)  # Default to minimum confidence
        
        # Configuration values
        min_position = self.config['min_position_percent']  # 2%
        max_position = self.config['max_position_percent']  # 5%
        min_confidence = self.config['min_confidence']      # 72%
        max_confidence = self.config['max_confidence']      # 100%
        
        # Ensure confidence is within bounds
        confidence = max(min_confidence, min(max_confidence, confidence))
        
        # Linear scaling: confidence 72% = 2%, confidence 100% = 5%
        confidence_range = max_confidence - min_confidence
        position_range = max_position - min_position
        
        # Calculate position percentage
        confidence_normalized = (confidence - min_confidence) / confidence_range
        position_percent = min_position + (position_range * confidence_normalized)
        
        # Calculate dollar amount
        position_amount = account_balance * position_percent
        
        # Convert to lot size
        symbol_info = mt5.symbol_info(self.config['symbol'])
        if symbol_info is None:
            logging.error("❌ Failed to get symbol info")
            return 0.0
        
        current_price = mt5.symbol_info_tick(self.config['symbol']).ask
        lot_size = position_amount / (symbol_info.trade_contract_size * current_price)
        
        # Apply broker limits
        min_lot = symbol_info.volume_min
        max_lot = min(symbol_info.volume_max, 10.0)  # Cap at 10 lots for safety
        lot_step = symbol_info.volume_step
        
        # Round to lot step
        lot_size = max(min_lot, min(max_lot, lot_size))
        lot_size = round(lot_size / lot_step) * lot_step
        
        # Log position sizing calculation
        logging.info(f"📊 Position Sizing Calculation:")
        logging.info(f"   Confidence: {confidence:.1%}")
        logging.info(f"   Position %: {position_percent:.1%}")
        logging.info(f"   Position $: ${position_amount:.2f}")
        logging.info(f"   Lot Size: {lot_size}")
        
        return lot_size
    
    def check_risk_limits(self, new_position_amount):
        """Check if new position would exceed risk limits."""
        
        # Reset daily risk if new day
        current_date = datetime.now().date()
        if current_date != self.performance['last_reset_date']:
            self.performance['daily_risk_used'] = 0.0
            self.performance['last_reset_date'] = current_date
            logging.info("🔄 Daily risk counter reset")
        
        # Check daily risk limit
        account_info = mt5.account_info()
        if account_info is None:
            return False
        
        balance = account_info.balance
        daily_risk_percent = new_position_amount / balance
        total_daily_risk = self.performance['daily_risk_used'] + daily_risk_percent
        
        if total_daily_risk > self.config['max_daily_risk']:
            logging.warning(f"⚠️ Daily risk limit exceeded: {total_daily_risk:.1%} > {self.config['max_daily_risk']:.1%}")
            return False
        
        # Check drawdown limit
        equity = account_info.equity
        current_drawdown = (balance - equity) / balance
        
        if current_drawdown >= self.config['emergency_stop_drawdown']:
            logging.warning(f"⚠️ Emergency drawdown stop: {current_drawdown:.1%} >= {self.config['emergency_stop_drawdown']:.1%}")
            return False
        
        # Check maximum concurrent positions
        positions = mt5.positions_get(symbol=self.config['symbol'])
        if positions is not None and len(positions) >= self.config['max_concurrent_positions']:
            logging.warning(f"⚠️ Maximum positions reached: {len(positions)}")
            return False
        
        # Check free margin
        free_margin_percent = account_info.margin_free / account_info.equity
        if free_margin_percent < self.config['min_free_margin_percent']:
            logging.warning(f"⚠️ Insufficient free margin: {free_margin_percent:.1%}")
            return False
        
        return True
    
    def get_live_data(self, bars=100):
        """Get live market data."""
        rates = mt5.copy_rates_from_pos(
            self.config['symbol'],
            mt5.TIMEFRAME_M5,
            0,  # From current bar
            bars
        )
        
        if rates is None:
            logging.warning("⚠️ Failed to get live data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals using trained models."""
        
        try:
            # Step 1: Feature engineering
            df_features = self.models['feature_eng'].process_complete_dataset(df)
            
            # Step 2: Get latest features
            latest_features = df_features.tail(1)
            
            # Step 3: Generate ensemble prediction
            ensemble_pred = self.models['ensemble'].predict(latest_features)
            
            # Step 4: Apply signal filtering (Phase 1 optimized parameters)
            signals = {
                'timestamp': df.index[-1],
                'close_price': df['close'].iloc[-1],
                'long_signal': False,
                'short_signal': False,
                'expected_value': 0.0,
                'confidence': 0.0,
                'risk_reward': 0.0
            }
            
            # Extract predictions
            if len(ensemble_pred) > 0:
                pred = ensemble_pred[0]
                
                # Apply optimized thresholds (from Phase 1 calibration)
                min_expected_value = 0.0004  # 4 pips
                min_confidence = 0.72        # 72%
                min_risk_reward = 2.0        # 2:1
                
                if pred.get('expected_value_long', 0) > min_expected_value:
                    if pred.get('confidence_long', 0) >= min_confidence:
                        if pred.get('risk_reward_long', 0) >= min_risk_reward:
                            signals['long_signal'] = True
                            signals['expected_value'] = pred['expected_value_long']
                            signals['confidence'] = pred['confidence_long']
                            signals['risk_reward'] = pred['risk_reward_long']
                
                elif pred.get('expected_value_short', 0) > min_expected_value:
                    if pred.get('confidence_short', 0) >= min_confidence:
                        if pred.get('risk_reward_short', 0) >= min_risk_reward:
                            signals['short_signal'] = True
                            signals['expected_value'] = pred['expected_value_short']
                            signals['confidence'] = pred['confidence_short']
                            signals['risk_reward'] = pred['risk_reward_short']
            
            return signals
            
        except Exception as e:
            logging.error(f"❌ Signal generation failed: {e}")
            return None
    
    def execute_trade(self, signal):
        """Execute trade with confidence-based position sizing."""
        
        if not signal['long_signal'] and not signal['short_signal']:
            return False
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("❌ Failed to get account info")
            return False
        
        balance = account_info.balance
        
        # Calculate confidence-based position size
        lot_size = self.calculate_confidence_based_position_size(signal, balance)
        
        if lot_size <= 0:
            logging.warning("⚠️ Position size too small, skipping trade")
            return False
        
        # Calculate position amount for risk checking
        symbol_info = mt5.symbol_info(self.config['symbol'])
        current_price = signal['close_price']
        position_amount = lot_size * symbol_info.trade_contract_size * current_price
        
        # Check risk limits
        if not self.check_risk_limits(position_amount):
            logging.warning("⚠️ Risk limits exceeded, skipping trade")
            return False
        
        # Calculate SL and TP
        atr = self.calculate_atr()
        
        if signal['long_signal']:
            trade_type = mt5.ORDER_TYPE_BUY
            sl_price = current_price - (atr * 1.5)  # 1.5 ATR stop loss
            tp_price = current_price + (atr * signal['risk_reward'] * 1.5)  # RR-based TP
            direction = "LONG"
        else:
            trade_type = mt5.ORDER_TYPE_SELL
            sl_price = current_price + (atr * 1.5)
            tp_price = current_price - (atr * signal['risk_reward'] * 1.5)
            direction = "SHORT"
        
        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config['symbol'],
            "volume": lot_size,
            "type": trade_type,
            "price": current_price,
            "sl": sl_price,
            "tp": tp_price,
            "magic": self.config['magic_number'],
            "comment": f"FX-ML-v69 Conf:{signal['confidence']:.0%} EV:{signal['expected_value']:.4f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Execute trade
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"❌ Trade execution failed: {result.retcode}")
            return False
        
        # Update daily risk used
        position_risk_percent = (position_amount / balance)
        self.performance['daily_risk_used'] += position_risk_percent
        
        # Log successful trade
        logging.info(f"✅ {direction} trade executed:")
        logging.info(f"   Confidence: {signal['confidence']:.1%}")
        logging.info(f"   Position Size: {position_risk_percent:.1%} of account")
        logging.info(f"   Volume: {lot_size} lots")
        logging.info(f"   Price: {current_price:.5f}")
        logging.info(f"   SL: {sl_price:.5f}")
        logging.info(f"   TP: {tp_price:.5f}")
        logging.info(f"   Expected Value: {signal['expected_value']:.4f}")
        logging.info(f"   Daily Risk Used: {self.performance['daily_risk_used']:.1%}")
        
        return True
    
    def calculate_atr(self, periods=14):
        """Calculate Average True Range."""
        df = self.get_live_data(periods + 5)
        if df is None:
            return 0.0015  # Default ATR for EURUSD
        
        # Simple ATR calculation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(periods).mean().iloc[-1]
        
        return atr if not np.isnan(atr) else 0.0015
    
    def update_performance(self):
        """Update performance metrics."""
        # Get recent trades
        history = mt5.history_deals_get(
            datetime.now() - timedelta(hours=24),  # Last 24 hours
            datetime.now()
        )
        
        if history is not None and len(history) > 0:
            df = pd.DataFrame(history, columns=history[0]._asdict().keys())
            
            # Update basic stats
            self.performance['trades'] = len(df)
            self.performance['wins'] = len(df[df['profit'] > 0])
            self.performance['losses'] = len(df[df['profit'] <= 0])
            self.performance['total_profit'] = df['profit'].sum()
            
            # Log performance every 10 trades
            if self.performance['trades'] % self.config.get('performance_log_frequency', 10) == 0:
                win_rate = self.performance['wins'] / self.performance['trades'] if self.performance['trades'] > 0 else 0
                logging.info(f"📊 Performance Update:")
                logging.info(f"   Trades: {self.performance['trades']}")
                logging.info(f"   Win Rate: {win_rate:.1%}")
                logging.info(f"   Total Profit: ${self.performance['total_profit']:.2f}")
                logging.info(f"   Daily Risk Used: {self.performance['daily_risk_used']:.1%}")
    
    def run_live_trading(self):
        """Main live trading loop."""
        
        logging.info("🚀 Starting Confidence-Based Live Trading Engine")
        logging.info("=" * 60)
        logging.info(f"📊 Position Sizing: {self.config['min_position_percent']:.0%} - {self.config['max_position_percent']:.0%} based on confidence")
        logging.info(f"🛡️ Daily Risk Limit: {self.config['max_daily_risk']:.0%}")
        logging.info(f"🚨 Emergency Stop: {self.config['emergency_stop_drawdown']:.0%} drawdown")
        
        while True:
            try:
                # Get live data
                df = self.get_live_data()
                if df is None:
                    time.sleep(60)  # Wait 1 minute and retry
                    continue
                
                # Generate signals
                signal = self.generate_signals(df)
                if signal is None:
                    time.sleep(60)
                    continue
                
                # Execute trades if signal is valid
                if signal['long_signal'] or signal['short_signal']:
                    self.execute_trade(signal)
                
                # Update performance
                self.update_performance()
                
                # Wait for next bar (5 minutes)
                time.sleep(300)
                
            except KeyboardInterrupt:
                logging.info("🛑 Live trading stopped by user")
                break
            except Exception as e:
                logging.error(f"❌ Live trading error: {e}")
                time.sleep(60)  # Wait and continue
        
        # Cleanup
        mt5.shutdown()
        logging.info("✅ Confidence-based trading engine shutdown complete")

if __name__ == "__main__":
    engine = ConfidenceBasedTradingEngine()
    engine.run_live_trading()