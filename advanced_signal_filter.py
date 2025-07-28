#!/usr/bin/env python3
"""Advanced Signal Filtering System

This system implements sophisticated signal filtering to achieve the critical
trade volume target of 25-50 trades per week (currently 121).

The approach uses multiple filtering layers:
1. Expected Value Thresholds (stricter)
2. Market Regime Filtering (only trade favorable regimes)
3. Session Quality Scoring (prefer high-quality sessions)
4. Signal Clustering (avoid redundant signals)
5. Dynamic Cooldown Periods (prevent overtrading)

Author: David Stetler
Date: 2025-01-29
"""

import sys
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("🔍 Advanced Signal Filtering System")
print("=" * 60)

@dataclass
class FilteringParameters:
    """Advanced signal filtering parameters."""
    # Primary EV Filtering
    min_expected_value: float = 0.0008      # Raise from 0.0003 to 0.0008 (8 pips)
    min_confidence: float = 0.80            # Raise from 0.7 to 0.8
    min_market_favorability: float = 0.80   # Raise from 0.7 to 0.8
    
    # Market Regime Filtering
    required_trend_strength: float = 0.7    # Only trade strong trends
    min_volatility_percentile: float = 0.3  # Avoid extreme low volatility
    max_volatility_percentile: float = 0.9  # Avoid extreme high volatility
    
    # Session Quality Filtering
    london_session_multiplier: float = 1.5  # Prefer London session
    ny_session_multiplier: float = 1.3      # Prefer NY session
    overlap_session_multiplier: float = 2.0 # Strongly prefer overlap
    asian_session_multiplier: float = 0.3   # Avoid Asian session
    
    # Signal Clustering Prevention
    min_signal_separation_hours: int = 2    # Minimum hours between signals
    max_signals_per_day: int = 3            # Maximum signals per day
    max_signals_per_session: int = 2        # Maximum signals per session
    
    # Dynamic Cooldown System
    cooldown_after_loss: int = 4            # Hours cooldown after loss
    cooldown_after_win: int = 1             # Hours cooldown after win
    max_consecutive_trades: int = 2         # Max consecutive trades
    
    # Risk-Based Filtering
    min_risk_reward_ratio: float = 2.5      # Higher RR requirement
    max_correlation_with_open: float = 0.4  # Avoid correlated positions


class AdvancedSignalFilter:
    """Advanced signal filtering system for trade volume control."""
    
    def __init__(self, params: FilteringParameters):
        self.params = params
        self.signal_history = []
        self.daily_signal_count = {}
        self.session_signal_count = {}
        self.last_signal_time = None
        self.consecutive_trade_count = 0
        self.last_trade_result = None
        
        print("🔧 Advanced Signal Filter initialized")
        print(f"📊 Filtering Parameters:")
        print(f"   Min EV: {params.min_expected_value:.4f} (8 pips)")
        print(f"   Min Confidence: {params.min_confidence:.0%}")
        print(f"   Min Favorability: {params.min_market_favorability:.0%}")
        print(f"   Signal Separation: {params.min_signal_separation_hours}h")
        print(f"   Max Signals/Day: {params.max_signals_per_day}")
        print(f"   Min RR: {params.min_risk_reward_ratio:.1f}:1")
    
    def calculate_session_quality_score(self, bar: Dict) -> float:
        """Calculate session quality score for the current bar."""
        hour = bar.get('hour', 12)
        base_score = 1.0
        
        # Apply session multipliers
        if 7 <= hour <= 15:  # London session
            base_score *= self.params.london_session_multiplier
        elif 13 <= hour <= 21:  # NY session
            base_score *= self.params.ny_session_multiplier
        elif 12 <= hour <= 15:  # Overlap (gets both London and NY)
            base_score *= self.params.overlap_session_multiplier
        elif hour >= 22 or hour <= 6:  # Asian session
            base_score *= self.params.asian_session_multiplier
        
        # Volatility quality adjustment
        atr_percentile = bar.get('atr_percentile', 0.5)
        if self.params.min_volatility_percentile <= atr_percentile <= self.params.max_volatility_percentile:
            base_score *= 1.2  # Good volatility range
        else:
            base_score *= 0.7  # Poor volatility range
        
        return base_score
    
    def check_signal_clustering(self, current_time: str) -> bool:
        """Check if signal is too close to previous signals."""
        if not self.last_signal_time:
            return True  # First signal is always allowed
        
        # Parse times (simplified - assume same day for demo)
        current_hour = int(current_time.split(' ')[1].split(':')[0])
        last_hour = int(self.last_signal_time.split(' ')[1].split(':')[0])
        
        hours_diff = abs(current_hour - last_hour)
        if hours_diff < self.params.min_signal_separation_hours:
            return False  # Too close to previous signal
        
        return True
    
    def check_daily_limits(self, current_time: str) -> bool:
        """Check daily signal limits."""
        date = current_time.split(' ')[0]  # Get date part
        daily_count = self.daily_signal_count.get(date, 0)
        
        return daily_count < self.params.max_signals_per_day
    
    def check_session_limits(self, bar: Dict) -> bool:
        """Check session signal limits."""
        hour = bar.get('hour', 12)
        
        # Determine session
        if 7 <= hour <= 15:
            session = 'london'
        elif 13 <= hour <= 21:
            session = 'ny'
        elif hour >= 22 or hour <= 6:
            session = 'asian'
        else:
            session = 'other'
        
        session_count = self.session_signal_count.get(session, 0)
        return session_count < self.params.max_signals_per_session
    
    def check_cooldown_period(self, current_time: str) -> bool:
        """Check if we're in a cooldown period."""
        if not self.last_signal_time or not self.last_trade_result:
            return True  # No previous trade
        
        # Calculate hours since last trade
        current_hour = int(current_time.split(' ')[1].split(':')[0])
        last_hour = int(self.last_signal_time.split(' ')[1].split(':')[0])
        hours_diff = abs(current_hour - last_hour)
        
        # Apply cooldown based on last result
        if self.last_trade_result == 'loss':
            required_cooldown = self.params.cooldown_after_loss
        else:
            required_cooldown = self.params.cooldown_after_win
        
        return hours_diff >= required_cooldown
    
    def check_consecutive_trade_limit(self) -> bool:
        """Check consecutive trade limits."""
        return self.consecutive_trade_count < self.params.max_consecutive_trades
    
    def apply_primary_filters(self, signal_data: Dict) -> Tuple[bool, str]:
        """Apply primary EV and confidence filters."""
        # Expected Value filter
        ev_long = signal_data.get('ev_long', 0)
        ev_short = signal_data.get('ev_short', 0)
        max_ev = max(ev_long, ev_short)
        
        if max_ev < self.params.min_expected_value:
            return False, f"EV too low: {max_ev:.4f} < {self.params.min_expected_value:.4f}"
        
        # Confidence filter
        confidence = signal_data.get('model_confidence', 0)
        if confidence < self.params.min_confidence:
            return False, f"Confidence too low: {confidence:.2f} < {self.params.min_confidence:.2f}"
        
        # Market favorability filter
        favorability_long = signal_data.get('market_favorability_long', 0)
        favorability_short = signal_data.get('market_favorability_short', 0)
        max_favorability = max(favorability_long, favorability_short)
        
        if max_favorability < self.params.min_market_favorability:
            return False, f"Favorability too low: {max_favorability:.2f} < {self.params.min_market_favorability:.2f}"
        
        return True, "Primary filters passed"
    
    def apply_market_regime_filters(self, signal_data: Dict) -> Tuple[bool, str]:
        """Apply market regime and trend strength filters."""
        # Trend strength filter
        trend_strength = signal_data.get('mtf_trend_alignment', 0.5)
        if trend_strength < self.params.required_trend_strength:
            return False, f"Trend too weak: {trend_strength:.2f} < {self.params.required_trend_strength:.2f}"
        
        # Volatility regime filter
        atr_percentile = signal_data.get('atr_percentile', 0.5)
        if not (self.params.min_volatility_percentile <= atr_percentile <= self.params.max_volatility_percentile):
            return False, f"Volatility out of range: {atr_percentile:.2f} not in [{self.params.min_volatility_percentile:.1f}, {self.params.max_volatility_percentile:.1f}]"
        
        return True, "Market regime filters passed"
    
    def apply_risk_filters(self, signal_data: Dict) -> Tuple[bool, str]:
        """Apply risk-based filters."""
        # Risk-reward filter
        rr_long = signal_data.get('rr_long', 0)
        rr_short = signal_data.get('rr_short', 0)
        max_rr = max(rr_long, rr_short)
        
        if max_rr < self.params.min_risk_reward_ratio:
            return False, f"Risk-reward too low: {max_rr:.2f} < {self.params.min_risk_reward_ratio:.1f}"
        
        # Correlation filter (simplified)
        correlation = signal_data.get('position_correlation', 0)
        if correlation > self.params.max_correlation_with_open:
            return False, f"Correlation too high: {correlation:.2f} > {self.params.max_correlation_with_open:.1f}"
        
        return True, "Risk filters passed"
    
    def apply_timing_filters(self, signal_data: Dict) -> Tuple[bool, str]:
        """Apply timing and clustering filters."""
        current_time = signal_data.get('timestamp', '2023-01-01 12:00:00')
        
        # Signal clustering check
        if not self.check_signal_clustering(current_time):
            return False, f"Too close to previous signal (min {self.params.min_signal_separation_hours}h separation)"
        
        # Daily limits check
        if not self.check_daily_limits(current_time):
            return False, f"Daily limit reached ({self.params.max_signals_per_day} signals/day)"
        
        # Session limits check
        if not self.check_session_limits(signal_data):
            return False, f"Session limit reached ({self.params.max_signals_per_session} signals/session)"
        
        # Cooldown period check
        if not self.check_cooldown_period(current_time):
            cooldown = self.params.cooldown_after_loss if self.last_trade_result == 'loss' else self.params.cooldown_after_win
            return False, f"In cooldown period ({cooldown}h after {self.last_trade_result})"
        
        # Consecutive trade limit check
        if not self.check_consecutive_trade_limit():
            return False, f"Consecutive trade limit reached ({self.params.max_consecutive_trades} trades)"
        
        return True, "Timing filters passed"
    
    def filter_signal(self, signal_data: Dict) -> Tuple[bool, str, float]:
        """Apply complete signal filtering pipeline."""
        # Calculate session quality score
        session_score = self.calculate_session_quality_score(signal_data)
        
        # Apply all filter layers
        filters = [
            self.apply_primary_filters,
            self.apply_market_regime_filters,
            self.apply_risk_filters,
            self.apply_timing_filters,
        ]
        
        for filter_func in filters:
            passed, reason = filter_func(signal_data)
            if not passed:
                return False, reason, 0.0
        
        # Calculate final signal quality score
        base_quality = max(signal_data.get('ev_long', 0), signal_data.get('ev_short', 0)) * 1000  # Convert to pips
        quality_score = base_quality * session_score * signal_data.get('model_confidence', 0.7)
        
        return True, "All filters passed", quality_score
    
    def update_signal_history(self, signal_data: Dict, accepted: bool, result: Optional[str] = None):
        """Update signal history and counters."""
        current_time = signal_data.get('timestamp', '2023-01-01 12:00:00')
        
        if accepted:
            # Update counters
            date = current_time.split(' ')[0]
            self.daily_signal_count[date] = self.daily_signal_count.get(date, 0) + 1
            
            hour = signal_data.get('hour', 12)
            if 7 <= hour <= 15:
                session = 'london'
            elif 13 <= hour <= 21:
                session = 'ny'
            elif hour >= 22 or hour <= 6:
                session = 'asian'
            else:
                session = 'other'
            
            self.session_signal_count[session] = self.session_signal_count.get(session, 0) + 1
            
            # Update tracking variables
            self.last_signal_time = current_time
            self.consecutive_trade_count += 1
            
            # Add to history
            self.signal_history.append({
                'timestamp': current_time,
                'accepted': True,
                'quality_score': signal_data.get('quality_score', 0),
                'result': result,
            })
        
        # Update trade result if provided
        if result:
            self.last_trade_result = result
            if result == 'loss':
                self.consecutive_trade_count = 0  # Reset on loss


def test_advanced_filtering():
    """Test the advanced signal filtering system."""
    print("\n🧪 Testing Advanced Signal Filtering System")
    print("=" * 50)
    
    # Initialize filter with aggressive parameters
    params = FilteringParameters(
        min_expected_value=0.0008,  # 8 pips minimum
        min_confidence=0.80,        # 80% confidence
        min_market_favorability=0.80,  # 80% favorability
        max_signals_per_day=3,      # Max 3 per day
        min_signal_separation_hours=2,  # 2h separation
    )
    
    signal_filter = AdvancedSignalFilter(params)
    
    # Test signals (simulating our current 121 signals/week)
    test_signals = []
    
    # Generate 121 test signals for one week
    import random
    random.seed(42)
    
    for i in range(121):  # Current signal volume
        # Create realistic signal data
        signal = {
            'timestamp': f"2023-01-{(i//17)+1:02d} {(i*2) % 24:02d}:00:00",  # Spread over week
            'hour': (i * 2) % 24,
            'ev_long': random.uniform(0.0002, 0.0015),  # 2-15 pips EV
            'ev_short': random.uniform(0.0002, 0.0015),
            'model_confidence': random.uniform(0.6, 0.95),
            'market_favorability_long': random.uniform(0.6, 0.9),
            'market_favorability_short': random.uniform(0.6, 0.9),
            'mtf_trend_alignment': random.uniform(0.4, 0.9),
            'atr_percentile': random.uniform(0.2, 0.8),
            'rr_long': random.uniform(1.5, 3.5),
            'rr_short': random.uniform(1.5, 3.5),
            'position_correlation': random.uniform(0, 0.6),
        }
        test_signals.append(signal)
    
    # Apply filtering
    accepted_signals = []
    rejected_signals = []
    
    for signal in test_signals:
        passed, reason, quality_score = signal_filter.filter_signal(signal)
        
        if passed:
            signal['quality_score'] = quality_score
            accepted_signals.append(signal)
            signal_filter.update_signal_history(signal, True)
        else:
            signal['rejection_reason'] = reason
            rejected_signals.append(signal)
    
    # Analyze results
    print(f"\n📊 Filtering Results:")
    print(f"   Original signals: {len(test_signals)} (121/week)")
    print(f"   Accepted signals: {len(accepted_signals)} ({len(accepted_signals)}/week)")
    print(f"   Rejected signals: {len(rejected_signals)} ({len(rejected_signals)}/week)")
    print(f"   Acceptance rate: {len(accepted_signals)/len(test_signals):.1%}")
    print(f"   Reduction factor: {len(test_signals)/len(accepted_signals):.1f}x")
    
    # Check if we hit target range
    target_met = 25 <= len(accepted_signals) <= 50
    print(f"   Target range (25-50): {'✅ ACHIEVED' if target_met else '❌ MISSED'}")
    
    # Analyze rejection reasons
    rejection_reasons = {}
    for signal in rejected_signals:
        reason = signal['rejection_reason'].split(':')[0]  # Get main reason
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
    print(f"\n🔍 Top Rejection Reasons:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {reason}: {count} signals ({count/len(rejected_signals):.1%})")
    
    # Analyze accepted signal quality
    if accepted_signals:
        avg_quality = sum(s['quality_score'] for s in accepted_signals) / len(accepted_signals)
        avg_ev = sum(max(s['ev_long'], s['ev_short']) for s in accepted_signals) / len(accepted_signals)
        avg_confidence = sum(s['model_confidence'] for s in accepted_signals) / len(accepted_signals)
        
        print(f"\n⭐ Accepted Signal Quality:")
        print(f"   Average quality score: {avg_quality:.2f}")
        print(f"   Average EV: {avg_ev:.4f} ({avg_ev*10000:.1f} pips)")
        print(f"   Average confidence: {avg_confidence:.1%}")
    
    return target_met, len(accepted_signals)


def optimize_filtering_parameters():
    """Optimize filtering parameters to achieve target trade volume."""
    print("\n🔧 Optimizing Filtering Parameters")
    print("=" * 50)
    
    best_params = None
    best_trade_count = float('inf')
    best_score = -float('inf')
    
    # Parameter ranges to test
    ev_thresholds = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
    confidence_thresholds = [0.75, 0.80, 0.85]
    favorability_thresholds = [0.75, 0.80, 0.85]
    max_daily_signals = [2, 3, 4]
    
    print("Testing parameter combinations...")
    
    for ev_thresh in ev_thresholds:
        for conf_thresh in confidence_thresholds:
            for fav_thresh in favorability_thresholds:
                for max_daily in max_daily_signals:
                    # Create test parameters
                    test_params = FilteringParameters(
                        min_expected_value=ev_thresh,
                        min_confidence=conf_thresh,
                        min_market_favorability=fav_thresh,
                        max_signals_per_day=max_daily,
                    )
                    
                    # Quick simulation
                    import random
                    random.seed(42)  # Consistent results
                    
                    accepted_count = 0
                    total_quality = 0
                    
                    for i in range(121):  # Test on 121 signals
                        # Generate test signal
                        signal_ev = random.uniform(0.0002, 0.0015)
                        signal_conf = random.uniform(0.6, 0.95)
                        signal_fav = random.uniform(0.6, 0.9)
                        
                        # Apply filters
                        if (signal_ev >= ev_thresh and 
                            signal_conf >= conf_thresh and 
                            signal_fav >= fav_thresh and
                            accepted_count < max_daily * 7):  # Weekly limit
                            
                            accepted_count += 1
                            total_quality += signal_ev * signal_conf * signal_fav
                    
                    # Score this combination
                    if 25 <= accepted_count <= 50:  # In target range
                        # Prefer middle of range with high quality
                        distance_from_ideal = abs(accepted_count - 37.5)
                        avg_quality = total_quality / max(accepted_count, 1)
                        score = 100 - distance_from_ideal + avg_quality * 100
                        
                        if score > best_score:
                            best_score = score
                            best_trade_count = accepted_count
                            best_params = test_params
    
    if best_params:
        print(f"\n🎯 Optimal Parameters Found:")
        print(f"   EV Threshold: {best_params.min_expected_value:.4f}")
        print(f"   Confidence Threshold: {best_params.min_confidence:.2f}")
        print(f"   Favorability Threshold: {best_params.min_market_favorability:.2f}")
        print(f"   Max Daily Signals: {best_params.max_signals_per_day}")
        print(f"   Estimated Trade Volume: {best_trade_count:.0f}/week")
        print(f"   Optimization Score: {best_score:.1f}")
        
        return best_params
    else:
        print("❌ No optimal parameters found in tested ranges")
        return FilteringParameters()  # Return default


def main():
    """Main function for advanced signal filtering."""
    print("🔍 Advanced Signal Filtering System")
    print("Reducing trade volume from 121 to 25-50 trades/week\n")
    
    # Test current filtering
    print("Testing current aggressive filtering...")
    target_met, trade_count = test_advanced_filtering()
    
    if not target_met:
        print(f"\n🔧 Current filtering achieved {trade_count} trades/week")
        print("Optimizing parameters for better results...")
        
        # Optimize parameters
        optimal_params = optimize_filtering_parameters()
        
        # Test optimal parameters
        print(f"\nTesting optimal parameters...")
        params = optimal_params
        signal_filter = AdvancedSignalFilter(params)
        
        # Quick test with optimal parameters
        import random
        random.seed(42)
        
        accepted_count = 0
        for i in range(121):
            signal = {
                'timestamp': f"2023-01-{(i//17)+1:02d} {(i*2) % 24:02d}:00:00",
                'hour': (i * 2) % 24,
                'ev_long': random.uniform(0.0002, 0.0015),
                'ev_short': random.uniform(0.0002, 0.0015),
                'model_confidence': random.uniform(0.6, 0.95),
                'market_favorability_long': random.uniform(0.6, 0.9),
                'market_favorability_short': random.uniform(0.6, 0.9),
                'mtf_trend_alignment': random.uniform(0.4, 0.9),
                'atr_percentile': random.uniform(0.2, 0.8),
                'rr_long': random.uniform(1.5, 3.5),
                'rr_short': random.uniform(1.5, 3.5),
                'position_correlation': random.uniform(0, 0.6),
            }
            
            passed, reason, quality_score = signal_filter.filter_signal(signal)
            if passed:
                accepted_count += 1
                signal_filter.update_signal_history(signal, True)
        
        print(f"   Optimal filtering result: {accepted_count} trades/week")
        final_target_met = 25 <= accepted_count <= 50
        print(f"   Target achievement: {'✅ SUCCESS' if final_target_met else '❌ NEEDS MORE WORK'}")
        
        return final_target_met
    else:
        print(f"✅ Target achieved: {trade_count} trades/week in range [25-50]")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)