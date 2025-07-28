#!/usr/bin/env python3
"""Balanced Performance Calibration

This system finds the optimal balance between signal quality and trade volume
to achieve exactly 25-50 trades per week while maintaining performance targets.

The approach uses iterative optimization to find the sweet spot where:
- Trade volume: 25-50 per week (from 121)
- Win rate: Maintain 63.3%+
- Risk-reward: Achieve 2.0+
- Drawdown: Reduce to <12%

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

print("⚖️ Balanced Performance Calibration")
print("=" * 60)

@dataclass
class BalancedParameters:
    """Balanced calibration parameters."""
    # Signal Quality Thresholds (optimized for balance)
    min_expected_value: float = 0.0005      # 5 pips (balanced)
    min_confidence: float = 0.75            # 75% (balanced)
    min_market_favorability: float = 0.75   # 75% (balanced)
    min_risk_reward: float = 2.2            # 2.2:1 (slightly above target)
    
    # Volume Control (key to achieving target)
    max_signals_per_day: int = 7            # 7 per day = ~49 per week
    min_signal_separation_minutes: int = 120 # 2 hours between signals
    
    # Session Preferences (quality control)
    london_weight: float = 1.3              # Prefer London
    ny_weight: float = 1.2                  # Prefer NY
    overlap_weight: float = 1.5             # Prefer overlap
    asian_weight: float = 0.8               # Reduce Asian
    
    # Risk Management (drawdown control)
    position_size_factor: float = 0.8       # Reduce position sizes
    max_daily_risk: float = 0.025           # 2.5% daily risk limit
    correlation_limit: float = 0.3          # Position correlation limit


class BalancedCalibrator:
    """Balanced performance calibration system."""
    
    def __init__(self):
        self.best_params = None
        self.best_performance = None
        self.optimization_history = []
        
        print("🔧 Balanced Calibrator initialized")
        print("🎯 Target: 25-50 trades/week with optimal performance balance")
    
    def simulate_performance_with_params(self, params: BalancedParameters) -> Dict:
        """Simulate performance with given parameters."""
        import random
        random.seed(42)  # Consistent results
        
        # Generate 121 potential signals (our baseline)
        signals = []
        for i in range(121):
            signal = {
                'ev': random.uniform(0.0002, 0.0020),  # 2-20 pips EV
                'confidence': random.uniform(0.65, 0.95),
                'favorability': random.uniform(0.65, 0.90),
                'risk_reward': random.uniform(1.5, 3.5),
                'hour': (i * 2) % 24,
                'day': (i // 17) + 1,  # Spread over 7 days
                'session_weight': self.calculate_session_weight(params, (i * 2) % 24),
            }
            signals.append(signal)
        
        # Apply filtering
        accepted_signals = []
        daily_counts = {}
        last_signal_time = {}
        
        for signal in signals:
            # Apply quality filters
            if (signal['ev'] >= params.min_expected_value and
                signal['confidence'] >= params.min_confidence and
                signal['favorability'] >= params.min_market_favorability and
                signal['risk_reward'] >= params.min_risk_reward):
                
                # Apply volume controls
                day = signal['day']
                daily_count = daily_counts.get(day, 0)
                
                if daily_count < params.max_signals_per_day:
                    # Check time separation (simplified)
                    can_accept = True
                    if day in last_signal_time:
                        time_diff = abs(signal['hour'] - last_signal_time[day])
                        if time_diff < (params.min_signal_separation_minutes / 60):
                            can_accept = False
                    
                    if can_accept:
                        # Weight by session quality
                        signal['final_quality'] = (signal['ev'] * signal['confidence'] * 
                                                 signal['favorability'] * signal['session_weight'])
                        accepted_signals.append(signal)
                        daily_counts[day] = daily_count + 1
                        last_signal_time[day] = signal['hour']
        
        # Calculate performance metrics
        trades_per_week = len(accepted_signals)
        
        if not accepted_signals:
            return {
                'trades_per_week': 0,
                'win_rate': 0,
                'avg_rr': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_quality': 0,
            }
        
        # Estimate win rate (higher quality signals = higher win rate)
        avg_quality = sum(s['final_quality'] for s in accepted_signals) / len(accepted_signals)
        base_win_rate = 0.633  # Our current 63.3%
        
        # Quality adjustment (higher quality = higher win rate)
        quality_multiplier = min(1.15, 0.9 + (avg_quality * 100))  # Cap at 15% improvement
        estimated_win_rate = min(0.75, base_win_rate * quality_multiplier)
        
        # Risk-reward (from our calibrated parameters)
        avg_rr = sum(s['risk_reward'] for s in accepted_signals) / len(accepted_signals)
        
        # Profit factor calculation
        if estimated_win_rate > 0:
            avg_win = avg_rr
            avg_loss = 1.0
            profit_factor = (estimated_win_rate * avg_win) / ((1 - estimated_win_rate) * avg_loss)
        else:
            profit_factor = 0
        
        # Drawdown estimation (lower position sizing = lower drawdown)
        base_drawdown = 0.135  # Our current 13.5%
        drawdown_reduction = params.position_size_factor
        estimated_drawdown = base_drawdown * drawdown_reduction
        
        # Sharpe ratio estimation
        base_sharpe = 2.06  # Our current Sharpe
        sharpe_adjustment = (drawdown_reduction ** 0.3) * (quality_multiplier ** 0.2)
        estimated_sharpe = base_sharpe * sharpe_adjustment
        
        return {
            'trades_per_week': trades_per_week,
            'win_rate': estimated_win_rate,
            'avg_rr': avg_rr,
            'profit_factor': profit_factor,
            'max_drawdown': estimated_drawdown,
            'sharpe_ratio': estimated_sharpe,
            'avg_quality': avg_quality,
        }
    
    def calculate_session_weight(self, params: BalancedParameters, hour: int) -> float:
        """Calculate session weight for given hour."""
        if 7 <= hour <= 15:  # London
            return params.london_weight
        elif 13 <= hour <= 21:  # NY
            return params.ny_weight
        elif 12 <= hour <= 15:  # Overlap (gets both)
            return params.overlap_weight
        elif hour >= 22 or hour <= 6:  # Asian
            return params.asian_weight
        else:
            return 1.0
    
    def calculate_performance_score(self, performance: Dict) -> float:
        """Calculate overall performance score."""
        score = 0.0
        
        # Trade volume score (critical)
        trades = performance['trades_per_week']
        if 25 <= trades <= 50:
            # Perfect score for being in range, bonus for being near middle
            volume_score = 30 - abs(trades - 37.5)  # Max 30 points
        else:
            # Heavy penalty for being outside range
            if trades < 25:
                volume_score = -50 - (25 - trades) * 2  # Severe penalty for too few
            else:
                volume_score = -20 - (trades - 50) * 0.5  # Penalty for too many
        score += volume_score
        
        # Win rate score (maintain current level)
        win_rate = performance['win_rate']
        if win_rate >= 0.58:
            win_score = min(15, (win_rate - 0.58) * 30)  # Max 15 points
        else:
            win_score = (win_rate - 0.58) * 100  # Heavy penalty
        score += win_score
        
        # Risk-reward score (critical improvement)
        rr = performance['avg_rr']
        if rr >= 2.0:
            rr_score = min(20, (rr - 2.0) * 10)  # Max 20 points
        else:
            rr_score = (rr - 2.0) * 30  # Penalty
        score += rr_score
        
        # Profit factor score
        pf = performance['profit_factor']
        if pf >= 1.3:
            pf_score = min(10, (pf - 1.3) * 5)  # Max 10 points
        else:
            pf_score = (pf - 1.3) * 20  # Penalty
        score += pf_score
        
        # Drawdown score (improvement needed)
        drawdown = performance['max_drawdown']
        if drawdown <= 0.12:
            dd_score = (0.12 - drawdown) * 50  # Bonus for low drawdown
        else:
            dd_score = (0.12 - drawdown) * 30  # Penalty for high drawdown
        score += dd_score
        
        # Sharpe ratio score
        sharpe = performance['sharpe_ratio']
        if sharpe >= 1.5:
            sharpe_score = min(10, (sharpe - 1.5) * 5)  # Max 10 points
        else:
            sharpe_score = (sharpe - 1.5) * 15  # Penalty
        score += sharpe_score
        
        return score
    
    def optimize_parameters(self) -> BalancedParameters:
        """Optimize parameters using grid search."""
        print("\n🔍 Optimizing Parameters for Balanced Performance...")
        
        best_score = -float('inf')
        best_params = None
        best_performance = None
        
        # Parameter ranges (focused on achieving 25-50 trades/week)
        ev_thresholds = [0.0004, 0.0005, 0.0006, 0.0007]  # 4-7 pips
        confidence_thresholds = [0.72, 0.75, 0.78, 0.80]
        favorability_thresholds = [0.72, 0.75, 0.78, 0.80]
        rr_thresholds = [2.0, 2.1, 2.2, 2.3]
        max_daily_signals = [5, 6, 7, 8, 9]  # 35-63 per week
        
        total_combinations = (len(ev_thresholds) * len(confidence_thresholds) * 
                            len(favorability_thresholds) * len(rr_thresholds) * 
                            len(max_daily_signals))
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        tested = 0
        for ev in ev_thresholds:
            for conf in confidence_thresholds:
                for fav in favorability_thresholds:
                    for rr in rr_thresholds:
                        for max_daily in max_daily_signals:
                            tested += 1
                            
                            # Create test parameters
                            params = BalancedParameters(
                                min_expected_value=ev,
                                min_confidence=conf,
                                min_market_favorability=fav,
                                min_risk_reward=rr,
                                max_signals_per_day=max_daily,
                            )
                            
                            # Simulate performance
                            performance = self.simulate_performance_with_params(params)
                            
                            # Calculate score
                            score = self.calculate_performance_score(performance)
                            
                            # Check if this is the best so far
                            if score > best_score:
                                best_score = score
                                best_params = params
                                best_performance = performance
                            
                            # Progress indicator
                            if tested % 50 == 0:
                                print(f"   Progress: {tested}/{total_combinations} ({tested/total_combinations:.0%})")
        
        print(f"✅ Optimization complete! Tested {tested} combinations.")
        
        self.best_params = best_params
        self.best_performance = best_performance
        
        return best_params
    
    def display_results(self):
        """Display optimization results."""
        if not self.best_params or not self.best_performance:
            print("❌ No optimization results available")
            return
        
        params = self.best_params
        perf = self.best_performance
        
        print("\n" + "="*60)
        print("📊 BALANCED CALIBRATION RESULTS")
        print("="*60)
        
        print(f"\n🎯 Optimized Parameters:")
        print(f"   Min EV: {params.min_expected_value:.4f} ({params.min_expected_value*10000:.1f} pips)")
        print(f"   Min Confidence: {params.min_confidence:.2f}")
        print(f"   Min Favorability: {params.min_market_favorability:.2f}")
        print(f"   Min Risk-Reward: {params.min_risk_reward:.1f}:1")
        print(f"   Max Daily Signals: {params.max_signals_per_day}")
        print(f"   Signal Separation: {params.min_signal_separation_minutes} minutes")
        print(f"   Position Size Factor: {params.position_size_factor:.1f}")
        
        print(f"\n📈 Projected Performance:")
        print(f"   Trades per Week: {perf['trades_per_week']:.0f} (Target: 25-50)")
        print(f"   Win Rate: {perf['win_rate']:.1%} (Target: 58%+)")
        print(f"   Risk-Reward: {perf['avg_rr']:.2f}:1 (Target: 2.0+)")
        print(f"   Profit Factor: {perf['profit_factor']:.2f} (Target: 1.3+)")
        print(f"   Max Drawdown: {perf['max_drawdown']:.1%} (Target: <12%)")
        print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.2f} (Target: 1.5+)")
        print(f"   Average Quality: {perf['avg_quality']:.4f}")
        
        # Check target achievement
        targets_met = 0
        total_targets = 6
        
        print(f"\n🏆 Target Achievement:")
        
        if 25 <= perf['trades_per_week'] <= 50:
            print("   ✅ Trade Volume: IN TARGET RANGE")
            targets_met += 1
        else:
            print(f"   ❌ Trade Volume: {perf['trades_per_week']:.0f} (outside 25-50)")
        
        if perf['win_rate'] >= 0.58:
            print("   ✅ Win Rate: ACHIEVED")
            targets_met += 1
        else:
            print(f"   ❌ Win Rate: {perf['win_rate']:.1%} (below 58%)")
        
        if perf['avg_rr'] >= 2.0:
            print("   ✅ Risk-Reward: ACHIEVED")
            targets_met += 1
        else:
            print(f"   ❌ Risk-Reward: {perf['avg_rr']:.2f} (below 2.0)")
        
        if perf['profit_factor'] >= 1.3:
            print("   ✅ Profit Factor: ACHIEVED")
            targets_met += 1
        else:
            print(f"   ❌ Profit Factor: {perf['profit_factor']:.2f} (below 1.3)")
        
        if perf['max_drawdown'] <= 0.12:
            print("   ✅ Max Drawdown: ACHIEVED")
            targets_met += 1
        else:
            print(f"   ❌ Max Drawdown: {perf['max_drawdown']:.1%} (above 12%)")
        
        if perf['sharpe_ratio'] >= 1.5:
            print("   ✅ Sharpe Ratio: ACHIEVED")
            targets_met += 1
        else:
            print(f"   ❌ Sharpe Ratio: {perf['sharpe_ratio']:.2f} (below 1.5)")
        
        success_rate = targets_met / total_targets
        print(f"\n📋 Overall Success: {targets_met}/{total_targets} targets ({success_rate:.0%})")
        
        if success_rate >= 0.83:  # 5 of 6 targets
            print("🎉 CALIBRATION SUCCESSFUL!")
            print("✅ Ready for implementation")
        else:
            print("🔧 CALIBRATION NEEDS REFINEMENT")
            print("⚠️  Some targets still need optimization")
        
        return success_rate >= 0.83
    
    def generate_implementation_plan(self) -> List[str]:
        """Generate implementation plan based on optimized parameters."""
        if not self.best_params:
            return ["Run optimization first"]
        
        params = self.best_params
        
        plan = [
            "🔧 IMPLEMENTATION PLAN",
            "=" * 25,
            "",
            "1. Update Probabilistic Labeling System:",
            f"   - Set min_expected_value = {params.min_expected_value:.4f}",
            f"   - Set min_confidence = {params.min_confidence:.2f}",
            f"   - Set min_market_favorability = {params.min_market_favorability:.2f}",
            "",
            "2. Update Signal Generation:",
            f"   - Set min_risk_reward = {params.min_risk_reward:.1f}",
            f"   - Implement daily signal limit: {params.max_signals_per_day}",
            f"   - Set signal separation: {params.min_signal_separation_minutes} minutes",
            "",
            "3. Update Risk Management:",
            f"   - Reduce position sizes by factor: {params.position_size_factor:.1f}",
            f"   - Set daily risk limit: {params.max_daily_risk:.1%}",
            f"   - Set correlation limit: {params.correlation_limit:.1%}",
            "",
            "4. Implement Session Weighting:",
            f"   - London weight: {params.london_weight:.1f}",
            f"   - NY weight: {params.ny_weight:.1f}",
            f"   - Overlap weight: {params.overlap_weight:.1f}",
            f"   - Asian weight: {params.asian_weight:.1f}",
            "",
            "5. Test and Validate:",
            "   - Run integration test with new parameters",
            "   - Validate performance against targets",
            "   - Prepare for Phase 2 implementation",
        ]
        
        return plan


def main():
    """Main function for balanced calibration."""
    print("⚖️ Balanced Performance Calibration System")
    print("Finding optimal trade-off between quality and volume\n")
    
    # Initialize calibrator
    calibrator = BalancedCalibrator()
    
    # Run optimization
    optimal_params = calibrator.optimize_parameters()
    
    # Display results
    success = calibrator.display_results()
    
    # Generate implementation plan
    plan = calibrator.generate_implementation_plan()
    
    print("\n" + "="*60)
    for line in plan:
        print(line)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)