#!/usr/bin/env python3
"""Test Confidence-Based Position Sizing

This script demonstrates how position sizing scales with confidence levels.
Run this to see exactly how your position sizes will be calculated.

Author: David Stetler
Date: 2025-01-29
"""

def test_confidence_sizing():
    """Test confidence-based position sizing with different scenarios."""
    
    print("🧪 Testing Confidence-Based Position Sizing")
    print("=" * 50)
    
    # Configuration (same as in mt5_config_confidence.py)
    min_position_percent = 0.02  # 2%
    max_position_percent = 0.05  # 5%
    min_confidence = 0.72        # 72%
    max_confidence = 1.00        # 100%
    
    # Test account balance
    account_balance = 10000  # $10,000 account
    
    # Test different confidence levels
    test_confidences = [0.72, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    
    print(f"📊 Account Balance: ${account_balance:,}")
    print(f"📈 Position Range: {min_position_percent:.0%} - {max_position_percent:.0%}")
    print(f"🎯 Confidence Range: {min_confidence:.0%} - {max_confidence:.0%}")
    print()
    
    print("Confidence Level → Position Size → Dollar Amount")
    print("-" * 50)
    
    confidence_range = max_confidence - min_confidence
    position_range = max_position_percent - min_position_percent
    
    for confidence in test_confidences:
        # Calculate position percentage (same formula as in the engine)
        confidence_normalized = (confidence - min_confidence) / confidence_range
        position_percent = min_position_percent + (position_range * confidence_normalized)
        
        # Calculate dollar amount
        position_amount = account_balance * position_percent
        
        print(f"   {confidence:.0%}           →    {position_percent:.1%}     →   ${position_amount:,.0f}")
    
    print()
    print("🎯 Key Points:")
    print(f"   • Minimum trade: {min_position_percent:.0%} (${account_balance * min_position_percent:,.0f}) at {min_confidence:.0%} confidence")
    print(f"   • Maximum trade: {max_position_percent:.0%} (${account_balance * max_position_percent:,.0f}) at {max_confidence:.0%} confidence")
    print(f"   • Linear scaling between confidence levels")
    print(f"   • Daily limit: 15% (${account_balance * 0.15:,.0f}) maximum per day")
    
    print()
    print("💡 Example Trading Day:")
    example_trades = [
        {"confidence": 0.78, "direction": "LONG"},
        {"confidence": 0.85, "direction": "SHORT"},
        {"confidence": 0.92, "direction": "LONG"}
    ]
    
    total_daily_risk = 0
    
    for i, trade in enumerate(example_trades, 1):
        confidence = trade["confidence"]
        direction = trade["direction"]
        
        # Calculate position
        confidence_normalized = (confidence - min_confidence) / confidence_range
        position_percent = min_position_percent + (position_range * confidence_normalized)
        position_amount = account_balance * position_percent
        
        total_daily_risk += position_percent
        
        print(f"   Trade {i}: {direction} at {confidence:.0%} confidence → {position_percent:.1%} (${position_amount:,.0f})")
    
    print(f"   Total Daily Risk: {total_daily_risk:.1%} (Limit: 15%)")
    
    if total_daily_risk <= 0.15:
        print("   ✅ Within daily risk limit")
    else:
        print("   ⚠️ Exceeds daily risk limit - trading would stop")
    
    print()
    print("🚀 Ready to use confidence-based position sizing!")

if __name__ == "__main__":
    test_confidence_sizing()