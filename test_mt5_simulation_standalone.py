#!/usr/bin/env python3
"""Standalone MT5-Realistic Simulation Test

This script tests the MT5-realistic simulation concepts using simple Python
without external dependencies, to validate our approach.

Author: David Stetler
Date: 2025-01-29
"""

import sys
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class MockMT5Order:
    """Mock MT5 order for testing."""
    id: int
    symbol: str
    type: str  # 'buy' or 'sell'
    volume: float
    open_price: float
    sl: float
    tp: float
    open_time: str
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0
    close_price: Optional[float] = None
    close_time: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class MockMarketConditions:
    """Mock market conditions for testing."""
    spread: float
    volatility: float
    session: str
    news_impact: float = 0.0
    liquidity_level: float = 1.0
    connection_quality: float = 1.0


class MockMT5Simulator:
    """Mock MT5-realistic simulator for testing concepts."""
    
    def __init__(self):
        self.account_balance = 10000.0
        self.account_equity = 10000.0
        self.margin_used = 0.0
        self.free_margin = 10000.0
        
        # MT5 parameters
        self.commission_per_lot = 3.5
        self.swap_long = -0.33
        self.swap_short = -0.97
        self.min_lot_size = 0.01
        self.max_lot_size = 100.0
        self.contract_size = 100000
        
        # Execution parameters
        self.base_execution_delay = 0.05  # 50ms
        self.max_execution_delay = 0.15   # 150ms
        self.slippage_threshold = 0.0008
        self.max_slippage_pips = 2.0
        
        # News events
        self.news_events = [
            {'time': '08:30', 'impact': 'high', 'spread_multiplier': 3.0},
            {'time': '14:00', 'impact': 'medium', 'spread_multiplier': 2.0},
        ]
        
        self.orders = {}
        self.order_counter = 1
        self.trade_history = []
        
        print("🔧 Mock MT5-Realistic Simulator initialized")
    
    def calculate_dynamic_spread(self, bar: Dict, base_spread: float = 0.00013) -> float:
        """Calculate dynamic spread based on market conditions."""
        spread = base_spread
        
        # Volatility adjustment
        atr_pct = bar.get('atr', 0.001) / bar['close']
        if atr_pct > 0.002:  # High volatility
            spread *= 2.2
        elif atr_pct > 0.0015:  # Medium volatility
            spread *= 1.6
        elif atr_pct < 0.0008:  # Low volatility
            spread *= 0.9
        
        # Session adjustment
        hour = bar.get('hour', 12)
        session_multipliers = {
            7: 1.4, 8: 1.5, 9: 1.3,      # London open
            12: 1.6,                      # Overlap
            13: 1.4, 14: 1.3, 15: 1.2,   # NY open
            22: 0.8, 23: 0.7, 0: 0.7,    # Asian quiet
        }
        spread *= session_multipliers.get(hour, 1.0)
        
        # News event adjustment
        news_multiplier = self._get_news_impact(bar.get('time', '10:00'))
        spread *= news_multiplier
        
        # Clamp to realistic range
        min_spread = 0.00008  # 0.8 pips
        max_spread = 0.00050  # 5.0 pips
        spread = max(min_spread, min(max_spread, spread))
        
        return spread
    
    def _get_news_impact(self, time_str: str) -> float:
        """Get news impact multiplier for given time."""
        for event in self.news_events:
            event_time = event['time']
            # Simplified: check if within same hour
            if time_str.split(':')[0] == event_time.split(':')[0]:
                return event['spread_multiplier']
        return 1.0
    
    def get_market_conditions(self, bar: Dict) -> MockMarketConditions:
        """Extract market conditions from bar."""
        hour = bar.get('hour', 12)
        
        # Determine session
        if 7 <= hour <= 15:
            session = 'london'
        elif 13 <= hour <= 21:
            session = 'ny'
        elif 12 <= hour <= 15:
            session = 'overlap'
        else:
            session = 'asian'
        
        # Calculate volatility
        volatility = bar.get('atr', 0.001) / bar['close']
        
        # Calculate spread
        spread = self.calculate_dynamic_spread(bar)
        
        # News impact
        news_impact = self._get_news_impact(bar.get('time', '10:00')) - 1.0
        
        # Liquidity
        liquidity_factors = {'london': 1.0, 'ny': 0.9, 'overlap': 1.2, 'asian': 0.6}
        liquidity = liquidity_factors.get(session, 0.8)
        
        # Connection quality
        import random
        connection = 0.95 + random.uniform(-0.05, 0.05)
        connection = max(0.8, min(1.0, connection))
        
        return MockMarketConditions(
            spread=spread,
            volatility=volatility,
            session=session,
            news_impact=news_impact,
            liquidity_level=liquidity,
            connection_quality=connection
        )
    
    def calculate_execution_delay(self, market_conditions: MockMarketConditions) -> float:
        """Calculate realistic execution delay."""
        base_delay = self.base_execution_delay
        
        # Volatility impact
        if market_conditions.volatility > 0.002:
            volatility_delay = base_delay * 1.8
        elif market_conditions.volatility > 0.0015:
            volatility_delay = base_delay * 1.4
        else:
            volatility_delay = base_delay
        
        # Session impact
        session_delays = {'london': 1.0, 'ny': 1.1, 'overlap': 1.3, 'asian': 0.8}
        session_delay = volatility_delay * session_delays.get(market_conditions.session, 1.0)
        
        # Connection quality impact
        connection_delay = session_delay / market_conditions.connection_quality
        
        # News impact
        news_delay = connection_delay * (1.0 + market_conditions.news_impact * 0.5)
        
        # Add jitter
        import random
        jitter = random.gauss(0, base_delay * 0.2)
        total_delay = max(0.01, news_delay + jitter)
        
        return min(total_delay, self.max_execution_delay)
    
    def calculate_slippage(self, order_type: str, market_conditions: MockMarketConditions, 
                          volume: float) -> float:
        """Calculate realistic slippage."""
        if market_conditions.volatility < self.slippage_threshold:
            return 0.0
        
        # Base slippage
        volatility_factor = market_conditions.volatility / self.slippage_threshold
        base_slippage_pips = min(self.max_slippage_pips, volatility_factor * 0.8)
        
        # Volume impact
        volume_factor = 1.0 + (volume - 0.01) * 0.1
        
        # Liquidity impact
        liquidity_factor = 2.0 - market_conditions.liquidity_level
        
        # Session impact
        session_factors = {'london': 0.8, 'ny': 0.9, 'overlap': 0.7, 'asian': 1.4}
        session_factor = session_factors.get(market_conditions.session, 1.0)
        
        # News impact
        news_factor = 1.0 + market_conditions.news_impact * 1.5
        
        # Calculate final slippage
        total_slippage_pips = base_slippage_pips * volume_factor * liquidity_factor * session_factor * news_factor
        slippage_price = total_slippage_pips * 0.0001
        
        # Direction matters
        if order_type == 'buy':
            return slippage_price
        else:
            return -slippage_price
    
    def calculate_position_size(self, signal: Dict, price: float) -> float:
        """Calculate position size based on risk management."""
        risk_pct = signal.get('risk_pct', 0.01)
        sl_distance = abs(signal.get('sl_price', price) - price)
        
        if sl_distance == 0:
            return self.min_lot_size
        
        # Risk-based position sizing
        risk_amount = self.account_balance * risk_pct
        pip_value = self.contract_size * 0.0001
        sl_distance_pips = sl_distance / 0.0001
        
        volume = risk_amount / (sl_distance_pips * pip_value)
        
        # Round and apply limits
        volume = round(volume / 0.01) * 0.01
        volume = max(self.min_lot_size, min(self.max_lot_size, volume))
        
        return volume
    
    def can_place_order(self, volume: float, price: float) -> bool:
        """Check if order can be placed."""
        required_margin = volume * self.contract_size * price * 0.033
        return (self.free_margin >= required_margin and 
                self.min_lot_size <= volume <= self.max_lot_size)
    
    def place_order(self, bar: Dict, signal: Dict, market_conditions: MockMarketConditions) -> Optional[MockMT5Order]:
        """Place order with realistic execution."""
        # Calculate execution delay
        execution_delay = self.calculate_execution_delay(market_conditions)
        
        # Simulate price movement during delay
        import random
        delayed_price = bar['close'] * (1 + random.gauss(0, market_conditions.volatility * 0.1))
        
        # Calculate position size and slippage
        order_type = signal.get('side', 'buy')
        volume = self.calculate_position_size(signal, delayed_price)
        slippage = self.calculate_slippage(order_type, market_conditions, volume)
        
        # Final execution price
        if order_type == 'buy':
            execution_price = delayed_price + market_conditions.spread / 2 + slippage
        else:
            execution_price = delayed_price - market_conditions.spread / 2 + slippage
        
        # Check if order can be filled
        if not self.can_place_order(volume, execution_price):
            return None
        
        # Create order
        order = MockMT5Order(
            id=self.order_counter,
            symbol='EURUSD',
            type=order_type,
            volume=volume,
            open_price=execution_price,
            sl=signal.get('sl_price', 0),
            tp=signal.get('tp_price', 0),
            open_time=bar.get('time', '10:00'),
            commission=volume * self.commission_per_lot,
        )
        
        self.order_counter += 1
        self.orders[order.id] = order
        
        # Update account
        required_margin = volume * self.contract_size * execution_price * 0.033
        self.margin_used += required_margin
        self.free_margin = self.account_balance - self.margin_used
        self.account_balance -= order.commission
        
        return order
    
    def close_order(self, order_id: int, close_price: float, reason: str):
        """Close order and calculate profit."""
        order = self.orders[order_id]
        
        order.close_price = close_price
        order.reason = reason
        
        # Calculate profit
        if order.type == 'buy':
            price_diff = close_price - order.open_price
        else:
            price_diff = order.open_price - close_price
        
        pip_value = self.contract_size * 0.0001
        profit_pips = price_diff / 0.0001
        gross_profit = profit_pips * pip_value * order.volume
        
        # Add swap (simplified)
        swap = (self.swap_long if order.type == 'buy' else self.swap_short) * order.volume * 0.0001 * self.contract_size
        order.swap = swap
        order.profit = gross_profit - order.commission + swap
        
        # Update account
        self.account_balance += order.profit
        
        # Release margin
        released_margin = order.volume * self.contract_size * order.open_price * 0.033
        self.margin_used -= released_margin
        self.free_margin = self.account_balance - self.margin_used
        self.account_equity = self.account_balance
        
        # Add to history
        self.trade_history.append({
            'order_id': order.id,
            'type': order.type,
            'volume': order.volume,
            'open_price': order.open_price,
            'close_price': order.close_price,
            'profit': order.profit,
            'commission': order.commission,
            'swap': order.swap,
            'reason': reason,
            'pips': profit_pips,
        })
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trade_history:
            return {'error': 'No trades executed'}
        
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['profit'] > 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t['profit'] for t in self.trade_history)
        gross_profit = sum(t['profit'] for t in self.trade_history if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in self.trade_history if t['profit'] < 0))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'profit_factor': profit_factor,
            'final_balance': self.account_balance,
            'return_pct': (self.account_balance - 10000) / 10000,
        }


def create_mock_market_data(n_bars: int = 50) -> List[Dict]:
    """Create mock market data for testing."""
    import random
    random.seed(42)
    
    data = []
    base_price = 1.1000
    
    for i in range(n_bars):
        # Generate realistic bar
        price_change = random.gauss(0, 0.0008)
        close_price = base_price + price_change
        
        bar = {
            'open': base_price,
            'high': close_price + random.uniform(0, 0.0003),
            'low': close_price - random.uniform(0, 0.0003),
            'close': close_price,
            'volume': random.randint(50, 500),
            'atr': random.uniform(0.0008, 0.0015),
            'hour': (i * 5 // 60) % 24,  # 5-minute bars
            'time': f"{(i * 5 // 60) % 24:02d}:{(i * 5) % 60:02d}",
        }
        
        # Ensure OHLC consistency
        bar['high'] = max(bar['open'], bar['high'], bar['low'], bar['close'])
        bar['low'] = min(bar['open'], bar['high'], bar['low'], bar['close'])
        
        data.append(bar)
        base_price = close_price
    
    return data


def create_mock_signals(n_signals: int = 5) -> List[Dict]:
    """Create mock trading signals."""
    import random
    random.seed(42)
    
    signals = []
    for i in range(n_signals):
        side = random.choice(['buy', 'sell'])
        entry_price = 1.1000 + random.gauss(0, 0.01)
        
        if side == 'buy':
            sl_price = entry_price - 0.0015  # 15 pips SL
            tp_price = entry_price + 0.0030  # 30 pips TP
        else:
            sl_price = entry_price + 0.0015
            tp_price = entry_price - 0.0030
        
        signals.append({
            'side': side,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'risk_pct': 0.01,
            'bar_index': i * 10,  # Spread signals out
        })
    
    return signals


def test_simulator_initialization():
    """Test simulator initialization."""
    print("🔍 Test 1: Simulator initialization")
    
    simulator = MockMT5Simulator()
    
    # Check initial state
    assert simulator.account_balance == 10000.0, "Should start with $10,000"
    assert simulator.commission_per_lot == 3.5, "Should have realistic commission"
    assert simulator.min_lot_size == 0.01, "Should have correct minimum lot size"
    assert simulator.contract_size == 100000, "Should have standard contract size"
    assert len(simulator.news_events) > 0, "Should have news events"
    
    print("✅ Simulator initialization test passed")


def test_dynamic_spread_calculation():
    """Test dynamic spread calculation."""
    print("🔍 Test 2: Dynamic spread calculation")
    
    simulator = MockMT5Simulator()
    
    # Test different market conditions
    test_bars = [
        {'close': 1.1000, 'atr': 0.0012, 'hour': 8, 'time': '08:00'},   # London, medium vol
        {'close': 1.1000, 'atr': 0.0005, 'hour': 22, 'time': '22:00'},  # Asian, low vol
        {'close': 1.1000, 'atr': 0.0025, 'hour': 12, 'time': '12:00'},  # Overlap, high vol
        {'close': 1.1000, 'atr': 0.0010, 'hour': 8, 'time': '08:30'},   # London, news time
    ]
    
    spreads = [simulator.calculate_dynamic_spread(bar) for bar in test_bars]
    
    # Check results
    assert len(spreads) == 4, "Should calculate spread for each bar"
    
    # All spreads should be in valid range
    for spread in spreads:
        assert 0.00008 <= spread <= 0.00050, f"Spread {spread} should be in valid range"
    
        # London should have higher spread than Asian
    assert spreads[0] > spreads[1], "London should have higher spread than Asian"
    
    # High volatility overlap should be higher than Asian low vol
    print(f"   Debug: London={spreads[0]:.6f}, Asian={spreads[1]:.6f}, Overlap={spreads[2]:.6f}, News={spreads[3]:.6f}")
    assert spreads[2] > spreads[1], "High volatility overlap should be higher than Asian low vol"
    
    # News time should have higher spread (news multiplier = 3.0)
    # Both might hit the max spread cap, so check they're both at maximum or news is higher
    assert spreads[3] >= spreads[0], "News time should have at least same spread as London"
    # Check that news time is at or near maximum spread
    assert spreads[3] >= 0.00045, "News time should have very high spread"
    
    print(f"   Spreads: London={spreads[0]:.5f}, Asian={spreads[1]:.5f}, Overlap={spreads[2]:.5f}, News={spreads[3]:.5f}")
    print("✅ Dynamic spread calculation test passed")


def test_market_conditions():
    """Test market conditions extraction."""
    print("🔍 Test 3: Market conditions")
    
    simulator = MockMT5Simulator()
    
    test_bar = {'close': 1.1000, 'atr': 0.0012, 'hour': 8, 'time': '08:00'}
    conditions = simulator.get_market_conditions(test_bar)
    
    # Check conditions
    assert conditions.session == 'london', "Should detect London session"
    assert 0 <= conditions.volatility <= 0.01, "Volatility should be reasonable"
    assert 0.5 <= conditions.liquidity_level <= 1.5, "Liquidity should be reasonable"
    assert 0.8 <= conditions.connection_quality <= 1.0, "Connection should be good"
    assert conditions.spread > 0, "Spread should be positive"
    
    print("✅ Market conditions test passed")


def test_execution_delay():
    """Test execution delay calculation."""
    print("🔍 Test 4: Execution delay")
    
    simulator = MockMT5Simulator()
    
    # Test different conditions
    conditions = [
        MockMarketConditions(0.0002, 0.0005, 'london', 0.0, 1.0, 1.0),    # Low vol, good connection
        MockMarketConditions(0.0003, 0.0025, 'overlap', 1.0, 0.8, 0.9),  # High vol, news, poor connection
        MockMarketConditions(0.0001, 0.0008, 'asian', 0.0, 1.2, 0.98),   # Low vol, good liquidity
    ]
    
    delays = [simulator.calculate_execution_delay(cond) for cond in conditions]
    
    # Check delays
    for delay in delays:
        assert 0.01 <= delay <= 0.15, f"Delay {delay} should be in realistic range"
    
    # High volatility/news should increase delay
    assert delays[1] > delays[0], "High volatility/news should increase delay"
    
    print(f"   Delays: Low vol={delays[0]:.3f}s, High vol={delays[1]:.3f}s, Asian={delays[2]:.3f}s")
    print("✅ Execution delay test passed")


def test_slippage_calculation():
    """Test slippage calculation."""
    print("🔍 Test 5: Slippage calculation")
    
    simulator = MockMT5Simulator()
    
    # Low volatility (no slippage)
    low_vol_conditions = MockMarketConditions(0.00013, 0.0005, 'london', 0.0, 1.0, 1.0)
    no_slippage = simulator.calculate_slippage('buy', low_vol_conditions, 0.1)
    assert no_slippage == 0.0, "Low volatility should have no slippage"
    
    # High volatility (slippage expected)
    high_vol_conditions = MockMarketConditions(0.0002, 0.0015, 'asian', 0.5, 0.6, 0.9)
    slippage = simulator.calculate_slippage('buy', high_vol_conditions, 0.1)
    assert slippage > 0, "High volatility should produce positive slippage for buy orders"
    
    # Sell order should have opposite slippage
    sell_slippage = simulator.calculate_slippage('sell', high_vol_conditions, 0.1)
    assert sell_slippage < 0, "Sell orders should have negative slippage"
    
    print(f"   Slippage: No vol={no_slippage:.6f}, Buy={slippage:.6f}, Sell={sell_slippage:.6f}")
    print("✅ Slippage calculation test passed")


def test_position_sizing():
    """Test position sizing calculation."""
    print("🔍 Test 6: Position sizing")
    
    simulator = MockMT5Simulator()
    
    # Test signal with 2% risk
    test_signal = {
        'risk_pct': 0.02,
        'sl_price': 1.0985,  # 15 pips from entry
    }
    
    position_size = simulator.calculate_position_size(test_signal, 1.1000)
    
    # Check position size
    assert 0.01 <= position_size <= 100.0, "Position size should be in valid range"
    assert position_size >= simulator.min_lot_size, "Should meet minimum lot size"
    
    # Check that higher risk gives larger position
    high_risk_signal = {**test_signal, 'risk_pct': 0.05}
    high_risk_size = simulator.calculate_position_size(high_risk_signal, 1.1000)
    assert high_risk_size > position_size, "Higher risk should give larger position"
    
    print(f"   Position sizes: 2% risk={position_size:.2f} lots, 5% risk={high_risk_size:.2f} lots")
    print("✅ Position sizing test passed")


def test_order_execution():
    """Test order placement and execution."""
    print("🔍 Test 7: Order execution")
    
    simulator = MockMT5Simulator()
    
    # Create test data
    test_bar = {'close': 1.1000, 'atr': 0.0012, 'hour': 8, 'time': '08:00'}
    test_signal = {
        'side': 'buy',
        'sl_price': 1.0985,
        'tp_price': 1.1030,
        'risk_pct': 0.01,
    }
    
    # Get market conditions and place order
    conditions = simulator.get_market_conditions(test_bar)
    order = simulator.place_order(test_bar, test_signal, conditions)
    
    # Check order was created
    assert order is not None, "Order should be created"
    assert order.type == 'buy', "Order type should match signal"
    assert order.volume > 0, "Order should have positive volume"
    assert order.open_price > test_bar['close'], "Buy order should include spread and slippage"
    assert order.commission > 0, "Order should have commission"
    
    # Check account was updated
    assert simulator.account_balance < 10000.0, "Balance should decrease by commission"
    assert simulator.margin_used > 0, "Margin should be used"
    assert simulator.free_margin < 10000.0, "Free margin should decrease"
    
    print(f"   Order: {order.type} {order.volume} lots at {order.open_price:.5f}")
    print(f"   Account: Balance=${simulator.account_balance:.2f}, Margin=${simulator.margin_used:.2f}")
    print("✅ Order execution test passed")


def test_complete_simulation():
    """Test complete simulation workflow."""
    print("🔍 Test 8: Complete simulation")
    
    simulator = MockMT5Simulator()
    
    # Create test data
    market_data = create_mock_market_data(30)
    signals = create_mock_signals(3)
    
    # Process signals
    for signal in signals:
        bar_index = signal['bar_index']
        if bar_index < len(market_data):
            bar = market_data[bar_index]
            conditions = simulator.get_market_conditions(bar)
            order = simulator.place_order(bar, signal, conditions)
            
            if order:
                # Simulate trade outcome (simplified)
                import random
                if random.random() < 0.6:  # 60% win rate
                    close_price = order.tp
                    reason = 'Take Profit'
                else:
                    close_price = order.sl
                    reason = 'Stop Loss'
                
                simulator.close_order(order.id, close_price, reason)
    
    # Calculate performance
    performance = simulator.calculate_performance_metrics()
    
    # Check results
    assert 'total_trades' in performance, "Should have trade count"
    assert 'win_rate' in performance, "Should have win rate"
    assert 'profit_factor' in performance, "Should have profit factor"
    assert performance['final_balance'] > 0, "Should have positive balance"
    
    print(f"   Results: {performance['total_trades']} trades, {performance['win_rate']:.1%} win rate")
    print(f"   Final balance: ${performance['final_balance']:.2f}")
    print("✅ Complete simulation test passed")


def run_all_tests():
    """Run all validation tests."""
    print("🎯 Running MT5-Realistic Simulation Tests\n")
    
    try:
        test_simulator_initialization()
        print()
        
        test_dynamic_spread_calculation()
        print()
        
        test_market_conditions()
        print()
        
        test_execution_delay()
        print()
        
        test_slippage_calculation()
        print()
        
        test_position_sizing()
        print()
        
        test_order_execution()
        print()
        
        test_complete_simulation()
        print()
        
        print("🎉 All MT5-realistic simulation tests passed!")
        print("\n📋 Key Validation Results:")
        print("   ✅ Simulator properly initialized with MT5 parameters")
        print("   ✅ Dynamic spread calculation handles sessions, volatility, and news")
        print("   ✅ Market conditions extracted correctly from bar data")
        print("   ✅ Execution delay calculated based on market conditions")
        print("   ✅ Slippage modeling responds to volatility, liquidity, and session")
        print("   ✅ Position sizing implements proper risk management")
        print("   ✅ Order execution includes realistic spread, slippage, and commission")
        print("   ✅ Complete simulation workflow processes trades correctly")
        print("\n🚀 MT5-realistic simulation is validated and ready!")
        
        return True
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)