# MT5 Configuration with Confidence-Based Position Sizing
MT5_CONFIG = {
    # MT5 CONNECTION SETTINGS
    'login': YOUR_MT5_LOGIN,           # Replace with your MT5 account number
    'password': 'YOUR_PASSWORD',       # Replace with your MT5 password
    'server': 'YOUR_BROKER_SERVER',    # Replace with your broker's server name
    'symbol': 'EURUSD',                # Trading symbol
    'timeframe': 'M5',                 # 5-minute timeframe
    'magic_number': 20250129,          # Unique EA identifier
    
    # CONFIDENCE-BASED POSITION SIZING (2-5% range)
    'position_size_method': 'confidence_based',
    'min_position_percent': 0.02,      # 2% minimum position size
    'max_position_percent': 0.05,      # 5% maximum position size
    'min_confidence': 0.72,            # Minimum confidence from our system (72%)
    'max_confidence': 1.00,            # Maximum confidence (100%)
    
    # SAFETY LIMITS (Adjusted for higher position sizes)
    'max_daily_risk': 0.15,            # 15% daily risk limit (higher due to bigger positions)
    'max_weekly_risk': 0.25,           # 25% weekly risk limit
    'emergency_stop_drawdown': 0.12,   # 12% emergency stop (higher threshold)
    'max_concurrent_positions': 2,     # Maximum 2 positions (reduced due to bigger sizes)
    
    # RISK MANAGEMENT
    'cooldown_after_loss_hours': 2,    # 2 hours cooldown after losing trade
    'max_consecutive_losses': 3,       # Stop after 3 consecutive losses
    'correlation_limit': 0.4,          # 40% correlation limit between positions
    
    # TRADE EXECUTION
    'slippage_tolerance': 3,           # 3 pip slippage tolerance
    'max_spread': 0.0003,              # Maximum spread to trade (3 pips)
    'min_free_margin_percent': 0.2,    # 20% minimum free margin
    
    # LOGGING AND MONITORING
    'log_level': 'INFO',               # Logging level
    'performance_log_frequency': 10,   # Log performance every 10 trades
    'daily_report_time': '18:00',      # Daily report time (6 PM)
}