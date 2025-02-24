from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Data parameters
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    data_period: str = '2y'  # Amount of historical data to fetch
    
    # Time window parameters
    lookback_days: int = 60  # Number of past days to consider
    forecast_days: int = 7   # Number of days to forecast
    
    # Model parameters
    lstm_units: int = 128
    lstm_layers: int = 3
    dropout_rate: float = 0.5
    learning_rate: float = 0.0005
    weight_decay: float = 0.02
    gradient_clip: float = 0.3
    l1_lambda: float = 0.01
    
    # Training parameters
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    
    # Ensemble parameters
    ensemble_weights: dict = None
    rf_n_estimators: int = 200
    svr_kernel: str = 'rbf'
    
    # Feature parameters
    technical_indicators = [
        'rsi', 'macd', 'bollinger_bands',
        'sma_20', 'sma_50', 'ema_12', 'ema_26'
    ]
    
    # Visualization parameters
    fig_width: int = 1200
    fig_height: int = 800
    
    # Trading parameters
    entry_threshold: float = 0.005    # More selective entries
    exit_threshold: float = 0.001
    stop_loss: float = 0.025         # Tighter stop loss (2.5%)
    take_profit: float = 0.035       # Lower but realistic profit target
    trailing_stop: float = 0.015     # Tighter trailing stop (1.5%)
    max_position_size: float = 0.08   # Smaller position size (8%)
    
    # Risk management
    risk_per_trade: float = 0.01     # Lower risk per trade (1%)
    max_trades: int = 1              # Keep single trade limit
    min_volatility: float = 0.01     # Higher volatility requirement
    max_drawdown: float = 0.15
    
    # RL Agent parameters
    rl_learning_rate: float = 0.0003  # Even lower learning rate
    rl_gamma: float = 0.95
    rl_epsilon: float = 0.9
    rl_epsilon_min: float = 0.1
    rl_epsilon_decay: float = 0.998   # Slower exploration decay
    rl_batch_size: int = 64
    
    # Trading thresholds for RL
    rl_position_threshold: float = 0.5
    rl_min_trade_interval: int = 15   # Minimum bars between trades
    rl_reward_scaling: float = 1.0
    
    # New RL parameters
    rl_update_target_freq: int = 100    # How often to update target network
    rl_target_update_tau: float = 0.001 # Soft update parameter
    rl_update_freq: int = 10            # How often to update the network
    rl_warmup_episodes: int = 50        # Number of episodes before starting updates
    
    # Add new parameters
    trade_cooldown: int = 3          # Bars to wait before next trade
    min_hold_bars: int = 4
    max_hold_bars: int = 15
    reward_scaling: float = 75.0
    reward_risk_factor: float = 0.5
    reward_time_decay: float = 0.001
    
    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'lstm': 0.4,
                'rf': 0.4,
                'svr': 0.2
            } 