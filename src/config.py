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
    entry_threshold: float = 0.002
    exit_threshold: float = 0.001
    stop_loss: float = 0.02
    take_profit: float = 0.03
    trailing_stop: float = 0.01
    max_position_size: float = 0.2
    
    # Risk management
    risk_per_trade: float = 0.02
    max_trades: int = 5
    min_volatility: float = 0.005
    max_drawdown: float = 0.15
    
    # RL Agent parameters
    rl_learning_rate: float = 0.0003
    rl_gamma: float = 0.95
    rl_epsilon_decay: float = 0.995
    rl_memory_size: int = 20000
    rl_batch_size: int = 256
    rl_update_target_freq: int = 50
    rl_min_epsilon: float = 0.10
    
    # Trading thresholds for RL
    rl_position_threshold: float = 0.5
    rl_min_trade_interval: int = 5
    rl_reward_scaling: float = 1.0
    
    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'lstm': 0.4,
                'rf': 0.4,
                'svr': 0.2
            } 