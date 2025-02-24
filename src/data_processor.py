import logging
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ta

logger = logging.getLogger(__name__)

class CryptoDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()
        
    def fetch_data(self, symbol: str, interval: str, period: str = '2y') -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            interval: Data interval ('1d', '1h', '15m')
            period: Amount of historical data ('1y', '2y', '5y', 'max')
        """
        try:
            # Convert symbol format from Binance to Yahoo Finance
            yahoo_symbol = symbol.replace('/', '-') + '=X'
            if symbol == 'BTC/USDT':
                yahoo_symbol = 'BTC-USD'  # Special case for Bitcoin
                
            logger.info(f"Fetching {yahoo_symbol} data for period: {period}")
            
            # Convert interval from Binance format to Yahoo Finance format
            interval_map = {'1d': '1d', '1h': '1h', '15m': '15m'}
            yf_interval = interval_map.get(interval, '1d')
            
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=yf_interval)
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            logger.info(f"Fetched {len(df)} rows of data")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical indicators"""
        try:
            logger.info("Adding technical indicators")
            
            # Store original columns
            self.feature_names = df.columns.tolist()
            
            # Trend Indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # Momentum Indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['stoch_rsi'] = ta.momentum.stochrsi(df['close'])
            df['williams_r'] = ta.momentum.williams_r(high=df['high'], low=df['low'], close=df['close'])
            df['roc'] = ta.momentum.roc(df['close'], window=10)  # Rate of Change instead of Momentum
            
            # Volume Indicators
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['mfi'] = ta.volume.money_flow_index(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                volume=df['volume']
            )
            df['adi'] = ta.volume.acc_dist_index(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                volume=df['volume']
            )
            
            # Volatility Indicators
            df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
            df['atr'] = ta.volatility.average_true_range(
                high=df['high'], 
                low=df['low'], 
                close=df['close']
            )
            
            # MACD
            df['macd'] = ta.trend.macd(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['macd_diff'] = ta.trend.macd_diff(df['close'])
            
            # Update feature names after adding indicators
            self.feature_names = df.columns.tolist()
            logger.info(f"Features available: {self.feature_names}")
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise

    def prepare_data(self, symbol: str, interval: str, lookback: int, 
                    start_date: str = None, end_date: str = None) -> Tuple:
        """Prepare data for training or backtesting
        
        Args:
            symbol: Trading pair symbol
            interval: Data interval
            lookback: Number of past days to consider
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
        """
        try:
            # Fetch and process data
            df = self.fetch_data(symbol, interval)
            
            # Filter data by date range if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            df = self.add_technical_indicators(df)
            
            # Scale the features
            scaled_data = self.scaler.fit_transform(df)
            df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
            
            # Create sequences
            X, y = self._create_sequences(df_scaled, lookback)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.validation_split, shuffle=False
            )
            
            logger.info(f"Prepared data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def _create_sequences(self, df: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        data = df.values
        X, y = [], []
        
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback, df.columns.get_loc('close')])
            
        return np.array(X), np.array(y) 
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]