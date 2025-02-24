import numpy as np
import pandas as pd
import logging
import random
import ta

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config, initial_capital=10000):
        self.config = config
        self.initial_capital = initial_capital
        
    def _calculate_reward(self, trades, current_price, current_data):
        if not trades or not trades[-1]['active']:
            return -0.001  # Smaller penalty
        
        trade = trades[-1]
        hold_time = len(trade['price_history'])
        
        # Penalize extremely short trades
        if hold_time < self.config.min_hold_bars:
            return -0.02
        
        # Calculate regular reward
        price_diff = current_price - trade['entry_price']
        unrealized_pnl = price_diff * trade['position_size'] * trade['position']
        pct_return = unrealized_pnl / (trade['entry_price'] * trade['position_size'])
        
        return pct_return * self.config.reward_scaling
        
    def _get_next_state(self, i, df, position, capital, trades):
        """Get the next state for the RL agent"""
        next_idx = min(i + 1, len(df) - 1)
        return np.array([
            df['returns'].iloc[next_idx],           # Current return
            df['pred_returns'].iloc[next_idx],      # Predicted return
            df['rsi'].iloc[next_idx] / 100,         # RSI normalized
            df['trend'].iloc[next_idx],             # Trend direction
            df['volatility'].iloc[next_idx],        # Market volatility
            position,                               # Current position
            capital / self.initial_capital,         # Capital ratio
            len([t for t in trades if t['active']]) / self.config.max_trades  # Trade capacity
        ])
        
    def run_backtest(self, y_true, y_pred, agent=None, training=False, 
                    batch_update_freq=1, replay_batch_size=32):
        """Run backtest with proper trade management"""
        capital = self.initial_capital
        max_capital = capital
        position = 0
        trades = []
        
        df = self._prepare_backtest_data(y_true, y_pred)
        
        for i in range(20, len(y_true)):
            current_price = y_true[i]
            
            # Check existing trades first
            if trades and trades[-1]['active']:
                exit_price = self._check_exit_conditions(
                    trades[-1], current_price, 
                    df['returns'].iloc[i], df['pred_returns'].iloc[i]
                )
                if exit_price:
                    pnl = self._close_trade(trades[-1], exit_price)
                    capital += pnl
                    position = 0
                    logger.info(f"Trade closed - PnL: ${pnl:.2f}, Capital: ${capital:.2f}")
            
            # Get state and action for RL agent
            if training and agent:
                state = self._get_state(i, df, position, capital, trades)
                action = agent.act(state)
                
                # Execute trades based on action
                if action == 0:  # Hold
                    pass
                elif action == 1 and position <= 0:  # Buy
                    if len([t for t in trades if t['active']]) < self.config.max_trades:
                        position_size = self._calculate_position_size(capital, current_price)
                        trades.append({
                            'type': 'long',
                            'entry_price': current_price,
                            'position': 1,
                            'position_size': position_size,
                            'active': True,
                            'price_history': [],
                            'max_price': current_price,
                            'min_price': current_price,
                            'trailing_stop': current_price * (1 - self.config.trailing_stop),
                            'exit_price': None,
                            'entry_time': i,
                            'exit_time': None,
                            'profit_loss': 0.0
                        })
                        position = 1
                        logger.info(f"Long entry at {current_price:.2f}")
                elif action == 2 and position >= 0:  # Sell
                    if len([t for t in trades if t['active']]) < self.config.max_trades:
                        position_size = self._calculate_position_size(capital, current_price)
                        trades.append({
                            'type': 'short',
                            'entry_price': current_price,
                            'position': -1,
                            'position_size': position_size,
                            'active': True,
                            'price_history': [],
                            'max_price': current_price,
                            'min_price': current_price,
                            'trailing_stop': current_price * (1 + self.config.trailing_stop),
                            'exit_price': None,
                            'entry_time': i,
                            'exit_time': None,
                            'profit_loss': 0.0
                        })
                        position = -1
                        logger.info(f"Short entry at {current_price:.2f}")

            # Update max capital
            max_capital = max(max_capital, capital)
            
            # Store experience if training
            if training and agent:
                reward = self._calculate_reward(trades, current_price, df.iloc[i])
                next_state = self._get_state(i, df, position, capital, trades)
                done = i == len(y_true)-1
                agent.remember(state, action, reward, next_state, done)
                
                # Batch update
                if i % batch_update_freq == 0:
                    agent.replay(replay_batch_size)
        
        # Close any remaining open trades at the end
        if trades and trades[-1]['active']:
            pnl = self._close_trade(trades[-1], y_true[-1])
            capital += pnl
            
        # Calculate and return metrics
        metrics = self._calculate_backtest_metrics(capital, trades, max_capital)
        return metrics
    
    def _calculate_entry_signal(self, returns, pred_returns, volatility, trades):
        """More selective entry signal calculation"""
        # Check active trades
        active_trades = len([t for t in trades if t['active']])
        if active_trades >= self.config.max_trades:
            return 0
        
        # Calculate trend strength and volatility adjustment
        trend_strength = (pred_returns - returns) / (volatility + 1e-9)
        vol_multiplier = min(1.0, self.config.min_volatility / volatility) if volatility > 0 else 0
        
        # Enhanced entry conditions
        if trend_strength > self.config.entry_threshold * vol_multiplier:
            if returns > 0 and pred_returns > returns:
                return 1  # Long
        elif trend_strength < -self.config.entry_threshold * vol_multiplier:
            if returns < 0 and pred_returns < returns:
                return -1  # Short
            
        return 0  # No trade

    def _open_trade(self, position, entry_price, position_size, capital, time_index):
        """Open a new trade"""
        return {
            'position': position,
            'entry_price': entry_price,
            'position_size': position_size,
            'price_history': [entry_price],
            'active': True,
            'max_price': entry_price,
            'min_price': entry_price,
            'trailing_stop': entry_price * (1 - self.config.trailing_stop if position > 0 else 1 + self.config.trailing_stop),
            'exit_price': None,
            'entry_time': time_index,
            'exit_time': None
        }

    def _check_exit_conditions(self, trade, current_price, returns, pred_returns):
        """Enhanced exit conditions with better loss management"""
        if not trade['active']:
            return None
        
        # Force minimum hold period
        trade['price_history'].append(current_price)
        if len(trade['price_history']) < self.config.min_hold_bars:
            return None
        
        # Update trade metrics
        trade['max_price'] = max(trade['max_price'], current_price)
        trade['min_price'] = min(trade['min_price'], current_price)
        
        # Calculate current trade metrics
        price_diff = current_price - trade['entry_price']
        unrealized_pnl = price_diff * trade['position_size'] * trade['position']
        pct_return = unrealized_pnl / (trade['entry_price'] * trade['position_size'])
        
        # Exit conditions
        if trade['position'] > 0:  # Long position
            if (current_price <= trade['entry_price'] * (1 - self.config.stop_loss) or
                current_price <= trade['max_price'] * (1 - self.config.trailing_stop) or
                len(trade['price_history']) >= self.config.max_hold_bars):
                return current_price
        else:  # Short position
            if (current_price >= trade['entry_price'] * (1 + self.config.stop_loss) or
                current_price >= trade['min_price'] * (1 + self.config.trailing_stop) or
                len(trade['price_history']) >= self.config.max_hold_bars):
                return current_price
            
        return None

    def _close_trade(self, trade, exit_price, slippage=0.001):
        """Close trade with detailed logging"""
        actual_exit_price = exit_price * (1 - slippage if trade['position'] > 0 else 1 + slippage)
        
        price_diff = actual_exit_price - trade['entry_price']
        pnl = price_diff * trade['position_size'] * trade['position']
        
        trade['exit_price'] = actual_exit_price
        trade['exit_time'] = len(trade['price_history'])
        trade['active'] = False
        trade['profit_loss'] = pnl
        
        logger.info(f"""
        Trade closed:
        Type: {trade['type']}
        Entry: ${trade['entry_price']:.2f}
        Exit: ${actual_exit_price:.2f}
        PnL: ${pnl:.2f}
        Duration: {len(trade['price_history'])} bars
        """)
        
        return pnl
        
    def _calculate_backtest_metrics(self, capital, trades, max_capital):
        """Calculate backtest performance metrics"""
        metrics = {}
        
        # Basic metrics
        n_trades = len(trades)
        if n_trades == 0:
            return {
                'total_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade': 0,
                'n_trades': 0
            }
        
        # Calculate trade statistics
        profitable_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]
        
        total_return = (capital - self.initial_capital) / self.initial_capital
        win_rate = len(profitable_trades) / n_trades * 100
        total_profit = sum(t['profit_loss'] for t in profitable_trades)
        total_loss = abs(sum(t['profit_loss'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Log summary
        logger.info("\n=== Trading Summary ===")
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Final Capital: ${capital:.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Number of Trades: {n_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Total Profit: ${total_profit:.2f}")
        logger.info(f"Total Loss: ${total_loss:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info("====================\n")
        
        return {
            'total_return': total_return,
            'n_trades': n_trades,
            'avg_trade_return': total_return / n_trades,
            'final_capital': capital,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss
        } 

    def _calculate_position_size(self, capital, current_price, volatility=0.01):
        """Dynamic position sizing based on volatility"""
        # Base risk amount
        risk_amount = capital * self.config.risk_per_trade
        
        # Adjust position size based on volatility
        vol_factor = min(1.0, self.config.min_volatility / volatility)
        
        # Calculate base position size
        base_size = min(
            risk_amount / (current_price * self.config.stop_loss),
            capital * self.config.max_position_size / current_price
        )
        
        return base_size * vol_factor  # Changed from max_size to base_size

    def _prepare_backtest_data(self, y_true, y_pred):
        """Prepare data for backtesting with technical indicators"""
        df = pd.DataFrame({
            'close': y_true,
            'pred': y_pred
        })
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['pred_returns'] = df['pred'].pct_change()
        
        # Add technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd_diff()
        
        # Calculate trend
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['trend'] = np.where(df['close'] > df['sma_20'], 1, -1)
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Forward fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df 

    def _get_state(self, i, df, position, capital, trades):
        """Get current state for the RL agent"""
        return np.array([
            df['returns'].iloc[i],
            df['pred_returns'].iloc[i],
            df['rsi'].iloc[i] / 100,
            df['trend'].iloc[i],
            df['volatility'].iloc[i],
            position,
            capital / self.initial_capital,
            len([t for t in trades if t['active']]) / self.config.max_trades
        ]) 