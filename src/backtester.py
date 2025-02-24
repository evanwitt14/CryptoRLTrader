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
        """Enhanced reward calculation with immediate feedback"""
        # Base case - no position
        if not trades or not trades[-1]['active']:
            # Penalize missed opportunities more when signals are strong
            signal_strength = abs(current_data['returns']) + abs(current_data['macd'])
            missed_reward = -0.1 * signal_strength
            return missed_reward
            
        trade = trades[-1]
        unrealized_pnl = (current_price - trade['entry_price']) * trade['position']
        position_size = abs(trade['position_size'])
        
        # Calculate returns
        pct_return = unrealized_pnl / (trade['entry_price'] * position_size)
        
        # Technical signal alignment
        signal_reward = 0.0
        if trade['position'] > 0:  # Long position
            if current_data['close'] > current_data['bb_upper']:
                signal_reward += 0.2  # Strong uptrend
            if current_data['rsi'] < 30:
                signal_reward += 0.1  # Oversold bounce potential
        else:  # Short position
            if current_data['close'] < current_data['bb_lower']:
                signal_reward += 0.2  # Strong downtrend
            if current_data['rsi'] > 70:
                signal_reward += 0.1  # Overbought reversal potential
                
        # Time decay and risk adjustment
        holding_penalty = -0.001 * len(trade['price_history'])
        volatility_penalty = -0.1 * current_data['volatility']
        
        # Combine rewards
        reward = (
            pct_return * 300.0 +      # Stronger emphasis on actual returns
            signal_reward +           # Technical signal alignment
            holding_penalty +         # Penalize long holds
            volatility_penalty       # Risk adjustment
        )
        
        # Multiply reward for profitable exits
        if not trade['active'] and pct_return > 0:
            reward *= 5.0
            
        return reward
        
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
        capital = self.initial_capital
        position = 0
        trades = []
        update_counter = 0
        max_capital = capital
        episode_rewards = []
        
        # Calculate signals
        df = pd.DataFrame({'close': y_true})
        df['pred'] = y_pred
        
        # Add more price action signals
        df['returns'] = df['close'].pct_change()
        df['pred_returns'] = df['pred'].pct_change()
        df['momentum'] = df['returns'].rolling(10).mean()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Calculate trend
        df['sma20'] = df['close'].rolling(20).mean()
        df['trend'] = np.where(df['close'] > df['sma20'], 1, -1)
        
        # Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Fill NaN values
        df = df.ffill().bfill()  # Using newer pandas methods
        
        # Add exploration noise during training
        if training and agent:
            noise_scale = max(0.002, agent.epsilon * 0.01)
            noise = np.random.normal(0, noise_scale, len(y_pred))
            y_pred = y_pred + noise
        
        for i in range(20, len(y_true)):
            current_price = y_true[i]
            
            if training and agent:
                # Get state and action
                state = np.array([
                    df['returns'].iloc[i],
                    df['pred_returns'].iloc[i],
                    df['rsi'].iloc[i] / 100,
                    df['trend'].iloc[i],
                    df['volatility'].iloc[i],
                    position,
                    capital / self.initial_capital,
                    len([t for t in trades if t['active']]) / self.config.max_trades
                ])
                
                action = agent.act(state)
                
                # Convert action to trading decision
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
                            'profit_loss': 0.0  # Initialize profit/loss
                        })
                        position = 1
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
                            'profit_loss': 0.0  # Initialize profit/loss
                        })
                        position = -1
                
                # Calculate reward and get next state
                reward = self._calculate_reward(trades, current_price, df.iloc[i])
                next_state = self._get_next_state(i, df, position, capital, trades)
                
                # Store experience
                agent.remember(state, action, reward, next_state, i == len(y_true)-1)
                
                # Batch update
                update_counter += 1
                if update_counter >= batch_update_freq:
                    agent.replay(replay_batch_size)
                    update_counter = 0
            
            # Skip if volatility is too low
            if df['volatility'].iloc[i] < self.config.min_volatility:
                continue
            
            # Check existing trade
            if position != 0:
                for trade in [t for t in trades if t['active']]:
                    exit_price = self._check_exit_conditions(
                        trade, current_price, df['returns'].iloc[i], df['pred_returns'].iloc[i]
                    )
                    
                    if exit_price:
                        capital += self._close_trade(trade, exit_price, 0.001)
                        position = 0
            
            # Check for new trade
            else:
                signal = self._calculate_entry_signal(
                    df['returns'].iloc[i], df['pred_returns'].iloc[i], 
                    df['volatility'].iloc[i], trades
                )
                
                if signal != 0:
                    position = signal
                    trade = self._open_trade(position, current_price, 
                                           capital * self.config.risk_per_trade, 
                                           capital, i)
                    trades.append(trade)
                    capital -= trade['position_size']
            
            # Update maximum capital
            max_capital = max(max_capital, capital)
        
        metrics = self._calculate_backtest_metrics(capital, trades, max_capital)
        metrics['trades'] = trades
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
        """Check if trade should be closed"""
        # Update price history and extremes
        trade['price_history'].append(current_price)
        if trade['position'] > 0:  # Long position
            trade['max_price'] = max(trade['max_price'], current_price)
            # Update trailing stop for longs
            new_stop = current_price * (1 - self.config.trailing_stop)
            trade['trailing_stop'] = max(trade['trailing_stop'], new_stop)
        else:  # Short position
            trade['min_price'] = min(trade['min_price'], current_price)
            # Update trailing stop for shorts
            new_stop = current_price * (1 + self.config.trailing_stop)
            trade['trailing_stop'] = min(trade['trailing_stop'], new_stop)
        
        # Calculate stop loss and take profit levels
        if trade['position'] > 0:
            stop_loss = trade['entry_price'] * (1 - self.config.stop_loss)
            take_profit = trade['entry_price'] * (1 + self.config.take_profit)
            # Exit conditions for longs
            if current_price <= stop_loss:
                logger.info(f"Long stop loss triggered at {current_price:.2f}")
                return current_price
            elif current_price >= take_profit:
                logger.info(f"Long take profit triggered at {current_price:.2f}")
                return current_price
            elif current_price <= trade['trailing_stop']:
                logger.info(f"Long trailing stop triggered at {current_price:.2f}")
                return current_price
        else:
            stop_loss = trade['entry_price'] * (1 + self.config.stop_loss)
            take_profit = trade['entry_price'] * (1 - self.config.take_profit)
            # Exit conditions for shorts
            if current_price >= stop_loss:
                logger.info(f"Short stop loss triggered at {current_price:.2f}")
                return current_price
            elif current_price <= take_profit:
                logger.info(f"Short take profit triggered at {current_price:.2f}")
                return current_price
            elif current_price >= trade['trailing_stop']:
                logger.info(f"Short trailing stop triggered at {current_price:.2f}")
                return current_price
        
        return None

    def _close_trade(self, trade, exit_price, slippage=0.001):
        """Close a trade and calculate profit/loss"""
        # Calculate slippage
        actual_exit_price = exit_price * (1 - slippage if trade['position'] > 0 else 1 + slippage)
        
        # Calculate profit/loss
        price_diff = (actual_exit_price - trade['entry_price']) * trade['position']
        trade['profit_loss'] = price_diff * trade['position_size']
        trade['exit_price'] = actual_exit_price
        trade['exit_time'] = len(trade['price_history'])
        trade['active'] = False
        
        # Return realized profit/loss
        return trade['profit_loss']

    def _calculate_backtest_metrics(self, capital, trades, max_capital):
        """Calculate backtest performance metrics"""
        total_return = (capital - self.initial_capital) / self.initial_capital
        n_trades = len(trades)
        
        if n_trades == 0:
            return {
                'total_return': 0,
                'n_trades': 0,
                'avg_trade_return': 0,
                'final_capital': capital,
                'win_rate': 0,
                'profit_factor': 0
            }
        
        # Calculate detailed metrics
        profitable_trades = [t for t in trades if t['profit_loss'] > 0]
        losing_trades = [t for t in trades if t['profit_loss'] <= 0]
        
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

    def _calculate_position_size(self, capital, current_price):
        """Calculate position size based on risk per trade"""
        risk_amount = capital * self.config.risk_per_trade
        position_size = min(
            risk_amount / (current_price * self.config.stop_loss),
            capital * self.config.max_position_size / current_price
        )
        return position_size 