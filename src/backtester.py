import numpy as np
import pandas as pd
import logging
import random

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config, initial_capital=10000):
        self.config = config
        self.initial_capital = initial_capital
        
    def _calculate_reward(self, trades, current_price):
        """Enhanced reward calculation"""
        if not trades or not trades[-1]['active']:
            return -0.001  # Small penalty for not being in a trade
            
        trade = trades[-1]
        unrealized_pnl = (current_price - trade['entry_price']) * trade['position']
        position_size = abs(trade['position_size'])
        
        # Calculate return-based reward
        pct_return = unrealized_pnl / (trade['entry_price'] * position_size)
        
        # Add time decay to encourage faster decisions
        time_held = len(trade['price_history'])
        time_penalty = -0.0001 * time_held
        
        # Add risk-adjusted component
        volatility = np.std(trade['price_history']) / trade['entry_price']
        risk_adjustment = pct_return / (volatility + 1e-9)
        
        return risk_adjustment + time_penalty
        
    def _get_next_state(self, i, pred_ma_short, pred_ma_long, volatility, 
                        position, capital, y_true):
        """Get the next state for the RL agent"""
        next_idx = min(i + 1, len(y_true) - 1)
        return np.array([
            pred_ma_short[next_idx],
            pred_ma_long[next_idx],
            volatility[next_idx],
            position,
            capital / self.initial_capital
        ])
        
    def run_backtest(self, y_true, y_pred, agent=None, training=False, 
                    batch_update_freq=1, replay_batch_size=32):
        capital = self.initial_capital
        position = 0
        trades = []
        max_capital = capital
        episode_rewards = []
        
        # Pre-calculate signals for efficiency
        pred_ma_short = pd.Series(y_pred).rolling(window=5).mean().values
        pred_ma_long = pd.Series(y_pred).rolling(window=20).mean().values
        volatility = pd.Series(y_true).rolling(window=20).std().values
        
        update_counter = 0
        
        for i in range(20, len(y_true)):
            current_price = y_true[i]
            
            if training and agent:
                # Create state
                state = np.array([
                    pred_ma_short[i],
                    pred_ma_long[i],
                    volatility[i],
                    position,
                    capital / self.initial_capital
                ])
                
                # Get action from agent
                action = agent.act(state)
                
                # Calculate reward and store experience
                reward = self._calculate_reward(trades, current_price)
                episode_rewards.append(reward)
                
                next_state = self._get_next_state(
                    i, pred_ma_short, pred_ma_long, volatility,
                    position, capital, y_true
                )
                
                agent.remember(state, action, reward, next_state, i == len(y_true)-1)
                
                # Batch update
                update_counter += 1
                if update_counter >= batch_update_freq:
                    agent.replay(replay_batch_size)
                    update_counter = 0
            
            # Skip if volatility is too low
            if volatility[i] < self.config.min_volatility:
                continue
            
            # Check existing trade
            if position != 0:
                for trade in [t for t in trades if t['active']]:
                    exit_price = self._check_exit_conditions(
                        trade, current_price, pred_ma_short[i], pred_ma_long[i]
                    )
                    
                    if exit_price:
                        capital += self._close_trade(trade, exit_price, 0.001)
                        position = 0
            
            # Check for new trade
            else:
                signal = self._calculate_entry_signal(
                    pred_ma_short[i], pred_ma_long[i], 
                    volatility[i], trades
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
    
    def _calculate_entry_signal(self, pred_short, pred_long, volatility, trades):
        """Calculate entry signal based on predictions and current market conditions"""
        # Check if we have too many active trades
        active_trades = len([t for t in trades if t['active']])
        if active_trades >= self.config.max_trades:
            return 0
        
        # Calculate trend strength
        trend_strength = (pred_short - pred_long) / pred_long
        
        # Long entry conditions
        if (trend_strength > self.config.entry_threshold and 
            pred_short > pred_long):
            return 1
        
        # Short entry conditions
        elif (trend_strength < -self.config.entry_threshold and 
              pred_short < pred_long):
            return -1
        
        return 0

    def _open_trade(self, position, entry_price, position_size, capital, time_index):
        """Open a new trade"""
        return {
            'position': position,
            'entry_price': entry_price,
            'position_size': position_size,
            'price_history': [entry_price],
            'active': True,
            'max_price': entry_price,
            'trailing_stop': entry_price * (1 - self.config.trailing_stop),
            'exit_price': None,
            'entry_time': time_index,
            'exit_time': None
        }

    def _check_exit_conditions(self, trade, current_price, pred_short, pred_long):
        """Check if trade should be closed"""
        # Update price history and maximum price
        trade['price_history'].append(current_price)
        trade['max_price'] = max(trade['max_price'], current_price)
        
        # Calculate stop loss and take profit levels
        stop_loss = trade['entry_price'] * (1 - self.config.stop_loss)
        take_profit = trade['entry_price'] * (1 + self.config.take_profit)
        
        # Update trailing stop
        new_stop = current_price * (1 - self.config.trailing_stop)
        trade['trailing_stop'] = max(trade['trailing_stop'], new_stop)
        
        # Check exit conditions
        if current_price <= stop_loss:
            logger.info(f"Stop loss triggered at {current_price:.2f}")
            return current_price
        elif current_price >= take_profit:
            logger.info(f"Take profit triggered at {current_price:.2f}")
            return current_price
        elif current_price <= trade['trailing_stop']:
            logger.info(f"Trailing stop triggered at {current_price:.2f}")
            return current_price
        elif pred_short < pred_long:
            logger.info(f"MA crossover exit at {current_price:.2f}")
            return current_price
        
        return None

    def _close_trade(self, trade, exit_price, transaction_cost):
        """Close a trade and calculate returns"""
        trade['active'] = False
        trade['exit_price'] = exit_price
        trade['exit_time'] = len(trade['price_history'])
        
        # Calculate trade return
        position_size = trade['position_size']
        trade_return = (exit_price - trade['entry_price']) * trade['position']
        trade_return -= position_size * transaction_cost  # Apply transaction cost
        
        # Calculate profit/loss in dollars and percentage
        trade['profit_loss'] = trade_return
        trade['return_pct'] = (trade_return / position_size) * 100
        
        # Log trade result
        direction = "LONG" if trade['position'] > 0 else "SHORT"
        logger.info(f"\nTrade {direction} closed:")
        logger.info(f"Entry: ${trade['entry_price']:.2f}")
        logger.info(f"Exit: ${exit_price:.2f}")
        logger.info(f"P/L: ${trade['profit_loss']:.2f} ({trade['return_pct']:.2f}%)")
        
        return position_size + trade_return

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