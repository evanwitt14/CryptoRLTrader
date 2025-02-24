import argparse
import logging
import copy
import torch
from config import Config
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from backtester import Backtester
from reinforcement_trainer import TradingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    parser.add_argument('--interval', type=str, default='1d')
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--episodes', type=int, default=1000)
    return parser.parse_args()

def train_agent(args, config, n_episodes=1000):
    data_processor = DataProcessor(config)
    model_trainer = ModelTrainer(config, data_processor)
    backtester = Backtester(config)
    
    # Load saved model and get its parameters
    logger.info("Loading saved model...")
    ensemble_model = model_trainer.load_ensemble(args.symbol)
    original_lookback = ensemble_model.rf.n_features_in_ // 25
    
    # Pre-fetch all data once
    logger.info("Loading training data...")
    _, X_test, _, y_test = data_processor.prepare_data(
        symbol=args.symbol,
        interval=args.interval,
        lookback=original_lookback
    )
    predictions = ensemble_model.predict(X_test)
    
    # Initialize agent with improved parameters
    state_size = 5
    action_size = 3
    agent = TradingAgent(
        state_size=state_size,
        action_size=action_size,
        config=config,
        learning_rate=0.0005,  # Lower learning rate
        gamma=0.98,  # Higher discount factor
        epsilon_decay=0.997,  # Slower exploration decay
        memory_size=5000  # Larger memory
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.model = agent.model.to(device)
    
    best_return = float('-inf')
    best_agent = None
    no_improvement_count = 0
    
    # Batch processing parameters
    batch_update_freq = 20  # Less frequent updates
    replay_batch_size = 128  # Larger batch size
    
    print(f"\nTraining Progress: 0/{n_episodes} episodes", end='', flush=True)
    
    for episode in range(n_episodes):
        results = backtester.run_backtest(
            y_test, predictions,
            agent=agent,
            training=True,
            batch_update_freq=batch_update_freq,
            replay_batch_size=replay_batch_size
        )
        
        if results['total_return'] > best_return:
            best_return = results['total_return']
            best_agent = copy.deepcopy(agent)
            print(f"\nNew best return: {best_return:.2%}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Early stopping if no improvement for many episodes
        if no_improvement_count >= 50:
            print("\nStopping early due to no improvement")
            break
            
        print(f"\rTraining Progress: {episode + 1}/{n_episodes} episodes", end='', flush=True)
    
    print("\nTraining completed!")
    print(f"Best return achieved: {best_return:.2%}")
    
    # Save best agent
    torch.save(best_agent.model.state_dict(), f"models/rl_agent_{args.symbol.replace('/', '_')}.pth")

if __name__ == "__main__":
    args = setup_args()
    config = Config()
    
    logger.info(f"Starting RL training for {args.symbol} with {args.episodes} episodes")
    train_agent(args, config, n_episodes=args.episodes) 