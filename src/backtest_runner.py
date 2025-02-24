import argparse
import logging
from config import Config
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from backtester import Backtester
from visualizer import Visualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    parser.add_argument('--interval', type=str, default='1d')
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--start_date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date for backtest (YYYY-MM-DD)')
    return parser.parse_args()

def main():
    args = setup_args()
    config = Config()
    
    try:
        # Initialize components
        data_processor = DataProcessor(config)
        model_trainer = ModelTrainer(config, data_processor)
        visualizer = Visualizer(config)
        
        # Load saved model
        logger.info("Loading saved model...")
        ensemble_model = model_trainer.load_ensemble(args.symbol)
        
        # Get fresh data for backtesting
        logger.info("Loading backtest data...")
        _, X_test, _, y_test = data_processor.prepare_data(
            symbol=args.symbol,
            interval=args.interval,
            lookback=args.lookback,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = ensemble_model.predict(X_test)
        
        # Run backtest with different parameters
        logger.info("Running backtest...")
        backtester = Backtester(config=config)
        backtest_results = backtester.run_backtest(y_test, predictions)
        
        # Log results
        logger.info("\nBacktest Results:")
        logger.info(f"Total Return: {backtest_results['total_return']:.2%}")
        logger.info(f"Number of Trades: {backtest_results['n_trades']}")
        logger.info(f"Average Trade Return: {backtest_results['avg_trade_return']:.2%}")
        logger.info(f"Final Capital: ${backtest_results['final_capital']:.2f}")
        
        # Visualize results
        visualizer.plot_predictions(y_test, predictions)
        visualizer.plot_trades(y_test, predictions, backtest_results['trades'])
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 