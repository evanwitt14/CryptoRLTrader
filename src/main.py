import argparse
import logging
from pathlib import Path

from config import Config
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizer import Visualizer
from backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Trading pair symbol')
    parser.add_argument('--interval', type=str, default='1d',
                       help='Time interval for data')
    parser.add_argument('--lookback', type=int, default=60,
                       help='Number of past days to consider')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    return parser.parse_args()

def main():
    args = setup_args()
    config = Config()
    
    try:
        # Initialize components
        data_processor = DataProcessor(config)
        model_trainer = ModelTrainer(config, data_processor)
        visualizer = Visualizer(config)

        # Load and process data
        logger.info("Loading and processing data...")
        X_train, X_test, y_train, y_test = data_processor.prepare_data(
            symbol=args.symbol,
            interval=args.interval,
            lookback=args.lookback
        )

        # Train models
        logger.info("Training models...")
        ensemble_model = model_trainer.train_ensemble(
            X_train, X_test, y_train, y_test,
            epochs=args.epochs
        )
        
        # Save the trained model
        logger.info("Saving trained model...")
        model_trainer.save_ensemble(ensemble_model, args.symbol)

        # Make predictions
        logger.info("Making predictions...")
        predictions = ensemble_model.predict(X_test)

        # Visualize results
        logger.info("Generating visualizations...")
        visualizer.plot_predictions(y_test, predictions)
        visualizer.plot_model_performance(model_trainer.get_metrics())

        # Add backtesting
        logger.info("Running backtest...")
        backtester = Backtester(config=config)
        backtest_results = backtester.run_backtest(y_test, predictions)
        
        logger.info("\nBacktest Results:")
        logger.info(f"Total Return: {backtest_results['total_return']:.2%}")
        logger.info(f"Number of Trades: {backtest_results['n_trades']}")
        logger.info(f"Average Trade Return: {backtest_results['avg_trade_return']:.2%}")
        logger.info(f"Final Capital: ${backtest_results['final_capital']:.2f}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 