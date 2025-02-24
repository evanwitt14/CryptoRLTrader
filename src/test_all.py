import unittest
import logging
import torch
import numpy as np
from pathlib import Path

from .config import Config
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .visualizer import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCryptoPrediction(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.visualizer = Visualizer()

    def test_full_pipeline(self):
        """Test the entire prediction pipeline"""
        try:
            # 1. Test data processing
            logger.info("Testing data processing...")
            X_train, X_test, y_train, y_test = self.data_processor.prepare_data(
                symbol='BTC/USDT',
                interval='1d',
                lookback=60
            )
            self.assertIsNotNone(X_train)
            self.assertIsNotNone(y_train)
            logger.info(f"Training data shape: {X_train.shape}")

            # 2. Test model training
            logger.info("Testing model training...")
            ensemble_model = self.model_trainer.train_ensemble(
                X_train, X_test, y_train, y_test,
                epochs=2  # Small number for testing
            )
            self.assertIsNotNone(ensemble_model)

            # 3. Test predictions
            logger.info("Testing predictions...")
            predictions = ensemble_model.predict(X_test)
            self.assertEqual(len(predictions), len(y_test))

            # 4. Test visualization
            logger.info("Testing visualization...")
            pred_plot = self.visualizer.plot_predictions(y_test, predictions)
            self.assertIsNotNone(pred_plot)

            metrics_plot = self.visualizer.plot_model_performance(
                self.model_trainer.get_metrics()
            )
            self.assertIsNotNone(metrics_plot)

            logger.info("All tests passed successfully!")

        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main(verbosity=2) 