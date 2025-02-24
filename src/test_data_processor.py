import unittest
import logging
from .config import Config
from .data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up test...")
        self.config = Config()
        self.processor = DataProcessor(self.config)

    def test_fetch_data(self):
        logger.info("Testing data fetching...")
        df = self.processor.fetch_data('BTC/USDT', '1d')
        logger.info(f"Fetched dataframe shape: {df.shape}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        self.assertFalse(df.empty)
        self.assertTrue('close' in df.columns)
        self.assertTrue('volume' in df.columns)

    def test_technical_indicators(self):
        logger.info("Testing technical indicators...")
        df = self.processor.fetch_data('BTC/USDT', '1d')
        df_with_indicators = self.processor.add_technical_indicators(df)
        
        logger.info(f"Columns after adding indicators: {df_with_indicators.columns.tolist()}")
        logger.info(f"Data shape after adding indicators: {df_with_indicators.shape}")
        
        for indicator in ['rsi', 'macd', 'bb_high', 'sma_20']:
            self.assertTrue(indicator in df_with_indicators.columns)

if __name__ == '__main__':
    unittest.main() 