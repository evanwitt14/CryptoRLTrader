import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from .config import Config
from .main import setup_args

class TestCryptoPrediction(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_config_initialization(self):
        """Test if config directories are created properly"""
        self.assertTrue(self.config.data_dir.exists())
        self.assertTrue(self.config.model_dir.exists())
        self.assertEqual(self.config.lstm_units, 128)
        self.assertEqual(self.config.batch_size, 32)

    @patch('argparse.ArgumentParser.parse_args')
    def test_argument_parsing(self, mock_args):
        """Test command line argument parsing"""
        mock_args.return_value = Mock(
            symbol='BTC/USDT',
            interval='1d',
            lookback=60,
            epochs=100
        )
        args = setup_args()
        self.assertEqual(args.symbol, 'BTC/USDT')
        self.assertEqual(args.interval, '1d')
        self.assertEqual(args.lookback, 60)
        self.assertEqual(args.epochs, 100)

if __name__ == '__main__':
    unittest.main() 