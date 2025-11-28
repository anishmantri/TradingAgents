import unittest
import pandas as pd
import numpy as np
from tradingagents.utils.quant_indicators import (
    calculate_volatility_scaled_rsi,
    calculate_impulse_macd,
    calculate_market_regime,
    calculate_vpt
)

class TestQuantIndicators(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=200)
        self.df = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + 100,
            'high': np.random.randn(200).cumsum() + 105,
            'low': np.random.randn(200).cumsum() + 95,
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        # Ensure high/low/close logic
        self.df['high'] = self.df[['open', 'close']].max(axis=1) + 1
        self.df['low'] = self.df[['open', 'close']].min(axis=1) - 1
        
        # Rename columns to match what stockstats expects (lowercase)
        # stockstats handles case insensitivity by lowercasing, so we avoid duplicates.
        # self.df['Close'] = self.df['close']
        # self.df['Volume'] = self.df['volume']

    def test_volatility_scaled_rsi(self):
        rsi_z = calculate_volatility_scaled_rsi(self.df)
        self.assertIsInstance(rsi_z, pd.Series)
        self.assertEqual(len(rsi_z), 200)
        # Check if values are roughly Z-scores (centered around 0)
        # First few will be NaN due to rolling windows
        valid_values = rsi_z.dropna()
        self.assertTrue(len(valid_values) > 0)
        self.assertTrue(valid_values.mean() < 10) # Should be small number, not 50

    def test_impulse_macd(self):
        impulse_df = calculate_impulse_macd(self.df)
        self.assertIsInstance(impulse_df, pd.DataFrame)
        self.assertIn('macd', impulse_df.columns)
        self.assertIn('impulse', impulse_df.columns)
        
        # Check impulse values are -1, 0, 1
        unique_impulses = impulse_df['impulse'].unique()
        for val in unique_impulses:
            self.assertIn(val, [-1, 0, 1])

    def test_market_regime(self):
        regime = calculate_market_regime(self.df)
        self.assertIsInstance(regime, pd.Series)
        
        # Check for expected categories
        unique_regimes = regime.unique()
        valid_regimes = [
            "Trending Bullish", 
            "Trending Bearish", 
            "Mean Reverting / Ranging", 
            "Neutral / Transition"
        ]
        for r in unique_regimes:
            self.assertIn(r, valid_regimes)

    def test_vpt(self):
        vpt = calculate_vpt(self.df)
        self.assertIsInstance(vpt, pd.Series)
        self.assertEqual(len(vpt), 200)
        self.assertFalse(vpt.isnull().all())

if __name__ == '__main__':
    unittest.main()
