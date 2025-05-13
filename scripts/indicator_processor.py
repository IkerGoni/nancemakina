#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger('indicator_processor')

class VectorizedIndicatorProcessor:
    """
    Optimized indicator processor that uses numpy vectorized operations
    for faster calculation of technical indicators.
    
    This processor calculates indicators incrementally when possible,
    reducing redundant calculations and improving performance.
    """
    
    def __init__(self):
        self._sma_cache = {}
        self._ema_cache = {}
        self._atr_cache = {}
        
    def calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average using vectorized operations.
        
        Args:
            prices: Array of price values
            period: SMA period
            
        Returns:
            Array of SMA values
        """
        # Use pandas rolling window which is highly optimized
        return pd.Series(prices).rolling(window=period).mean().to_numpy()
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average using vectorized operations.
        
        Args:
            prices: Array of price values
            period: EMA period
            
        Returns:
            Array of EMA values
        """
        # Use pandas exponential weighted mean
        return pd.Series(prices).ewm(span=period, adjust=False).mean().to_numpy()
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Average True Range using vectorized operations.
        
        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: ATR period
            
        Returns:
            Array of ATR values
        """
        # Calculate true range
        high_low = high - low
        high_close_prev = np.abs(high[1:] - close[:-1])
        low_close_prev = np.abs(low[1:] - close[:-1])
        
        # Insert a zero at the beginning for the previous close comparison
        high_close_prev = np.insert(high_close_prev, 0, 0)
        low_close_prev = np.insert(low_close_prev, 0, 0)
        
        # Calculate true range
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Calculate ATR
        atr = pd.Series(tr).rolling(window=period).mean().to_numpy()
        
        return atr
    
    def calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Relative Strength Index using vectorized operations.
        
        Args:
            prices: Array of price values
            period: RSI period
            
        Returns:
            Array of RSI values
        """
        # Calculate price changes
        delta = np.diff(prices)
        delta = np.insert(delta, 0, 0)
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate average gains and losses
        avg_gains = pd.Series(gains).rolling(window=period).mean().to_numpy()
        avg_losses = pd.Series(losses).rolling(window=period).mean().to_numpy()
        
        # Calculate RS and RSI
        rs = np.where(avg_losses == 0, 100, avg_gains / avg_losses)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands using vectorized operations.
        
        Args:
            prices: Array of price values
            period: Bollinger Bands period
            num_std: Number of standard deviations for the bands
            
        Returns:
            Tuple of (middle band, upper band, lower band)
        """
        # Calculate middle band (SMA)
        middle_band = self.calculate_sma(prices, period)
        
        # Calculate standard deviation
        rolling_std = pd.Series(prices).rolling(window=period).std().to_numpy()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        
        return middle_band, upper_band, lower_band
    
    def calculate_macd(self, prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD using vectorized operations.
        
        Args:
            prices: Array of price values
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        # Calculate fast and slow EMAs
        fast_ema = self.calculate_ema(prices, fast_period)
        slow_ema = self.calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all common indicators in a single pass.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Extract price arrays
        close_prices = result['close'].to_numpy()
        high_prices = result['high'].to_numpy()
        low_prices = result['low'].to_numpy()
        
        # Calculate SMAs
        result['sma_20'] = self.calculate_sma(close_prices, 20)
        result['sma_50'] = self.calculate_sma(close_prices, 50)
        result['sma_100'] = self.calculate_sma(close_prices, 100)
        result['sma_200'] = self.calculate_sma(close_prices, 200)
        
        # Calculate EMAs
        result['ema_12'] = self.calculate_ema(close_prices, 12)
        result['ema_26'] = self.calculate_ema(close_prices, 26)
        
        # Calculate ATR
        result['atr_14'] = self.calculate_atr(high_prices, low_prices, close_prices, 14)
        
        # Calculate RSI
        result['rsi_14'] = self.calculate_rsi(close_prices, 14)
        
        # Calculate Bollinger Bands
        mid, upper, lower = self.calculate_bollinger_bands(close_prices, 20, 2.0)
        result['bb_middle'] = mid
        result['bb_upper'] = upper
        result['bb_lower'] = lower
        
        # Calculate MACD
        macd, signal, hist = self.calculate_macd(close_prices)
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist
        
        return result

if __name__ == "__main__":
    # Simple test code
    import matplotlib.pyplot as plt
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=200)
    np.random.seed(42)
    
    # Create a sine wave with noise
    x = np.linspace(0, 4*np.pi, 200)
    prices = 100 + 20*np.sin(x) + np.random.normal(0, 5, 200).cumsum()
    
    # Create test dataframe
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.rand(200) * 5,
        'low': prices - np.random.rand(200) * 5,
        'close': prices + np.random.normal(0, 2, 200),
        'volume': np.random.rand(200) * 1000
    })
    
    # Initialize processor
    processor = VectorizedIndicatorProcessor()
    
    # Calculate indicators
    result = processor.calculate_all_indicators(df)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(result['close'], label='Close')
    plt.plot(result['sma_20'], label='SMA 20')
    plt.plot(result['sma_50'], label='SMA 50')
    plt.plot(result['sma_200'], label='SMA 200')
    plt.legend()
    plt.title('Price and SMAs')
    
    plt.subplot(3, 1, 2)
    plt.plot(result['rsi_14'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='-')
    plt.axhline(y=30, color='g', linestyle='-')
    plt.legend()
    plt.title('RSI')
    
    plt.subplot(3, 1, 3)
    plt.plot(result['macd'], label='MACD')
    plt.plot(result['macd_signal'], label='Signal')
    plt.bar(range(len(result)), result['macd_hist'], width=0.5, label='Histogram')
    plt.legend()
    plt.title('MACD')
    
    plt.tight_layout()
    plt.savefig('indicator_test.png')
    plt.close()
    
    print("Test completed and plot saved to indicator_test.png") 