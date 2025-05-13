import pandas as pd
from collections import deque, defaultdict
import logging
from typing import Dict, Deque, Optional, List, Tuple, Callable
import numpy as np
import asyncio
from datetime import datetime
import sys
sys.path.append('scripts')

from src.models import Kline
from src.config_loader import ConfigManager
from src.connectors import convert_symbol_to_api_format # For consistent symbol usage

# Import our new vectorized indicator processor
from scripts.indicator_processor import VectorizedIndicatorProcessor

logger = logging.getLogger(__name__)

# Define a maximum length for k-line deques to prevent memory issues
# Needs to be at least max(sma_long_period) + some buffer for calculations
# For SMA 200, we need at least 200 candles. Let's use a bit more.
MAX_KLINE_BUFFER_LENGTH = 300 

class DataProcessor:
    def __init__(self, config_manager: ConfigManager, max_buffer_length: int = 1000):
        self.config_manager = config_manager
        
        # Initialize indicator processor
        self.indicator_processor = VectorizedIndicatorProcessor()
        
        # Buffers to store incoming klines
        self.kline_buffers: Dict[str, Dict[str, deque]] = {}
        self.MAX_KLINE_BUFFER_LENGTH = max_buffer_length
        
        # Storage for indicator data
        self.indicator_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Track last processed timestamp for each symbol and interval
        self.last_processed: Dict[str, Dict[str, int]] = {}
        
        # Buffer for parallel processing
        self.pending_klines: Dict[str, Dict[str, List[Kline]]] = {}
        self.batch_size = 20  # Process in batches for better performance
        
        self.config_manager.register_callback(self._handle_config_update) # Listen for config changes
        self._active_pairs_timeframes: Dict[str, List[str]] = {}
        self._initialize_buffers_from_config()

    def _handle_config_update(self, new_config: Dict):
        logger.info("DataProcessor: Configuration updated. Re-initializing buffers and settings.")
        self._initialize_buffers_from_config(new_config)

    def _initialize_buffers_from_config(self, config: Optional[Dict] = None):
        if config is None:
            config = self.config_manager.get_config()
        
        new_active_pairs_timeframes: Dict[str, List[str]] = {}
        pairs_config = config.get("pairs", {})
        global_settings = config.get("global_settings", {})
        default_timeframes = global_settings.get("v1_strategy", {}).get("indicator_timeframes", ["1m"])

        for pair_symbol_config, pair_details in pairs_config.items():
            if pair_details.get("enabled", False):
                api_symbol = convert_symbol_to_api_format(pair_symbol_config)
                timeframes_for_pair = pair_details.get("indicator_timeframes", default_timeframes)
                new_active_pairs_timeframes[api_symbol] = timeframes_for_pair
                # Ensure buffers exist for newly active pairs/timeframes
                for tf in timeframes_for_pair:
                    if tf not in self.kline_buffers[api_symbol]:
                        self.kline_buffers[api_symbol][tf] = deque(maxlen=MAX_KLINE_BUFFER_LENGTH)
                    if tf not in self.indicator_data[api_symbol]:
                        self.indicator_data[api_symbol][tf] = pd.DataFrame()
        
        # Cleanup buffers for pairs/timeframes that are no longer active (optional, to save memory)
        # For simplicity, we are not removing old buffers now, but this could be added.
        self._active_pairs_timeframes = new_active_pairs_timeframes
        logger.debug(f"DataProcessor initialized/updated for pairs and timeframes: {self._active_pairs_timeframes}")

    async def process_kline(self, kline: Kline, market_type: Optional[str] = None):
        """ Process an incoming Kline object. market_type is for context if needed. """
        api_symbol = kline.symbol # Assuming kline.symbol is already in API format (e.g., BTCUSDT)
        interval = kline.interval

        if api_symbol not in self._active_pairs_timeframes or interval not in self._active_pairs_timeframes.get(api_symbol, []):
            # logger.debug(f"Skipping kline for inactive pair/timeframe: {api_symbol} {interval}")
            return

        # Initialize buffers if needed
        if api_symbol not in self.kline_buffers:
            self.kline_buffers[api_symbol] = {}
            self.indicator_data[api_symbol] = {}
            self.last_processed[api_symbol] = {}
            self.pending_klines[api_symbol] = {}
            
        if interval not in self.kline_buffers[api_symbol]:
            self.kline_buffers[api_symbol][interval] = deque(maxlen=self.MAX_KLINE_BUFFER_LENGTH)
            self.indicator_data[api_symbol][interval] = pd.DataFrame()
            self.last_processed[api_symbol][interval] = 0
            self.pending_klines[api_symbol][interval] = []
            
        # Add kline to buffer
        self.kline_buffers[api_symbol][interval].append(kline)
        
        # Add to pending klines for batch processing
        self.pending_klines[api_symbol][interval].append(kline)
        
        # Process in batches for better performance
        if len(self.pending_klines[api_symbol][interval]) >= self.batch_size or kline.is_closed:
            await self._process_pending_klines(api_symbol, interval)

    async def _process_pending_klines(self, api_symbol: str, interval: str):
        """Process batches of klines for better performance"""
        pending = self.pending_klines[api_symbol][interval]
        if not pending:
            return
            
        # Prepare data for updating indicators
        for kline in pending:
            self._update_dataframe(api_symbol, interval, kline)
            
        # Update last processed timestamp
        self.last_processed[api_symbol][interval] = pending[-1].timestamp
        
        # Clear pending klines
        self.pending_klines[api_symbol][interval] = []
    
    def _update_dataframe(self, api_symbol: str, interval: str, kline: Kline):
        """Update dataframe with new kline data using optimized approach"""
        df = self.indicator_data[api_symbol][interval]
        
        # Check if kline timestamp already exists in dataframe
        timestamp_exists = False
        if not df.empty and 'timestamp' in df.columns:
            timestamp_exists = kline.timestamp in df['timestamp'].values
            
        if timestamp_exists:
            # Update existing row
            idx = df.index[df['timestamp'] == kline.timestamp][0]
            df.at[idx, 'open'] = kline.open
            df.at[idx, 'high'] = kline.high
            df.at[idx, 'low'] = kline.low
            df.at[idx, 'close'] = kline.close
            df.at[idx, 'volume'] = kline.volume
            df.at[idx, 'is_closed'] = kline.is_closed
            
            # Update indicators incrementally only if candle is closed
            if kline.is_closed:
                self.indicator_processor.update_indicators(df, idx)
        else:
            # Create new row
            new_row = {
                'timestamp': kline.timestamp,
                'open': kline.open,
                'high': kline.high,
                'low': kline.low,
                'close': kline.close,
                'volume': kline.volume,
                'is_closed': kline.is_closed
            }
            
            # Add to dataframe
            df_new_row = pd.DataFrame([new_row])
            if df.empty:
                self.indicator_data[api_symbol][interval] = df_new_row
            else:
                # Append and sort by timestamp
                self.indicator_data[api_symbol][interval] = pd.concat([df, df_new_row], ignore_index=True)
                self.indicator_data[api_symbol][interval] = self.indicator_data[api_symbol][interval].sort_values('timestamp').reset_index(drop=True)
            
            # If we have enough data and candle is closed, calculate indicators
            if len(self.indicator_data[api_symbol][interval]) > 1 and kline.is_closed:
                self._update_indicators_batch(api_symbol, interval)
    
    def _update_indicators_batch(self, api_symbol: str, interval: str):
        """Update all indicators in a single batch operation for efficiency"""
        df = self.indicator_data[api_symbol][interval]
        
        # Only process if we have data
        if df.empty:
            return
            
        # Check if we already have indicator columns
        has_indicators = 'sma_short' in df.columns and 'sma_long' in df.columns
        
        # If we don't have indicators yet, do a full batch calculation
        if not has_indicators:
            self.indicator_data[api_symbol][interval] = self.indicator_processor.batch_process_indicators(df)
        else:
            # For existing data, process only the new row
            latest_idx = len(df) - 1
            self.indicator_processor.update_indicators(df, latest_idx)
            
    def _update_indicators(self, api_symbol: str, interval: str):
        """Legacy method for backward compatibility - delegates to batch processing"""
        kline_deque = self.kline_buffers[api_symbol][interval]
        if not kline_deque:
            return
            
        # Convert deque to DataFrame if needed
        if api_symbol not in self.indicator_data or interval not in self.indicator_data[api_symbol]:
            self.indicator_data[api_symbol][interval] = pd.DataFrame()
            
        # Check if we need to initialize the DataFrame
        df = self.indicator_data[api_symbol][interval]
        if df.empty:
            # Initialize with basic OHLCV data
            df = pd.DataFrame({
                "timestamp": [k.timestamp for k in kline_deque],
                "open": [k.open for k in kline_deque],
                "high": [k.high for k in kline_deque],
                "low": [k.low for k in kline_deque],
                "close": [k.close for k in kline_deque],
                "volume": [k.volume for k in kline_deque],
                "is_closed": [k.is_closed for k in kline_deque]
            })
            self.indicator_data[api_symbol][interval] = df
            
            # Calculate all indicators in batch
            self.indicator_data[api_symbol][interval] = self.indicator_processor.batch_process_indicators(df)
        else:
            # Update the last candle with optimized indicators
            latest_idx = len(df) - 1
            self.indicator_processor.update_indicators(df, latest_idx)
            
    async def get_indicator_data(self, api_symbol: str, interval: str) -> pd.DataFrame:
        """Get indicator data for a specific symbol and interval"""
        if api_symbol not in self.indicator_data or interval not in self.indicator_data[api_symbol]:
            # Initialize empty DataFrames
            if api_symbol not in self.indicator_data:
                self.indicator_data[api_symbol] = {}
            self.indicator_data[api_symbol][interval] = pd.DataFrame()
            
        return self.indicator_data[api_symbol][interval]
        
    async def get_latest_kline(self, api_symbol: str, interval: str) -> Optional[Kline]:
        """Get the latest kline for a specific symbol and interval"""
        if api_symbol not in self.kline_buffers or interval not in self.kline_buffers[api_symbol]:
            return None
            
        buffer = self.kline_buffers[api_symbol][interval]
        if not buffer:
            return None
            
        return buffer[-1]

    def get_latest_kline(self, api_symbol: str, interval: str) -> Optional[Kline]:
        if api_symbol in self.kline_buffers and interval in self.kline_buffers[api_symbol]:
            if self.kline_buffers[api_symbol][interval]:
                return self.kline_buffers[api_symbol][interval][-1]
        return None

    def get_indicator_dataframe(self, api_symbol: str, interval: str) -> Optional[pd.DataFrame]:
        if api_symbol in self.indicator_data and interval in self.indicator_data[api_symbol]:
            return self.indicator_data[api_symbol][interval].copy() # Return a copy
        return None

    def get_latest_indicators(self, api_symbol: str, interval: str) -> Optional[pd.Series]:
        df = self.get_indicator_dataframe(api_symbol, interval)
        if df is not None and not df.empty:
            return df.iloc[-1]
        return None

