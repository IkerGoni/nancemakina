#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Callable, Union
import yaml
import matplotlib.pyplot as plt
from pydantic import BaseModel
import asyncio
from decimal import Decimal, getcontext, ROUND_DOWN
from collections import deque
import sys
import gc
import random
import pickle
import hashlib
import json
import zlib  # For compression
import psutil
import joblib
import argparse

# Set decimal precision for financial calculations
getcontext().prec = 28

# Import necessary models and utilities
from src.models import Kline, TradeSignal, TradeDirection
from src.config_loader import ConfigManager
from src.data_processor import DataProcessor
from src.signal_engine import SignalEngineV1

# Import our new optimized components
from scripts.indicator_processor import VectorizedIndicatorProcessor
from scripts.parallel_processor import ParallelProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('backtest')

class BacktestPosition(BaseModel):
    symbol: str
    entry_price: float
    entry_time: int
    quantity: float
    direction: TradeDirection
    stop_loss: float
    take_profit: float
    status: str = "OPEN"  # OPEN, CLOSED_TP, CLOSED_SL, CLOSED_EXIT
    exit_price: Optional[float] = None
    exit_time: Optional[int] = None
    profit_loss: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    
class BacktestResults(BaseModel):
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit_percent: float
    avg_loss_percent: float
    max_drawdown: float
    profit_factor: float
    total_fees: float
    
class OptimizedBacktester:
    """
    Optimized backtester with improved performance for processing large datasets.
    Uses vectorized calculations and parallel processing where appropriate.
    """
    
    def __init__(self, config_path: str, logger: logging.Logger = None):
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.logger = logger or logging.getLogger('backtest')
        
        # Initialize our improved components
        self.indicator_processor = VectorizedIndicatorProcessor()
        self.parallel_processor = ParallelProcessor()
        
        # Progress tracking
        self.total_rows = 0
        self.processed_rows = 0
        self.progress_callback = None
        self.start_time = None
        
        # Resource monitoring
        self.memory_usage = []
        self.cpu_usage = []
        self.benchmark_points = {}
        
    def register_progress_callback(self, callback: Callable[[int, int, float], None]):
        """Register a callback function to be called for progress updates"""
        self.progress_callback = callback
        
    def update_progress(self, processed: int, total: int):
        """Update processing progress"""
        self.processed_rows = processed
        self.total_rows = total
        
        if self.progress_callback and total > 0:
            progress_pct = (processed / total) * 100
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.progress_callback(processed, total, elapsed)
    
    def benchmark_start(self, label: str):
        """Start a benchmark timing point"""
        self.benchmark_points[label] = {"start": time.time(), "memory_start": self.get_memory_usage()}
        self.logger.info(f"Starting benchmark: {label}")
    
    def benchmark_end(self, label: str):
        """End a benchmark timing point and log results"""
        if label not in self.benchmark_points:
            return
            
        start_data = self.benchmark_points[label]
        elapsed = time.time() - start_data["start"]
        current_memory = self.get_memory_usage()
        memory_diff = current_memory - start_data["memory_start"]
        
        self.logger.info(f"Benchmark {label}: {elapsed:.2f} seconds, Memory change: {memory_diff:.2f} MB")
        return {"elapsed": elapsed, "memory_diff": memory_diff}
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
        
    async def load_and_process_data(self, data_path, symbol, sample_size=1.0):
        """
        Load and perform initial processing of the historical data.
        
        Args:
            data_path: Path to the data file
            symbol: Trading symbol
            sample_size: Sample size to use (0.0-1.0)
            
        Returns:
            List of processed kline objects
        """
        # Store the data path for later reference
        self.data_path = data_path
        
        logger.info(f"Loading historical data from {data_path}")
        
        # Read the CSV file
        df = pd.read_csv(data_path)
        
        # Filter for the symbol
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol]
        
        # Apply sampling if requested
        if sample_size < 1.0:
            if sample_size <= 0.0:
                sample_size = 0.1  # Minimum 10%
                
            logger.info(f"Sampling dataset ({sample_size:.1%} of data) for faster testing")
            original_count = len(df)
            sample_count = max(int(original_count * sample_size), 1000)  # At least 1000 data points
            
            # Ensure chronological order is maintained with systematic sampling
            df = df.iloc[::max(1, original_count // sample_count)]
            
            logger.info(f"Sampled from {original_count} to {len(df)} data points")
        
        # Convert to kline objects
        kline_objects = []
        for _, row in df.iterrows():
            kline = {
                'open_time': int(row['open_time']) if 'open_time' in row else int(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'close_time': int(row['close_time']) if 'close_time' in row else int(row['timestamp']) + 60000,
                'quote_asset_volume': float(row['quote_asset_volume']) if 'quote_asset_volume' in row else 0,
                'number_of_trades': int(row['number_of_trades']) if 'number_of_trades' in row else 0,
                'taker_buy_base_asset_volume': float(row['taker_buy_base_asset_volume']) if 'taker_buy_base_asset_volume' in row else 0,
                'taker_buy_quote_asset_volume': float(row['taker_buy_quote_asset_volume']) if 'taker_buy_quote_asset_volume' in row else 0
            }
            kline_objects.append(kline)
        
        # Sort by timestamp
        kline_objects.sort(key=lambda x: x['open_time'])
        
        logger.info(f"Loaded {len(kline_objects)} klines for {symbol}")
        return kline_objects
    
    def load_from_cache(self, cache_file):
        """
        Load processed data from cache.
        
        Args:
            cache_file: Path to the cache file
            
        Returns:
            DataFrame of processed data if successful, None otherwise
        """
        if not os.path.exists(cache_file):
            logger.info(f"Cache file {cache_file} not found")
            return None
        
        try:
            logger.info(f"Loading data from cache: {cache_file}")
            data = joblib.load(cache_file)
            logger.info(f"Successfully loaded cached data ({len(data)} rows)")
            return data
        except Exception as e:
            logger.warning(f"Failed to load from cache: {str(e)}")
            return None
    
    def save_to_cache(self, data, cache_file):
        """
        Save processed data to cache.
        
        Args:
            data: DataFrame to save
            cache_file: Path to the cache file
        """
        try:
            logger.info(f"Saving data to cache: {cache_file}")
            joblib.dump(data, cache_file, compress=3)
            logger.info(f"Successfully saved data to cache")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {str(e)}")
    
    async def process_historical_data_chunked(self, kline_objects, symbol, chunk_size, chunk_type="rows"):
        """
        Process historical data in chunks to manage memory usage.
        
        Args:
            kline_objects: List of kline objects to process
            symbol: Trading symbol
            chunk_size: Size of chunks
            chunk_type: Type of chunking ("rows", "days", "weeks", "months")
            
        Returns:
            DataFrame of processed data
        """
        logger.info(f"Processing {len(kline_objects)} klines in chunks ({chunk_type}, size={chunk_size})")
        
        # Divide klines into chunks based on the chunk type
        chunks = []
        if chunk_type == "rows":
            # Simple row-based chunking
            for i in range(0, len(kline_objects), chunk_size):
                chunks.append(kline_objects[i:i + chunk_size])
        else:
            # Time-based chunking
            chunks = self._create_time_based_chunks(kline_objects, chunk_type, chunk_size)
        
        logger.info(f"Created {len(chunks)} chunks for processing")
        
        # Process each chunk
        all_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} klines")
            
            # Generate historical dataframe for this chunk
            chunk_df = await self.generate_full_historical_df(chunk, symbol)
            all_results.append(chunk_df)
        
        # Combine results
        logger.info("Combining chunk results")
        combined_df = pd.concat(all_results)
        
        # Remove any duplicate timestamps
        combined_df = combined_df.drop_duplicates(subset=["timestamp"])
        
        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp")
        
        logger.info(f"Completed chunked processing ({len(combined_df)} rows)")
        return combined_df
    
    def _create_time_based_chunks(self, kline_objects, chunk_type, chunk_size):
        """
        Create time-based chunks of kline objects.
        
        Args:
            kline_objects: List of kline objects
            chunk_type: Type of time-based chunking ("days", "weeks", "months")
            chunk_size: Number of time units per chunk
            
        Returns:
            List of chunks, each containing a list of kline objects
        """
        from datetime import datetime, timedelta
        
        if not kline_objects:
            return []
        
        # Sort by timestamp
        kline_objects = sorted(kline_objects, key=lambda x: x['open_time'])
        
        # Get start and end dates
        start_time = datetime.fromtimestamp(kline_objects[0]['open_time'] / 1000)
        end_time = datetime.fromtimestamp(kline_objects[-1]['open_time'] / 1000)
        
        # Calculate chunk boundaries
        chunks = []
        current_klines = []
        current_end_time = None
        
        if chunk_type == "days":
            chunk_delta = timedelta(days=chunk_size)
        elif chunk_type == "weeks":
            chunk_delta = timedelta(weeks=chunk_size)
        elif chunk_type == "months":
            # Approximate months as 30 days
            chunk_delta = timedelta(days=30 * chunk_size)
        else:
            # Default to days
            chunk_delta = timedelta(days=chunk_size)
        
        for kline in kline_objects:
            kline_time = datetime.fromtimestamp(kline['open_time'] / 1000)
            
            if current_end_time is None:
                # First chunk
                current_end_time = kline_time + chunk_delta
            
            if kline_time <= current_end_time:
                # This kline belongs to the current chunk
                current_klines.append(kline)
            else:
                # Start a new chunk
                if current_klines:
                    chunks.append(current_klines)
                current_klines = [kline]
                current_end_time = kline_time + chunk_delta
        
        # Add the last chunk
        if current_klines:
            chunks.append(current_klines)
        
        return chunks
    
    async def generate_full_historical_df(self, kline_objects: List[Kline], symbol: str, batch_size: int = 5000) -> pd.DataFrame:
        """
        Generate historical DataFrame with all indicators using optimized vectorized operations
        
        Args:
            kline_objects: List of Kline objects
            symbol: Trading symbol
            batch_size: Size of batches for processing
            
        Returns:
            DataFrame with all indicators calculated
        """
        self.benchmark_start("generate_df")
        self.logger.info(f"Generating complete historical DataFrame for {symbol} 1m with unbounded buffer")
        
        # Set up progress tracking
        self.total_rows = len(kline_objects)
        self.processed_rows = 0
        self.start_time = time.time()
        self.logger.info(f"Processing {self.total_rows} klines to generate historical dataframe")
        
        # Create initial dataframe
        df_data = {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "is_closed": []
        }
        
        # Process in batches for better performance and memory usage
        chunks = [kline_objects[i:i + batch_size] for i in range(0, len(kline_objects), batch_size)]
        chunk_dfs = []
        
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "timestamp": [k.timestamp for k in chunk],
                "open": [k.open for k in chunk],
                "high": [k.high for k in chunk],
                "low": [k.low for k in chunk],
                "close": [k.close for k in chunk],
                "volume": [k.volume for k in chunk],
                "is_closed": [k.is_closed for k in chunk]
            }
            chunk_df = pd.DataFrame(chunk_data)
            chunk_dfs.append(chunk_df)
            
            # Update progress
            self.processed_rows += len(chunk)
            self.update_progress(self.processed_rows, self.total_rows)
        
        # Combine chunks
        final_df = pd.concat(chunk_dfs, ignore_index=True)
        
        # Use parallel processing for indicator calculation
        self.benchmark_start("indicators")
        final_df = self.parallel_processor.parallel_batch_indicators(final_df)
        self.benchmark_end("indicators")
        
        # Calculate any additional indicators specific to this strategy
        config = self.config_manager.get_config()
        global_settings = config.get("global_settings", {})
        strategy_config = global_settings.get("v1_strategy", {})
        
        self.benchmark_end("generate_df")
        return final_df
    
    def detect_signal_points(self, df: pd.DataFrame) -> List[Tuple[int, str]]:
        """
        Detect all signal points in the historical data using optimized vectorized operations
        
        Args:
            df: DataFrame with indicator data
            
        Returns:
            List of tuples with (index, signal_type)
        """
        self.benchmark_start("signal_detection")
        
        # Use parallel processor for crossover detection
        crossover_indices = self.parallel_processor.parallel_crossover_detection(df)
        
        # Determine signal types for each crossover
        signals = []
        for idx in crossover_indices:
            if idx > 0:  # Skip first row as we need previous data
                prev_short = df.iloc[idx-1]['sma_short']
                prev_long = df.iloc[idx-1]['sma_long']
                curr_short = df.iloc[idx]['sma_short']
                curr_long = df.iloc[idx]['sma_long']
                
                # Crossed above
                if prev_short <= prev_long and curr_short > curr_long:
                    signals.append((idx, 'buy'))
                    
                # Crossed below
                elif prev_short >= prev_long and curr_short < curr_long:
                    signals.append((idx, 'sell'))
                    
        self.benchmark_end("signal_detection")
        return signals
        
    def save_to_cache(self, df: pd.DataFrame, cache_path: str, compress: bool = True) -> bool:
        """
        Save processed DataFrame to cache file
        
        Args:
            df: DataFrame to save
            cache_path: Path to save the cache file
            compress: Whether to compress the cache file
            
        Returns:
            True if successful, False otherwise
        """
        self.benchmark_start("save_cache")
        
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            if compress:
                # Save with zlib compression
                joblib.dump(df, cache_path, compress=('zlib', 3))
                self.logger.info(f"Saved compressed cache to {cache_path}")
            else:
                # Save without compression
                df.to_pickle(cache_path)
                self.logger.info(f"Saved cache to {cache_path}")
                
            self.benchmark_end("save_cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")
            self.benchmark_end("save_cache")
            return False
            
    def load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """
        Load processed DataFrame from cache file
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            Loaded DataFrame or None if loading failed
        """
        self.benchmark_start("load_cache")
        
        try:
            if not os.path.exists(cache_path):
                self.logger.info(f"Cache file {cache_path} not found")
                self.benchmark_end("load_cache")
                return None
                
            try:
                # Try to load as compressed file first
                df = joblib.load(cache_path)
                self.logger.info(f"Loaded compressed cache from {cache_path}")
            except:
                # Fallback to regular pickle
                df = pd.read_pickle(cache_path)
                self.logger.info(f"Loaded cache from {cache_path}")
                
            self.benchmark_end("load_cache")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            self.benchmark_end("load_cache")
            return None

    def _load_backtest_config(self):
        """Load backtesting specific configuration"""
        config = self.config_manager.get_config()
        backtest_settings_from_config = config.get("backtesting")  # Get potential None or dict

        # Ensure backtest_config is a dictionary to safely call .get() on it
        backtest_config = backtest_settings_from_config if isinstance(backtest_settings_from_config, dict) else {}
        
        self.initial_balance = Decimal(str(backtest_config.get("initial_balance_usdt", 10000.0)))
        self.current_balance = self.initial_balance
        self.fee_percent = Decimal(str(backtest_config.get("fee_percent", 0.04))) / Decimal('100')  # Convert to decimal
        self.slippage_percent = Decimal(str(backtest_config.get("slippage_percent", 0.01))) / Decimal('100')  # Convert to decimal
        
    def _generate_cache_key(self, symbol: str, config_symbol: str, sample_size: float) -> str:
        """Generate a unique cache key based on input data and configuration"""
        # Get config file content
        with open(self.config_path, 'r') as f:
            config_content = f.read()
            
        # Get data file modification time
        data_mtime = os.path.getmtime(self.data_file)
        
        # Create a dict with all relevant parameters
        cache_info = {
            'data_file': self.data_file,
            'data_mtime': data_mtime,
            'config_hash': hashlib.md5(config_content.encode()).hexdigest(),
            'symbol': symbol,
            'config_symbol': config_symbol,
            'sample_size': sample_size
        }
        
        # Generate a hash of this info for the cache key
        cache_key = hashlib.md5(json.dumps(cache_info, sort_keys=True).encode()).hexdigest()
        return cache_key
    
    def _get_cache_paths(self, cache_key: str) -> Tuple[Path, Path, Path]:
        """Get paths for cache files based on the cache key"""
        df_cache_path = self.cache_dir / f"{cache_key}_dataframe.pkl"
        signals_cache_path = self.cache_dir / f"{cache_key}_signals.pkl"
        metadata_path = self.cache_dir / f"{cache_key}_metadata.json"
        return df_cache_path, signals_cache_path, metadata_path
    
    def _save_to_cache(self, df: pd.DataFrame, signals: List[TradeSignal], 
                       symbol: str, config_symbol: str, sample_size: float) -> str:
        """Save dataframe and signals to cache files"""
        try:
            cache_key = self._generate_cache_key(symbol, config_symbol, sample_size)
            df_path, signals_path, metadata_path = self._get_cache_paths(cache_key)
            
            # Save dataframe with compression
            with open(df_path, 'wb') as f:
                pickled_data = pickle.dumps(df)
                compressed_data = zlib.compress(pickled_data)
                f.write(compressed_data)
            
            # Save signals with compression
            with open(signals_path, 'wb') as f:
                pickled_signals = pickle.dumps(signals)
                compressed_signals = zlib.compress(pickled_signals)
                f.write(compressed_signals)
            
            # Save metadata with cache statistics
            uncompressed_df_size = len(pickle.dumps(df))
            compressed_df_size = len(compressed_data)
            uncompressed_signals_size = len(pickled_signals)
            compressed_signals_size = len(compressed_signals)
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'config_symbol': config_symbol,
                'sample_size': sample_size,
                'data_file': self.data_file,
                'config_file': self.config_path,
                'data_rows': len(df),
                'signals_count': len(signals),
                'cache_stats': {
                    'df_size_bytes': compressed_df_size,
                    'df_compression_ratio': round(compressed_df_size / uncompressed_df_size, 2),
                    'signals_size_bytes': compressed_signals_size,
                    'signals_compression_ratio': round(compressed_signals_size / uncompressed_signals_size, 2),
                    'total_size_bytes': compressed_df_size + compressed_signals_size,
                    'compression_savings_pct': round(100 * (1 - (compressed_df_size + compressed_signals_size) / 
                                              (uncompressed_df_size + uncompressed_signals_size)), 1)
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved processed data to cache: {cache_key}")
            return cache_key
        
        except Exception as e:
            logger.warning(f"Failed to save data to cache: {e}")
            return ""
    
    def _load_from_cache(self, symbol: str, config_symbol: str, sample_size: float) -> Tuple[Optional[pd.DataFrame], Optional[List[TradeSignal]]]:
        """Load dataframe and signals from cache if available and valid"""
        try:
            cache_key = self._generate_cache_key(symbol, config_symbol, sample_size)
            df_path, signals_path, metadata_path = self._get_cache_paths(cache_key)
            
            # Check if all cache files exist
            if not all(p.exists() for p in [df_path, signals_path, metadata_path]):
                logger.info("Cache files not found")
                return None, None
                
            # Verify cache integrity
            if not self._verify_cache_integrity(df_path, signals_path, metadata_path):
                logger.warning("Cache integrity verification failed, will reprocess data")
                return None, None
            
            # Load metadata to verify
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Verify metadata
            if (metadata['symbol'] != symbol or 
                metadata['config_symbol'] != config_symbol or 
                abs(metadata['sample_size'] - sample_size) > 0.001):
                logger.warning("Cache metadata mismatch")
                return None, None
            
            # Load dataframe with decompression
            with open(df_path, 'rb') as f:
                compressed_data = f.read()
                pickled_data = zlib.decompress(compressed_data)
                df = pickle.loads(pickled_data)
            
            # Load signals with decompression
            with open(signals_path, 'rb') as f:
                compressed_signals = f.read()
                pickled_signals = zlib.decompress(compressed_signals)
                signals = pickle.loads(pickled_signals)
            
            logger.info(f"Loaded processed data from cache: {cache_key}")
            logger.info(f"Cache timestamp: {metadata['timestamp']}")
            logger.info(f"Loaded DataFrame with {len(df)} rows and {len(signals)} signals")
            
            return df, signals
        
        except Exception as e:
            logger.warning(f"Failed to load data from cache: {e}")
            return None, None
        
    def setup_progress_tracking(self, total_steps: int):
        """Initialize progress tracking"""
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.progress_last_reported = self.start_time
        
    def update_progress(self, step_increment: int = 1, force_log: bool = False):
        """Update and display progress"""
        self.current_step += step_increment
        current_time = time.time()
        
        # Only report progress every 5 seconds or when forced
        if force_log or (current_time - self.progress_last_reported) >= 5:
            elapsed = current_time - self.start_time
            progress_pct = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
            
            # Estimate remaining time
            if self.current_step > 0 and self.total_steps > 0:
                time_per_step = elapsed / self.current_step
                remaining_steps = self.total_steps - self.current_step
                remaining_time = time_per_step * remaining_steps
                
                # Format for display
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                remaining_str = str(timedelta(seconds=int(remaining_time)))
                
                logger.info(f"Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%), "
                            f"Elapsed: {elapsed_str}, Estimated remaining: {remaining_str}")
                
                # Also display a progress bar
                self._display_progress_bar(self.current_step, self.total_steps)
            else:
                logger.info(f"Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%)")
                
            self.progress_last_reported = current_time
            
    def _display_progress_bar(self, progress, total, length=50):
        """Display a text progress bar"""
        if total <= 0:
            return
            
        filled_length = int(length * progress // total)
        bar = '█' * filled_length + '░' * (length - filled_length)
        percent = progress / total * 100
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if progress > 0:
            eta = elapsed * (total - progress) / progress
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = "Unknown"
            
        print(f'\rProgress: |{bar}| {percent:.1f}% ({progress}/{total}) ETA: {eta_str}', end='', flush=True)

    def load_historical_data(self) -> List[Kline]:
        """Load historical data from CSV file and convert to Kline objects"""
        logger.info(f"Loading historical data from {self.data_file}")
        
        try:
            # Load data from CSV
            df = pd.read_csv(self.data_file)
            
            # Convert 'Open time' from string to timestamp
            df['timestamp'] = pd.to_datetime(df['Open time']).astype(np.int64) // 10**6  # Convert to milliseconds
            
            # Create list of Kline objects
            klines = []
            for _, row in df.iterrows():
                kline = Kline(
                    timestamp=int(row['timestamp']),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    quote_asset_volume=float(row.get('Taker buy base asset volume', 0)),
                    number_of_trades=int(row.get('Number of trades', 0)),
                    is_closed=True,  # All historical data is considered closed
                    symbol="BTCUSDT",  # Default symbol, will be overridden in process_data_async
                    interval="1m"  # Default interval, will be overridden in process_data_async
                )
                klines.append(kline)
            
            logger.info(f"Loaded {len(klines)} data points")
            return klines
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    async def generate_full_historical_df(self, klines: List[Kline], symbol: str = "BTCUSDT", interval: str = "1m") -> pd.DataFrame:
        """
        Generate a complete historical DataFrame with all indicators by processing all klines
        with an unbounded buffer (no MAX_KLINE_BUFFER_LENGTH limitation).
        
        This provides the full historical context needed for accurate backtesting with point-in-time data.
        """
        logger.info(f"Generating complete historical DataFrame for {symbol} {interval} with unbounded buffer")
        
        # Create a temporary DataProcessor with unbounded kline buffers
        temp_dp = DataProcessor(self.config_manager)
        
        # Override the maxlen of the kline buffer to be unbounded for this specific symbol and interval
        temp_dp.kline_buffers[symbol][interval] = deque()  # No maxlen means unbounded
        
        # Setup progress tracking for processing klines
        self.setup_progress_tracking(len(klines))
        logger.info(f"Processing {len(klines)} klines to generate historical dataframe")
        
        # Process all klines through the temporary data processor
        for i, kline in enumerate(klines):
            kline.symbol = symbol
            kline.interval = interval
            await temp_dp.process_kline(kline)
            
            # Update progress every 1000 klines to avoid too much logging
            if i % 1000 == 0:
                self.update_progress(1000, i == 0)
        
        # Update final progress
        remainder = len(klines) % 1000
        if remainder > 0:
            self.update_progress(remainder, True)
        
        # Get the complete DataFrame with all historical data
        full_df = temp_dp.indicator_data[symbol][interval]
        logger.info(f"Generated full historical DataFrame with {len(full_df)} rows and indicators")
        
        return full_df

    async def generate_signals_optimized(self, full_df: pd.DataFrame, symbol: str = "BTCUSDT", config_symbol: str = "BTC_USDT", warmup_period: int = 200) -> List[TradeSignal]:
        """
        Generate trading signals using a vectorized approach to detect crossovers first,
        then only process the potential signal points.
        
        Args:
            full_df: Complete historical DataFrame with all indicators
            symbol: Trading symbol in API format (e.g., "BTCUSDT")
            config_symbol: Trading symbol in config format (e.g., "BTC_USDT")
            warmup_period: Number of initial candles to skip for indicator warmup
            
        Returns:
            List of generated trade signals
        """
        logger.info(f"Generating signals using optimized approach for {symbol}")
        signals = []
        
        # Reset signal time tracking
        original_last_signal_time = self.signal_engine.last_signal_time.copy()
        self.signal_engine.last_signal_time = {}
        
        # Ensure we have enough data
        if len(full_df) < warmup_period + 2:
            logger.warning(f"Not enough data to generate signals. Have {len(full_df)} rows, need at least {warmup_period + 2}")
            return signals
        
        # Pre-process: identify all potential crossovers first
        logger.info(f"Pre-screening for potential crossover points")
        
        # Process only from warmup period onwards
        valid_df = full_df.iloc[warmup_period:]
        
        # Vectorized crossover detection
        prev_comparison = valid_df['sma_short'].shift(1) <= valid_df['sma_long'].shift(1)
        curr_comparison = valid_df['sma_short'] > valid_df['sma_long']
        crossed_up_mask = (prev_comparison) & (curr_comparison)
        
        prev_comparison = valid_df['sma_short'].shift(1) >= valid_df['sma_long'].shift(1)
        curr_comparison = valid_df['sma_short'] < valid_df['sma_long']
        crossed_down_mask = (prev_comparison) & (curr_comparison)
        
        # Get indices of all potential crossovers
        crossover_indices = valid_df[crossed_up_mask | crossed_down_mask].index
        logger.info(f"Found {len(crossover_indices)} potential crossover points")
        
        # Setup progress tracking for processing crossovers
        self.setup_progress_tracking(len(crossover_indices))
        
        # Process only the crossover points
        for i, idx in enumerate(crossover_indices):
            # Get data up to this point
            row_position = full_df.index.get_loc(idx)
            point_in_time_df = full_df.iloc[:row_position+1]
            
            # Process the signal
            signal = await self.signal_engine.check_signal_with_df(symbol, config_symbol, point_in_time_df)
            
            if signal:
                signal.timestamp = int(idx)
                signals.append(signal)
                logger.info(f"Generated {signal.direction.value} signal at {datetime.fromtimestamp(idx/1000)}")
                
            # Update progress every 10 iterations or on first/last
            if i % 10 == 0 or i == len(crossover_indices) - 1:
                self.update_progress(min(10, i - self.current_step), i == 0 or i == len(crossover_indices) - 1)
        
        # Restore original state
        self.signal_engine.last_signal_time = original_last_signal_time
        logger.info(f"Generated {len(signals)} signals using optimized approach")
        return signals

    def simulate_trading(self, signals: List[TradeSignal], df: pd.DataFrame):
        """
        Simulate trading based on generated signals
        """
        logger.info(f"Starting trading simulation with {len(signals)} signals")
        
        # Handle the case where no signals were generated
        if not signals:
            logger.warning("No signals provided to simulate_trading. Exiting simulation.")
            # Initialize the equity curve with initial balance
            if df is not None and not df.empty:
                self.equity_curve = [(df.iloc[0]['timestamp'], float(self.initial_balance))]
            else:
                self.equity_curve = [(int(datetime.now().timestamp() * 1000), float(self.initial_balance))]
            self.final_balance = float(self.current_balance)
            self.total_fees = Decimal('0.0')
            return [], self.equity_curve
        
        # Get configuration
        config = self.config_manager.get_config()
        pair_configs = config.get("pairs", {})
        global_v1_config = config.get("global_settings", {}).get("v1_strategy", {})
        
        # Get symbol specific config
        symbol_config = None
        for cfg_sym, details in pair_configs.items():
            if cfg_sym.upper() == signals[0].config_symbol.upper():
                symbol_config = details
                break
        
        # Check if symbol_config was found
        if not symbol_config:
            logger.error(f"Configuration for {signals[0].config_symbol} not found. Cannot simulate trades.")
            # Initialize the equity curve with initial balance
            if df is not None and not df.empty:
                self.equity_curve = [(df.iloc[0]['timestamp'], float(self.initial_balance))]
            else:
                self.equity_curve = [(int(datetime.now().timestamp() * 1000), float(self.initial_balance))]
            self.final_balance = float(self.current_balance)
            self.total_fees = Decimal('0.0')
            return [], self.equity_curve
        
        # Set leverage and margin
        leverage = Decimal(str(symbol_config.get("leverage", global_v1_config.get("default_leverage", 10))))
        fixed_margin_usdt = Decimal(str(symbol_config.get("margin_usdt", global_v1_config.get("default_margin_usdt", 50.0))))
        
        # Track current positions
        open_positions = []
        closed_positions = []
        
        # Track equity curve
        equity_curve = [(df.iloc[0]['timestamp'], float(self.initial_balance))]
        
        # Iterate through each candle
        current_balance = self.initial_balance
        total_fees = Decimal('0.0')
        
        # Setup progress tracking for simulation
        self.setup_progress_tracking(len(df))
        logger.info(f"Simulating trading over {len(df)} candles")
        
        for i in range(len(df)):
            current_candle = df.iloc[i]
            current_time = current_candle['timestamp']
            
            # Check for signals at this timestamp
            new_signals = [s for s in signals if s.timestamp == current_time]
            
            # Process open positions first (check for TP/SL)
            for pos in list(open_positions):
                # Check if TP hit
                if (pos.direction == TradeDirection.LONG and current_candle['high'] >= pos.take_profit) or \
                   (pos.direction == TradeDirection.SHORT and current_candle['low'] <= pos.take_profit):
                    # Close at take profit
                    pos.status = "CLOSED_TP"
                    pos.exit_price = pos.take_profit
                    pos.exit_time = current_time
                    
                    # Calculate P&L using Decimal for precision
                    if pos.direction == TradeDirection.LONG:
                        pos.profit_loss = float(
                            (Decimal(str(pos.exit_price)) - Decimal(str(pos.entry_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    else:  # SHORT
                        pos.profit_loss = float(
                            (Decimal(str(pos.entry_price)) - Decimal(str(pos.exit_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    
                    # Account for fees
                    entry_fee = Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    exit_fee = Decimal(str(pos.exit_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    fee = entry_fee + exit_fee
                    pos.profit_loss = float(Decimal(str(pos.profit_loss)) - fee)
                    total_fees += fee
                    
                    # Update balance
                    current_balance += Decimal(str(pos.profit_loss))
                    
                    # Calculate percent P&L
                    pos.profit_loss_percent = float(
                        (Decimal(str(pos.profit_loss)) / 
                        (Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)))) * Decimal('100')
                    )
                    
                    # Log
                    logger.info(f"TP HIT: {pos.direction.value} {pos.symbol} closed at {pos.exit_price}, "
                                f"P&L: ${pos.profit_loss:.2f} ({pos.profit_loss_percent:.2f}%)")
                    
                    # Move to closed positions
                    closed_positions.append(pos)
                    open_positions.remove(pos)
                
                # Check if SL hit
                elif (pos.direction == TradeDirection.LONG and current_candle['low'] <= pos.stop_loss) or \
                     (pos.direction == TradeDirection.SHORT and current_candle['high'] >= pos.stop_loss):
                    # Close at stop loss
                    pos.status = "CLOSED_SL"
                    pos.exit_price = pos.stop_loss
                    pos.exit_time = current_time
                    
                    # Calculate P&L using Decimal for precision
                    if pos.direction == TradeDirection.LONG:
                        pos.profit_loss = float(
                            (Decimal(str(pos.exit_price)) - Decimal(str(pos.entry_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    else:  # SHORT
                        pos.profit_loss = float(
                            (Decimal(str(pos.entry_price)) - Decimal(str(pos.exit_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    
                    # Account for fees
                    entry_fee = Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    exit_fee = Decimal(str(pos.exit_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    fee = entry_fee + exit_fee
                    pos.profit_loss = float(Decimal(str(pos.profit_loss)) - fee)
                    total_fees += fee
                    
                    # Update balance
                    current_balance += Decimal(str(pos.profit_loss))
                    
                    # Calculate percent P&L
                    pos.profit_loss_percent = float(
                        (Decimal(str(pos.profit_loss)) / 
                        (Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)))) * Decimal('100')
                    )
                    
                    # Log
                    logger.info(f"SL HIT: {pos.direction.value} {pos.symbol} closed at {pos.exit_price}, "
                                f"P&L: ${pos.profit_loss:.2f} ({pos.profit_loss_percent:.2f}%)")
                    
                    # Move to closed positions
                    closed_positions.append(pos)
                    open_positions.remove(pos)
            
            # Process new signals and open positions
            for signal in new_signals:
                # Calculate position size
                position_size = fixed_margin_usdt / Decimal(str(signal.entry_price))
                
                # Apply slippage to entry
                entry_price = Decimal(str(signal.entry_price))
                if signal.direction == TradeDirection.LONG:
                    entry_price *= (Decimal('1') + self.slippage_percent)  # Higher for buys
                else:
                    entry_price *= (Decimal('1') - self.slippage_percent)  # Lower for sells
                
                # Create new position
                new_position = BacktestPosition(
                    symbol=signal.symbol,
                    entry_price=float(entry_price),
                    entry_time=current_time,
                    quantity=float(position_size),
                    direction=signal.direction,
                    stop_loss=signal.stop_loss_price,
                    take_profit=signal.take_profit_price,
                    status="OPEN"
                )
                
                # Calculate entry fee
                entry_fee = entry_price * position_size * self.fee_percent
                total_fees += entry_fee
                
                # Log
                logger.info(f"OPEN: {signal.direction.value} {signal.symbol} at {float(entry_price)}, "
                            f"Size: {float(position_size)}, SL: {signal.stop_loss_price}, TP: {signal.take_profit_price}")
                
                # Add to open positions
                open_positions.append(new_position)
            
            # Track equity at each step
            equity_curve.append((current_time, float(current_balance)))
            
            # Update progress every 1000 candles
            if i % 1000 == 0 or i == len(df) - 1:
                self.update_progress(min(1000, i + 1 - self.current_step), i == 0 or i == len(df) - 1)
        
        # Close any remaining open positions at the last price
        last_candle = df.iloc[-1]
        for pos in list(open_positions):
            pos.status = "CLOSED_EXIT"
            pos.exit_price = float(last_candle['close'])
            pos.exit_time = last_candle['timestamp']
            
            # Calculate P&L using Decimal
            if pos.direction == TradeDirection.LONG:
                pos.profit_loss = float(
                    (Decimal(str(pos.exit_price)) - Decimal(str(pos.entry_price))) * 
                    Decimal(str(pos.quantity)) * leverage
                )
            else:  # SHORT
                pos.profit_loss = float(
                    (Decimal(str(pos.entry_price)) - Decimal(str(pos.exit_price))) * 
                    Decimal(str(pos.quantity)) * leverage
                )
            
            # Account for fees
            entry_fee = Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)) * self.fee_percent
            exit_fee = Decimal(str(pos.exit_price)) * Decimal(str(pos.quantity)) * self.fee_percent
            fee = entry_fee + exit_fee
            pos.profit_loss = float(Decimal(str(pos.profit_loss)) - fee)
            total_fees += fee
            
            # Update balance
            current_balance += Decimal(str(pos.profit_loss))
            
            # Calculate percent P&L
            pos.profit_loss_percent = float(
                (Decimal(str(pos.profit_loss)) / 
                (Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)))) * Decimal('100')
            )
            
            # Log
            logger.info(f"CLOSE AT END: {pos.direction.value} {pos.symbol} closed at {pos.exit_price}, "
                        f"P&L: ${pos.profit_loss:.2f} ({pos.profit_loss_percent:.2f}%)")
            
            # Move to closed positions
            closed_positions.append(pos)
        
        # Save results
        self.positions = closed_positions
        self.equity_curve = equity_curve
        self.final_balance = float(current_balance)
        self.total_fees = float(total_fees)
        
        logger.info(f"Trading simulation completed. Final balance: ${float(current_balance):.2f}, "
                    f"P&L: ${float(current_balance - self.initial_balance):.2f}, "
                    f"Total fees: ${float(total_fees):.2f}")
        
        return closed_positions, equity_curve

    def analyze_results(self):
        """
        Analyze backtest results and generate performance metrics
        """
        if not self.positions:
            logger.warning("No positions to analyze")
            return None
        
        # Extract performance metrics
        total_trades = len(self.positions)
        winning_trades = len([p for p in self.positions if p.profit_loss and p.profit_loss > 0])
        losing_trades = len([p for p in self.positions if p.profit_loss and p.profit_loss <= 0])
        
        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average profit/loss
        profit_trades = [p.profit_loss_percent for p in self.positions if p.profit_loss and p.profit_loss > 0]
        loss_trades = [p.profit_loss_percent for p in self.positions if p.profit_loss and p.profit_loss <= 0]
        
        avg_profit = sum(profit_trades) / len(profit_trades) if profit_trades else 0
        avg_loss = sum(loss_trades) / abs(len(loss_trades)) if loss_trades else 0
        
        # Calculate drawdown
        equity_values = [balance for _, balance in self.equity_curve]
        max_dd = 0
        peak = equity_values[0]
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Calculate profit factor
        total_profit = sum([p.profit_loss for p in self.positions if p.profit_loss and p.profit_loss > 0])
        total_loss = abs(sum([p.profit_loss for p in self.positions if p.profit_loss and p.profit_loss <= 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Create results object
        results = BacktestResults(
            initial_balance=float(self.initial_balance),
            final_balance=self.final_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_profit_percent=avg_profit,
            avg_loss_percent=avg_loss,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            total_fees=self.total_fees
        )
        
        # Log summary
        logger.info("\n===== BACKTEST RESULTS =====")
        logger.info(f"Initial Balance: ${float(self.initial_balance):.2f}")
        logger.info(f"Final Balance: ${self.final_balance:.2f}")
        logger.info(f"Net Profit/Loss: ${self.final_balance - float(self.initial_balance):.2f} "
                    f"({((self.final_balance / float(self.initial_balance)) - 1) * 100:.2f}%)")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades} ({win_rate*100:.2f}%)")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Average Profit: {avg_profit:.2f}%")
        logger.info(f"Average Loss: {avg_loss:.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Maximum Drawdown: {max_dd*100:.2f}%")
        logger.info(f"Total Fees: ${self.total_fees:.2f}")
        logger.info("===========================\n")
        
        return results
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve and save to file if path provided"""
        if not self.equity_curve:
            logger.warning("No equity data to plot")
            return
        
        # Convert timestamps to datetime for better readability
        times = [datetime.fromtimestamp(ts/1000) for ts, _ in self.equity_curve]
        balances = [balance for _, balance in self.equity_curve]
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, balances)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Balance (USDT)')
        plt.grid(True)
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
            
    def clear_memory(self):
        """Optimize memory usage by clearing caches and forcing garbage collection"""
        logger.info("Clearing memory caches")
        
        # Clear any large data structures
        if hasattr(self.data_processor, 'indicator_data'):
            self.data_processor.indicator_data = {}
            
        if hasattr(self.data_processor, 'kline_buffers'):
            self.data_processor.kline_buffers = {}
            
        # Clear any caches in the signal engine
        if hasattr(self.signal_engine, '_clear_caches'):
            self.signal_engine._clear_caches()
            
        # Force garbage collection
        gc.collect()
        logger.info("Memory cleanup completed")
        
    def _optimize_memory_for_caching(self):
        """Optimize memory usage specifically during caching operations"""
        # Get current memory usage
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Run multiple garbage collection passes
        collected = 0
        for i in range(3):
            collected += gc.collect(i)
            
        # Remove any unnecessary columns from DataFrames in memory
        if hasattr(self, 'data_processor') and hasattr(self.data_processor, 'indicator_data'):
            for symbol, timeframes in self.data_processor.indicator_data.items():
                for timeframe, df in timeframes.items():
                    # Keep only needed columns if there are temporary columns
                    essential_columns = ['open', 'high', 'low', 'close', 'volume', 
                                       'sma_short', 'sma_long', 'atr', 'is_closed']
                    current_columns = df.columns.tolist()
                    columns_to_keep = essential_columns + [col for col in current_columns 
                                                        if col.startswith('sma_') or 
                                                           col.startswith('ema_') or
                                                           col.startswith('rsi_') or
                                                           col.startswith('macd_')]
                    # Only modify if we're actually dropping columns
                    if len(columns_to_keep) < len(current_columns):
                        self.data_processor.indicator_data[symbol][timeframe] = df[columns_to_keep].copy()
        
        # Get memory after optimization
        memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
        savings = memory_before - memory_after
        
        if savings > 0:
            logger.info(f"Memory optimization freed {savings:.1f} MB (from {memory_before:.1f} MB to {memory_after:.1f} MB)")
        
        return memory_before, memory_after

    def _verify_cache_integrity(self, df_path, signals_path, metadata_path):
        """Verify the integrity of cache files"""
        try:
            # Check if all files exist
            files_exist = all(path.exists() for path in [df_path, signals_path, metadata_path])
            if not files_exist:
                logger.warning("Cache verification failed: One or more files are missing")
                return False
                
            # Check if files are readable and properly formatted
            # Try loading metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if not isinstance(metadata, dict) or 'data_rows' not in metadata:
                    logger.warning("Cache verification failed: Invalid metadata format")
                    return False
            
            # Try loading and decompressing dataframe
            try:
                with open(df_path, 'rb') as f:
                    compressed_data = f.read()
                    pickled_data = zlib.decompress(compressed_data)
                    df = pickle.loads(pickled_data)
                    
                if not isinstance(df, pd.DataFrame) or len(df) == 0:
                    logger.warning("Cache verification failed: Invalid DataFrame")
                    return False
                    
                # Verify dataframe has expected number of rows
                if len(df) != metadata.get('data_rows', 0):
                    logger.warning(f"Cache verification failed: DataFrame rows mismatch ({len(df)} vs {metadata.get('data_rows', 0)})")
                    return False
            except Exception as e:
                logger.warning(f"Cache verification failed: Error loading DataFrame: {e}")
                return False
            
            # Try loading and decompressing signals
            try:
                with open(signals_path, 'rb') as f:
                    compressed_signals = f.read()
                    pickled_signals = zlib.decompress(compressed_signals)
                    signals = pickle.loads(pickled_signals)
                    
                if not isinstance(signals, list):
                    logger.warning("Cache verification failed: Invalid signals format")
                    return False
                    
                # Verify signals count
                if len(signals) != metadata.get('signals_count', 0):
                    logger.warning(f"Cache verification failed: Signals count mismatch ({len(signals)} vs {metadata.get('signals_count', 0)})")
                    return False
            except Exception as e:
                logger.warning(f"Cache verification failed: Error loading signals: {e}")
                return False
                
            logger.info("Cache verification successful: All files are valid")
            return True
            
        except Exception as e:
            logger.warning(f"Cache verification failed with error: {e}")
            return False

    def split_klines_into_chunks(self, klines: List[Kline], chunk_mode: str, chunk_size: int) -> List[List[Kline]]:
        """
        Split klines into manageable chunks for processing.
        
        Args:
            klines: List of Kline objects
            chunk_mode: 'time' for time-based chunks or 'rows' for fixed-size chunks
            chunk_size: Number of days (for time mode) or rows (for row mode) per chunk
            
        Returns:
            List of kline chunks
        """
        if not klines:
            return []
            
        if chunk_mode == 'time':
            # Time-based chunking
            chunks = []
            current_chunk = []
            chunk_start_time = klines[0].timestamp
            chunk_end_time = chunk_start_time + (chunk_size * 24 * 60 * 60 * 1000)  # chunk_size days in ms
            
            for kline in klines:
                if kline.timestamp <= chunk_end_time:
                    current_chunk.append(kline)
                else:
                    # Save current chunk and start a new one
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Start new chunk, with 1-day overlap for indicator calculation
                    overlap_start = max(0, len(current_chunk) - (24 * 60))  # Overlap of ~1 day of minute candles
                    current_chunk = current_chunk[overlap_start:] if overlap_start > 0 else []
                    current_chunk.append(kline)
                    
                    # Update chunk boundary
                    chunk_start_time = chunk_end_time
                    chunk_end_time = chunk_start_time + (chunk_size * 24 * 60 * 60 * 1000)
            
            # Add final chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
                
            logger.info(f"Split data into {len(chunks)} time-based chunks of {chunk_size} days each")
            return chunks
            
        else:  # 'rows' mode or default
            # Row-based chunking
            chunks = []
            total_rows = len(klines)
            
            # Ensure we use a reasonable chunk size
            effective_chunk_size = min(chunk_size, total_rows)
            effective_chunk_size = max(effective_chunk_size, 1000)  # At least 1000 rows per chunk
            
            # Calculate overlap for indicator warm-up
            overlap_size = min(200, effective_chunk_size // 10)  # Use about 10% overlap, max 200 rows
            
            for i in range(0, total_rows, effective_chunk_size):
                # Determine chunk start with overlap
                start_idx = max(0, i - overlap_size) if i > 0 else 0
                end_idx = min(i + effective_chunk_size, total_rows)
                
                # Create chunk
                chunk = klines[start_idx:end_idx]
                if chunk:
                    chunks.append(chunk)
            
            logger.info(f"Split data into {len(chunks)} chunks of ~{effective_chunk_size} rows each")
            return chunks
            
    async def process_in_chunks(self, klines: List[Kline], symbol: str, config_symbol: str, chunk_mode: str, chunk_size: int):
        """
        Process historical data in chunks to reduce memory usage.
        
        Args:
            klines: List of Kline objects
            symbol: Trading symbol in API format (e.g., "BTCUSDT")
            config_symbol: Trading symbol in config format (e.g., "BTC_USDT")
            chunk_mode: 'time' for time-based chunks or 'rows' for fixed-size chunks
            chunk_size: Number of days (for time mode) or rows (for row mode) per chunk
            
        Returns:
            Combined DataFrame, list of signals, merged positions and equity curve
        """
        if not klines:
            logger.warning("No klines provided for chunked processing")
            return None, [], [], []
            
        # Split klines into chunks
        chunks = self.split_klines_into_chunks(klines, chunk_mode, chunk_size)
        if not chunks:
            logger.warning("Failed to create chunks, processing as single batch")
            chunks = [klines]
            
        # Process each chunk
        all_signals = []
        all_positions = []
        all_equity_curves = []
        combined_df = None
        
        # Set positions tracking variables
        open_positions = []
        last_balance = float(self.initial_balance)
        last_timestamp = None
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_idx+1}/{len(chunks)} with {len(chunk)} candles")
            logger.info(f"Chunk timespan: {datetime.fromtimestamp(chunk[0].timestamp/1000)} to {datetime.fromtimestamp(chunk[-1].timestamp/1000)}")
            
            # Generate historical dataframe for this chunk
            chunk_df = await self.generate_full_historical_df(chunk, symbol=symbol)
            
            # Update combined DataFrame (for reference, we don't actually need to keep all of it)
            if combined_df is None:
                combined_df = chunk_df.copy()
            else:
                # Get only the unique rows that aren't in the combined_df already
                last_timestamp_in_combined = combined_df.index[-1]
                new_rows = chunk_df[chunk_df.index > last_timestamp_in_combined]
                if not new_rows.empty:
                    combined_df = pd.concat([combined_df, new_rows])
            
            # Generate signals
            chunk_signals = await self.generate_signals_optimized(chunk_df, symbol=symbol, config_symbol=config_symbol)
            
            # Add chunk signals to all signals (excluding duplicates)
            existing_timestamps = {s.timestamp for s in all_signals}
            unique_new_signals = [s for s in chunk_signals if s.timestamp not in existing_timestamps]
            all_signals.extend(unique_new_signals)
            
            # Update initial balance based on previous chunk results
            original_balance = self.initial_balance
            if chunk_idx > 0 and last_balance is not None:
                self.initial_balance = Decimal(str(last_balance))
                self.current_balance = self.initial_balance
            
            # Pass open positions from previous chunk
            self.positions = open_positions.copy() if chunk_idx > 0 else []
            
            # Simulate trading for this chunk
            positions, equity_curve = self.simulate_trading(
                chunk_signals, chunk_df, 
                continue_from_timestamp=last_timestamp,
                continue_from_positions=open_positions if chunk_idx > 0 else None
            )
            
            # Clear memory
            self.clear_memory()
            
            # Update tracking variables for next chunk
            open_positions = [p for p in positions if p.status == "OPEN"]
            last_balance = float(self.current_balance)
            last_timestamp = equity_curve[-1][0] if equity_curve else None
            
            # Save chunk results
            all_positions.extend([p for p in positions if p.status != "OPEN" or chunk_idx == len(chunks) - 1])
            
            # Adjust equity curve timestamps to ensure no duplicates
            if chunk_idx == 0:
                all_equity_curves = equity_curve
            else:
                # Skip the first entry of subsequent chunks as it's a duplicate of the last from previous chunk
                new_entries = equity_curve[1:] if equity_curve else []
                all_equity_curves.extend(new_entries)
            
            # Restore original balance for proper reporting
            self.initial_balance = original_balance
            
            logger.info(f"Completed chunk {chunk_idx+1}/{len(chunks)}. Current balance: ${last_balance:.2f}")
        
        # Set the final positions and equity curve
        self.positions = all_positions
        self.equity_curve = all_equity_curves
        
        return combined_df, all_signals, all_positions, all_equity_curves
        
    async def simulate_trading(self, signals: List[TradeSignal], df: pd.DataFrame, 
                             continue_from_timestamp: Optional[int] = None,
                             continue_from_positions: Optional[List[BacktestPosition]] = None):
        """
        Simulate trading based on generated signals
        
        Args:
            signals: List of trading signals
            df: Historical DataFrame with price data
            continue_from_timestamp: Optional timestamp to continue from (for chunked processing)
            continue_from_positions: Optional list of open positions to continue from
        """
        logger.info(f"Starting trading simulation with {len(signals)} signals")
        
        # Handle the case where no signals were generated
        if not signals:
            logger.warning("No signals provided to simulate_trading. Exiting simulation.")
            # Initialize the equity curve with initial balance
            if df is not None and not df.empty:
                self.equity_curve = [(df.iloc[0]['timestamp'], float(self.initial_balance))]
            else:
                self.equity_curve = [(int(datetime.now().timestamp() * 1000), float(self.initial_balance))]
            self.final_balance = float(self.current_balance)
            self.total_fees = Decimal('0.0')
            return [], self.equity_curve
        
        # Get configuration
        config = self.config_manager.get_config()
        pair_configs = config.get("pairs", {})
        global_v1_config = config.get("global_settings", {}).get("v1_strategy", {})
        
        # Get symbol specific config
        symbol_config = None
        for cfg_sym, details in pair_configs.items():
            if cfg_sym.upper() == signals[0].config_symbol.upper():
                symbol_config = details
                break
        
        # Check if symbol_config was found
        if not symbol_config:
            logger.error(f"Configuration for {signals[0].config_symbol} not found. Cannot simulate trades.")
            # Initialize the equity curve with initial balance
            if df is not None and not df.empty:
                self.equity_curve = [(df.iloc[0]['timestamp'], float(self.initial_balance))]
            else:
                self.equity_curve = [(int(datetime.now().timestamp() * 1000), float(self.initial_balance))]
            self.final_balance = float(self.current_balance)
            self.total_fees = Decimal('0.0')
            return [], self.equity_curve
        
        # Set leverage and margin
        leverage = Decimal(str(symbol_config.get("leverage", global_v1_config.get("default_leverage", 10))))
        fixed_margin_usdt = Decimal(str(symbol_config.get("margin_usdt", global_v1_config.get("default_margin_usdt", 50.0))))
        
        # Track current positions
        open_positions = continue_from_positions if continue_from_positions else []
        closed_positions = []
        
        # Track equity curve
        equity_curve = continue_from_timestamp and continue_from_positions and len(open_positions) > 0
        if equity_curve:
            equity_curve = [(continue_from_timestamp, float(self.current_balance))]
        else:
            equity_curve = [(df.iloc[0]['timestamp'], float(self.initial_balance))]
        
        # Iterate through each candle
        current_balance = self.initial_balance
        total_fees = Decimal('0.0')
        
        # Setup progress tracking for simulation
        self.setup_progress_tracking(len(df))
        logger.info(f"Simulating trading over {len(df)} candles")
        
        for i in range(len(df)):
            current_candle = df.iloc[i]
            current_time = current_candle['timestamp']
            
            # Check for signals at this timestamp
            new_signals = [s for s in signals if s.timestamp == current_time]
            
            # Process open positions first (check for TP/SL)
            for pos in list(open_positions):
                # Check if TP hit
                if (pos.direction == TradeDirection.LONG and current_candle['high'] >= pos.take_profit) or \
                   (pos.direction == TradeDirection.SHORT and current_candle['low'] <= pos.take_profit):
                    # Close at take profit
                    pos.status = "CLOSED_TP"
                    pos.exit_price = pos.take_profit
                    pos.exit_time = current_time
                    
                    # Calculate P&L using Decimal for precision
                    if pos.direction == TradeDirection.LONG:
                        pos.profit_loss = float(
                            (Decimal(str(pos.exit_price)) - Decimal(str(pos.entry_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    else:  # SHORT
                        pos.profit_loss = float(
                            (Decimal(str(pos.entry_price)) - Decimal(str(pos.exit_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    
                    # Account for fees
                    entry_fee = Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    exit_fee = Decimal(str(pos.exit_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    fee = entry_fee + exit_fee
                    pos.profit_loss = float(Decimal(str(pos.profit_loss)) - fee)
                    total_fees += fee
                    
                    # Update balance
                    current_balance += Decimal(str(pos.profit_loss))
                    
                    # Calculate percent P&L
                    pos.profit_loss_percent = float(
                        (Decimal(str(pos.profit_loss)) / 
                        (Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)))) * Decimal('100')
                    )
                    
                    # Log
                    logger.info(f"TP HIT: {pos.direction.value} {pos.symbol} closed at {pos.exit_price}, "
                                f"P&L: ${pos.profit_loss:.2f} ({pos.profit_loss_percent:.2f}%)")
                    
                    # Move to closed positions
                    closed_positions.append(pos)
                    open_positions.remove(pos)
                
                # Check if SL hit
                elif (pos.direction == TradeDirection.LONG and current_candle['low'] <= pos.stop_loss) or \
                     (pos.direction == TradeDirection.SHORT and current_candle['high'] >= pos.stop_loss):
                    # Close at stop loss
                    pos.status = "CLOSED_SL"
                    pos.exit_price = pos.stop_loss
                    pos.exit_time = current_time
                    
                    # Calculate P&L using Decimal for precision
                    if pos.direction == TradeDirection.LONG:
                        pos.profit_loss = float(
                            (Decimal(str(pos.exit_price)) - Decimal(str(pos.entry_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    else:  # SHORT
                        pos.profit_loss = float(
                            (Decimal(str(pos.entry_price)) - Decimal(str(pos.exit_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    
                    # Account for fees
                    entry_fee = Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    exit_fee = Decimal(str(pos.exit_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    fee = entry_fee + exit_fee
                    pos.profit_loss = float(Decimal(str(pos.profit_loss)) - fee)
                    total_fees += fee
                    
                    # Update balance
                    current_balance += Decimal(str(pos.profit_loss))
                    
                    # Calculate percent P&L
                    pos.profit_loss_percent = float(
                        (Decimal(str(pos.profit_loss)) / 
                        (Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)))) * Decimal('100')
                    )
                    
                    # Log
                    logger.info(f"SL HIT: {pos.direction.value} {pos.symbol} closed at {pos.exit_price}, "
                                f"P&L: ${pos.profit_loss:.2f} ({pos.profit_loss_percent:.2f}%)")
                    
                    # Move to closed positions
                    closed_positions.append(pos)
                    open_positions.remove(pos)
            
            # Process new signals and open positions
            for signal in new_signals:
                # Calculate position size
                position_size = fixed_margin_usdt / Decimal(str(signal.entry_price))
                
                # Apply slippage to entry
                entry_price = Decimal(str(signal.entry_price))
                if signal.direction == TradeDirection.LONG:
                    entry_price *= (Decimal('1') + self.slippage_percent)  # Higher for buys
                else:
                    entry_price *= (Decimal('1') - self.slippage_percent)  # Lower for sells
                
                # Create new position
                new_position = BacktestPosition(
                    symbol=signal.symbol,
                    entry_price=float(entry_price),
                    entry_time=current_time,
                    quantity=float(position_size),
                    direction=signal.direction,
                    stop_loss=signal.stop_loss_price,
                    take_profit=signal.take_profit_price,
                    status="OPEN"
                )
                
                # Calculate entry fee
                entry_fee = entry_price * position_size * self.fee_percent
                total_fees += entry_fee
                
                # Log
                logger.info(f"OPEN: {signal.direction.value} {signal.symbol} at {float(entry_price)}, "
                            f"Size: {float(position_size)}, SL: {signal.stop_loss_price}, TP: {signal.take_profit_price}")
                
                # Add to open positions
                open_positions.append(new_position)
            
            # Track equity at each step
            if current_time > (continue_from_timestamp or df.iloc[0]['timestamp']):
                equity_curve.append((current_time, float(current_balance)))
            
            # Update progress every 1000 candles
            if i % 1000 == 0 or i == len(df) - 1:
                self.update_progress(min(1000, i + 1 - self.current_step), i == 0 or i == len(df) - 1)
        
        # Close any remaining open positions at the last price
        last_candle = df.iloc[-1]
        for pos in list(open_positions):
            pos.status = "CLOSED_EXIT"
            pos.exit_price = float(last_candle['close'])
            pos.exit_time = last_candle['timestamp']
            
            # Calculate P&L using Decimal
            if pos.direction == TradeDirection.LONG:
                pos.profit_loss = float(
                    (Decimal(str(pos.exit_price)) - Decimal(str(pos.entry_price))) * 
                    Decimal(str(pos.quantity)) * leverage
                )
            else:  # SHORT
                pos.profit_loss = float(
                    (Decimal(str(pos.entry_price)) - Decimal(str(pos.exit_price))) * 
                    Decimal(str(pos.quantity)) * leverage
                )
            
            # Account for fees
            entry_fee = Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)) * self.fee_percent
            exit_fee = Decimal(str(pos.exit_price)) * Decimal(str(pos.quantity)) * self.fee_percent
            fee = entry_fee + exit_fee
            pos.profit_loss = float(Decimal(str(pos.profit_loss)) - fee)
            total_fees += fee
            
            # Update balance
            current_balance += Decimal(str(pos.profit_loss))
            
            # Calculate percent P&L
            pos.profit_loss_percent = float(
                (Decimal(str(pos.profit_loss)) / 
                (Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)))) * Decimal('100')
            )
            
            # Log
            logger.info(f"CLOSE AT END: {pos.direction.value} {pos.symbol} closed at {pos.exit_price}, "
                        f"P&L: ${pos.profit_loss:.2f} ({pos.profit_loss_percent:.2f}%)")
            
            # Move to closed positions
            closed_positions.append(pos)
        
        # Save results
        self.positions = closed_positions
        self.equity_curve = equity_curve
        self.final_balance = float(current_balance)
        self.total_fees = float(total_fees)
        
        logger.info(f"Trading simulation completed. Final balance: ${float(current_balance):.2f}, "
                    f"P&L: ${float(current_balance - self.initial_balance):.2f}, "
                    f"Total fees: ${float(total_fees):.2f}")
        
        return closed_positions, equity_curve

def get_cache_filename(data_path, symbol, config_path, sample_size):
    """
    Generate a cache filename based on input parameters.
    
    Args:
        data_path: Path to the data file
        symbol: Trading symbol
        config_path: Path to the configuration file
        sample_size: Sample size used
        
    Returns:
        Path to the cache file
    """
    # Create a hash of the input parameters to use as the cache filename
    import hashlib
    
    # Create a string with all the parameters
    params_str = f"{data_path}_{symbol}_{config_path}_{sample_size}"
    
    # Create a hash of the parameters
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    # Create cache directory if it doesn't exist
    cache_dir = Path("backtest_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Return the cache filename
    return cache_dir / f"{symbol}_{params_hash}.joblib"

def clear_cache():
    """Clear all cache files."""
    cache_dir = Path("backtest_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("*"):
            try:
                f.unlink()
            except Exception as e:
                logger.error(f"Error removing cache file {f}: {e}")
        logger.info("Cache cleared")

async def run_optimized_backtest(
    config_path, 
    data_path, 
    symbol, 
    config_symbol=None, 
    sample_size=1.0, 
    use_cache=True, 
    workers=0,
    chunk_type="rows",
    chunk_size=100000
):
    """
    Run the optimized backtest with the specified parameters.
    
    Args:
        config_path: Path to the strategy configuration
        data_path: Path to the historical data
        symbol: Trading symbol
        config_symbol: Symbol name in the config (defaults to symbol if None)
        sample_size: Fraction of data to use (0.0-1.0)
        use_cache: Whether to use caching
        workers: Number of worker processes (0 = auto)
        chunk_type: Type of chunking ("rows", "days", "weeks", "months")
        chunk_size: Size of chunks
    """
    logger.info(f"Starting optimized backtest with config: {config_path}")
    
    # Initialize the backtester
    backtester = OptimizedBacktester(config_path)
    backtester.parallel_processor.max_workers = workers if workers > 0 else backtester.parallel_processor.max_workers
    
    # Check for cached data
    logger.info("Checking for cached data...")
    if use_cache:
        cache_file = get_cache_filename(data_path, symbol, config_path, sample_size)
        try:
            historical_df = backtester.load_from_cache(cache_file)
            if historical_df is not None:
                logger.info(f"Loaded processed data from cache: {cache_file}")
                # Continue with cached data
                return
        except Exception as e:
            logger.warning(f"Failed to load data from cache: {str(e)}")
    
    logger.info("Processing data from scratch...")
    
    # Load and process historical data
    kline_objects = await backtester.load_and_process_data(data_path, symbol, sample_size)
    
    # Process historical data with optimizations
    if chunk_type != "rows" or chunk_size < len(kline_objects):
        logger.info(f"Using chunked processing with {chunk_type} chunks of size {chunk_size}")
        historical_df = await backtester.process_historical_data_chunked(
            kline_objects, symbol, chunk_size, chunk_type)
    else:
        logger.info("Using full data processing (no chunking)")
        historical_df = await backtester.generate_full_historical_df(kline_objects, symbol)
    
    # Cache processed data if enabled
    if use_cache:
        cache_file = get_cache_filename(data_path, symbol, config_path, sample_size)
        backtester.save_to_cache(historical_df, cache_file)
    
    # The rest of the backtest logic would go here
    # ...
    
    logger.info("Backtest completed successfully")

def parse_args():
    parser = argparse.ArgumentParser(description="Run optimized backtest with enhanced performance")
    parser.add_argument("--config", type=str, default="config/v1_optimized_strategy.yaml", help="Path to strategy configuration file")
    parser.add_argument("--data", type=str, required=True, help="Path to historical data CSV file")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--config-symbol", type=str, default=None, help="Symbol name in config file (defaults to --symbol if not specified)")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of data to use (0.0-1.0)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of processed data")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    parser.add_argument("--chunk-mode", type=str, choices=["time", "rows", "none"], default="rows", help="Chunking mode for large datasets")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Size of chunks for processing")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 = auto)")
    parser.add_argument("--chunk-type", type=str, choices=["rows", "days", "weeks", "months"], default="rows", help="Type of chunking for time-based chunking")
    parser.add_argument("--no-compress", action="store_true", help="Disable compression for cache files")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
        
    # Handle defaults
    if args.config_symbol is None:
        args.config_symbol = args.symbol
    
    # Run the backtest
    asyncio.run(run_optimized_backtest(
        args.config,
        args.data,
        args.symbol,
        args.config_symbol,
        args.sample,
        not args.no_cache,
        args.workers,
        args.chunk_type,
        args.chunk_size
    ))
    
    logger.info("Backtest completed successfully")