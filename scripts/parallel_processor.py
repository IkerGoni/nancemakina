#!/usr/bin/env python3
import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Dict, Any, Optional
import logging
import time
from functools import partial

logger = logging.getLogger('parallel_processor')

class ParallelProcessor:
    """
    Utility for parallel processing of CPU-intensive operations.
    Uses multiprocessing to speed up indicator calculations and other operations.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes to use.
                If None, uses the number of CPU cores.
        """
        self.max_workers = max_workers or mp.cpu_count()
        logger.info(f"Initialized parallel processor with {self.max_workers} workers")
        
    def chunk_data(self, df: pd.DataFrame, n_chunks: int) -> List[pd.DataFrame]:
        """
        Split a DataFrame into roughly equal chunks for parallel processing.
        
        Args:
            df: DataFrame to split
            n_chunks: Number of chunks to create
            
        Returns:
            List of DataFrame chunks
        """
        if len(df) < n_chunks:
            return [df]
            
        chunk_size = len(df) // n_chunks
        chunks = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_chunks - 1 else len(df)
            chunks.append(df.iloc[start_idx:end_idx].copy())
            
        return chunks
        
    def process_chunks(self, func: Callable, chunks: List[Any], *args, **kwargs) -> List[Any]:
        """
        Process chunks of data in parallel.
        
        Args:
            func: Function to apply to each chunk
            chunks: List of data chunks to process
            *args, **kwargs: Additional arguments to pass to func
            
        Returns:
            List of results from processing each chunk
        """
        if len(chunks) == 1:
            # Don't use multiprocessing for single chunk
            return [func(chunks[0], *args, **kwargs)]
            
        # Create a pool with the specified number of workers
        n_workers = min(self.max_workers, len(chunks))
        
        # Create a partial function with the provided args and kwargs
        partial_func = partial(func, *args, **kwargs)
        
        # Process chunks in parallel
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(partial_func, chunks)
            
        return results
        
    def merge_results(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge results from parallel processing back into a single DataFrame.
        
        Args:
            results: List of DataFrames to merge
            
        Returns:
            Merged DataFrame
        """
        if not results:
            return pd.DataFrame()
            
        # Concatenate DataFrames and sort by timestamp if present
        merged_df = pd.concat(results, ignore_index=True)
        
        if 'timestamp' in merged_df.columns:
            merged_df.sort_values('timestamp', inplace=True)
            merged_df.reset_index(drop=True, inplace=True)
            
        return merged_df
        
    def parallel_apply(self, df: pd.DataFrame, func: Callable, *args, **kwargs) -> pd.DataFrame:
        """
        Apply a function to a DataFrame in parallel.
        
        Args:
            df: DataFrame to process
            func: Function to apply to each chunk
            *args, **kwargs: Additional arguments to pass to func
            
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
            
        # Determine number of chunks based on CPU cores
        n_chunks = min(self.max_workers, max(1, len(df) // 10000))
        
        # Split DataFrame into chunks
        chunks = self.chunk_data(df, n_chunks)
        
        # Process chunks in parallel
        start_time = time.time()
        results = self.process_chunks(func, chunks, *args, **kwargs)
        elapsed = time.time() - start_time
        
        # Merge results
        merged_df = self.merge_results(results)
        
        logger.info(f"Parallel processing completed in {elapsed:.2f} seconds using {n_chunks} chunks")
        return merged_df

    def parallel_batch_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators in parallel.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicator columns added
        """
        from scripts.indicator_processor import VectorizedIndicatorProcessor
        
        def process_chunk(chunk):
            # Create a processor instance for this chunk
            processor = VectorizedIndicatorProcessor()
            # Process all indicators
            return processor.batch_process_indicators(chunk)
            
        return self.parallel_apply(df, process_chunk)
        
    def parallel_crossover_detection(self, df: pd.DataFrame) -> List[int]:
        """
        Detect crossover points in parallel.
        
        Args:
            df: DataFrame with indicator data
            
        Returns:
            List of indices where crossovers occur
        """
        def detect_crossovers(chunk):
            if 'sma_short' not in chunk.columns or 'sma_long' not in chunk.columns:
                return []
                
            # Skip first row as we need previous values for comparison
            if len(chunk) <= 1:
                return []
                
            # Vectorized crossover detection
            prev_comparison = chunk['sma_short'].shift(1) <= chunk['sma_long'].shift(1)
            curr_comparison = chunk['sma_short'] > chunk['sma_long']
            crossed_up_mask = (prev_comparison) & (curr_comparison)
            
            prev_comparison = chunk['sma_short'].shift(1) >= chunk['sma_long'].shift(1)
            curr_comparison = chunk['sma_short'] < chunk['sma_long']
            crossed_down_mask = (prev_comparison) & (curr_comparison)
            
            # Get indices of all potential crossovers
            combined_mask = crossed_up_mask | crossed_down_mask
            
            # Return the original indices from the full DataFrame
            return chunk.index[combined_mask].tolist()
            
        # Process in parallel
        n_chunks = min(self.max_workers, max(1, len(df) // 10000))
        chunks = self.chunk_data(df, n_chunks)
        results = self.process_chunks(detect_crossovers, chunks)
        
        # Combine all indices
        all_indices = []
        for indices in results:
            all_indices.extend(indices)
            
        return sorted(all_indices) 