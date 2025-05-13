#!/usr/bin/env python3
"""
Utility for analyzing backtest cache performance and efficiency.

This script provides tools for:
1. Measuring cache size and compression efficiency
2. Analyzing performance improvements from caching
3. Validating cache data integrity
4. Cleaning up old or corrupted cache files
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cache_analyzer')

class CacheAnalyzer:
    """Utility for analyzing backtest caching efficiency and performance."""
    
    def __init__(self, cache_dir='backtest_cache'):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            logger.warning(f"Cache directory {cache_dir} does not exist. Creating it.")
            self.cache_dir.mkdir(exist_ok=True)
            
        self.cache_files = []
        self.scan_cache_files()
        
    def scan_cache_files(self):
        """Scan and collect information about all cache files."""
        self.cache_files = []
        
        # Look for pickle and joblib files
        for file_path in self.cache_dir.glob('**/*'):
            if file_path.is_file() and file_path.suffix in ['.pkl', '.joblib', '.cache']:
                file_info = {
                    'path': file_path,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                    'extension': file_path.suffix,
                }
                self.cache_files.append(file_info)
                    
        logger.info(f"Found {len(self.cache_files)} cache files")
        
    def get_total_cache_size(self):
        """Get the total size of all cache files in MB."""
        total_size = sum(file_info['size'] for file_info in self.cache_files)
        return total_size / (1024 * 1024)  # Convert to MB
        
    def get_cache_size_by_type(self):
        """Get cache size breakdown by file type."""
        size_by_type = {}
        for file_info in self.cache_files:
            extension = file_info['extension']
            if extension not in size_by_type:
                size_by_type[extension] = 0
            size_by_type[extension] += file_info['size'] / (1024 * 1024)  # Convert to MB
            
        return size_by_type
        
    def analyze_compression_efficiency(self):
        """Analyze compression efficiency of cache files."""
        results = []
        
        for file_info in self.cache_files:
            try:
                file_path = file_info['path']
                if file_path.suffix in ['.joblib', '.cache']:
                    # Try to load and get original data size
                    start_time = time.time()
                    data = joblib.load(file_path)
                    load_time = time.time() - start_time
                    
                    # Estimate uncompressed size
                    uncompressed_size = 0
                    if isinstance(data, pd.DataFrame):
                        # Pickle to memory to estimate size
                        import pickle
                        import io
                        buffer = io.BytesIO()
                        pickle.dump(data, buffer)
                        uncompressed_size = len(buffer.getvalue())
                    
                    compression_ratio = uncompressed_size / file_info['size'] if uncompressed_size > 0 else 1.0
                    
                    results.append({
                        'file': file_path.name,
                        'compressed_size_mb': file_info['size'] / (1024 * 1024),
                        'uncompressed_size_mb': uncompressed_size / (1024 * 1024),
                        'compression_ratio': compression_ratio,
                        'load_time_ms': load_time * 1000
                    })
            except Exception as e:
                logger.warning(f"Error analyzing {file_info['path']}: {str(e)}")
                
        return results
        
    def validate_cache_files(self):
        """Validate all cache files to ensure they can be loaded properly."""
        valid_files = 0
        invalid_files = []
        
        for file_info in self.cache_files:
            try:
                file_path = file_info['path']
                if file_path.suffix == '.joblib':
                    joblib.load(file_path)
                elif file_path.suffix == '.pkl':
                    pd.read_pickle(file_path)
                valid_files += 1
            except Exception as e:
                invalid_files.append((file_info['path'], str(e)))
                
        return valid_files, invalid_files
        
    def clean_invalid_files(self):
        """Remove invalid or corrupted cache files."""
        _, invalid_files = self.validate_cache_files()
        
        for file_path, error in invalid_files:
            logger.info(f"Removing invalid cache file {file_path}: {error}")
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing {file_path}: {str(e)}")
                
        return len(invalid_files)
        
    def clean_old_cache_files(self, max_age_days=30):
        """Remove cache files older than specified age."""
        now = datetime.now()
        removed_count = 0
        
        for file_info in self.cache_files:
            age_days = (now - file_info['modified']).days
            if age_days > max_age_days:
                logger.info(f"Removing old cache file {file_info['path']} (age: {age_days} days)")
                try:
                    Path(file_info['path']).unlink()
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Error removing {file_info['path']}: {str(e)}")
                    
        return removed_count
        
    def plot_cache_statistics(self, output_file=None):
        """Generate a plot of cache statistics."""
        if not self.cache_files:
            logger.warning("No cache files found to plot statistics")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot cache size by type
        size_by_type = self.get_cache_size_by_type()
        labels = list(size_by_type.keys())
        sizes = list(size_by_type.values())
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        ax1.set_title('Cache Size by File Type')
        
        # Plot file modification dates histogram
        dates = [file_info['modified'] for file_info in self.cache_files]
        ax2.hist(dates, bins=20)
        ax2.set_title('Cache File Age Distribution')
        ax2.set_xlabel('Modification Date')
        ax2.set_ylabel('Number of Files')
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved cache statistics plot to {output_file}")
        else:
            plt.show()
            
    def analyze_and_report(self):
        """Run a full analysis and print a detailed report."""
        # Get basic statistics
        total_size = self.get_total_cache_size()
        size_by_type = self.get_cache_size_by_type()
        valid_files, invalid_files = self.validate_cache_files()
        
        # Print report
        print("\n" + "="*50)
        print(" BACKTEST CACHE ANALYSIS REPORT")
        print("="*50)
        print(f"Total cache files: {len(self.cache_files)}")
        print(f"Total cache size: {total_size:.2f} MB")
        print(f"Valid files: {valid_files}")
        print(f"Invalid files: {len(invalid_files)}")
        
        print("\nCache size by file type:")
        for ext, size in size_by_type.items():
            print(f"  {ext}: {size:.2f} MB")
            
        if invalid_files:
            print("\nInvalid cache files:")
            for path, error in invalid_files:
                print(f"  {path}: {error}")
                
        # Check compression efficiency for a sample of files
        compression_data = self.analyze_compression_efficiency()
        if compression_data:
            avg_ratio = sum(d['compression_ratio'] for d in compression_data) / len(compression_data)
            avg_load_time = sum(d['load_time_ms'] for d in compression_data) / len(compression_data)
            print(f"\nCompression analysis (sample of {len(compression_data)} files):")
            print(f"  Average compression ratio: {avg_ratio:.2f}x")
            print(f"  Average load time: {avg_load_time:.2f} ms")
            
        print("="*50 + "\n")
        
def main():
    parser = argparse.ArgumentParser(description="Analyze backtest cache performance and efficiency")
    parser.add_argument('--cache-dir', default='backtest_cache', help='Directory containing cache files')
    parser.add_argument('--clean', action='store_true', help='Clean invalid cache files')
    parser.add_argument('--clean-old', type=int, metavar='DAYS', help='Clean cache files older than specified days')
    parser.add_argument('--plot', action='store_true', help='Generate plots of cache statistics')
    parser.add_argument('--output', help='Output file for plots (if --plot is specified)')
    
    args = parser.parse_args()
    
    analyzer = CacheAnalyzer(args.cache_dir)
    
    if args.clean:
        removed = analyzer.clean_invalid_files()
        print(f"Removed {removed} invalid cache files")
        analyzer.scan_cache_files()  # Rescan after cleaning
        
    if args.clean_old:
        removed = analyzer.clean_old_cache_files(args.clean_old)
        print(f"Removed {removed} cache files older than {args.clean_old} days")
        analyzer.scan_cache_files()  # Rescan after cleaning
        
    analyzer.analyze_and_report()
    
    if args.plot:
        analyzer.plot_cache_statistics(args.output)
        
if __name__ == '__main__':
    main() 