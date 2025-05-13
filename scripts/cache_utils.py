#!/usr/bin/env python3
"""
Utility script for managing backtest cache files
"""
import argparse
import json
import os
from pathlib import Path
import shutil
from datetime import datetime

def list_cache(cache_dir="backtest_cache", verbose=False):
    """List all cache entries with metadata"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory '{cache_dir}' does not exist")
        return
    
    # Group files by their cache key prefix
    cache_entries = {}
    
    for file in cache_path.glob("*"):
        if file.is_file():
            parts = file.name.split("_")
            if len(parts) >= 2:
                cache_key = parts[0]
                file_type = "_".join(parts[1:])
                
                if cache_key not in cache_entries:
                    cache_entries[cache_key] = {}
                    
                cache_entries[cache_key][file_type] = file
    
    # Print cache entries
    if not cache_entries:
        print("Cache is empty")
        return
    
    print(f"Found {len(cache_entries)} cache entries:")
    
    for i, (key, files) in enumerate(cache_entries.items(), 1):
        metadata_file = files.get("metadata.json")
        
        if metadata_file and metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                timestamp = datetime.fromisoformat(metadata['timestamp'])
                age = datetime.now() - timestamp
                
                print(f"{i}. Cache Key: {key}")
                print(f"   Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({age.days} days, {age.seconds//3600} hours ago)")
                print(f"   Symbol: {metadata['symbol']} ({metadata['config_symbol']})")
                print(f"   Sample Size: {metadata['sample_size']:.2f}")
                print(f"   Data: {metadata['data_rows']} rows, {metadata['signals_count']} signals")
                
                # Display cache stats if available
                if 'cache_stats' in metadata:
                    stats = metadata['cache_stats']
                    total_size_mb = stats.get('total_size_bytes', 0) / (1024 * 1024)
                    print(f"   Size: {total_size_mb:.2f} MB, Compression: {stats.get('compression_savings_pct', 0):.1f}% saved")
                
                if verbose:
                    print(f"   Config: {metadata['config_file']}")
                    print(f"   Data File: {metadata['data_file']}")
                    file_sizes = []
                    for file_path in files.values():
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        file_sizes.append(f"{file_path.name}: {size_mb:.2f} MB")
                    print(f"   Files: {', '.join(file_sizes)}")
                    
                print()
            except Exception as e:
                print(f"{i}. Cache Key: {key} (Error reading metadata: {e})")
        else:
            print(f"{i}. Cache Key: {key} (No metadata available)")
            if verbose:
                print(f"   Files: {', '.join(f.name for f in files.values())}")
            print()

def clear_cache(cache_dir="backtest_cache", older_than_days=None):
    """Clear all or old cache files"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Cache directory '{cache_dir}' does not exist")
        return
    
    if older_than_days is not None:
        # Only delete files older than specified days
        now = datetime.now()
        deleted_count = 0
        total_count = 0
        
        for file in cache_path.glob("*metadata.json"):
            total_count += 1
            try:
                with open(file, 'r') as f:
                    metadata = json.load(f)
                
                timestamp = datetime.fromisoformat(metadata['timestamp'])
                age = now - timestamp
                
                if age.days >= older_than_days:
                    # Get cache key
                    cache_key = file.name.split("_")[0]
                    # Delete all files with this cache key
                    for related_file in cache_path.glob(f"{cache_key}_*"):
                        related_file.unlink()
                        deleted_count += 1
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        if total_count == 0:
            print("No cache entries found")
        else:
            print(f"Deleted {deleted_count} files from {total_count} cache entries older than {older_than_days} days")
    else:
        # Delete all cache files
        file_count = 0
        for file in cache_path.glob("*"):
            if file.is_file():
                file.unlink()
                file_count += 1
        
        if file_count == 0:
            print("Cache is already empty")
        else:
            print(f"Deleted {file_count} cache files")

def main():
    parser = argparse.ArgumentParser(description="Backtest Cache Utility")
    parser.add_argument('--cache-dir', type=str, default='backtest_cache', help='Cache directory path')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List cache entries')
    list_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed information')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache')
    clear_parser.add_argument('--older-than', type=int, default=None, 
                             help='Only clear cache entries older than specified days')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_cache(args.cache_dir, args.verbose)
    elif args.command == 'clear':
        clear_cache(args.cache_dir, args.older_than)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 