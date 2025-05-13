#!/bin/bash

echo "Killing any running backtest processes..."
pkill -f "python -m scripts.backtest" || echo "No backtest processes to kill"

# Wait a moment for processes to fully terminate
sleep 2

echo "Starting optimized backtest with 10% data sample..."
# Run the optimized backtest with a small sample of data for quick testing
# Using cache by default for faster consecutive runs
./run_optimized_backtest.sh --quick

echo "Optimized backtest started." 