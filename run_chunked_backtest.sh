#!/bin/bash

# Chunked Backtest Runner with Optimizations
# This script runs the optimized backtest in chunks to manage memory and performance

# Default values
CONFIG="config/v1_optimized_strategy.yaml"
DATA="backtest_data/merged_futures_data.csv"
SYMBOL="BTCUSDT"
CONFIG_SYMBOL="BTC_USDT"
WORKERS=0  # 0 means auto-detect CPU cores
CHUNK_SIZE=50000
CHUNK_TYPE="rows"
CACHE=true
COMPRESS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --data)
      DATA="$2"
      shift 2
      ;;
    --symbol)
      SYMBOL="$2"
      shift 2
      ;;
    --config-symbol)
      CONFIG_SYMBOL="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --chunk-type)
      CHUNK_TYPE="$2"
      shift 2
      ;;
    --no-cache)
      CACHE=false
      shift
      ;;
    --no-compress)
      COMPRESS=false
      shift
      ;;
    --sample)
      SAMPLE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Check if the data file exists
if [ ! -f "$DATA" ]; then
  echo "Error: Data file not found: $DATA"
  exit 1
fi

# Check if the config file exists
if [ ! -f "$CONFIG" ]; then
  echo "Error: Config file not found: $CONFIG"
  exit 1
fi

# Print execution details
echo "Running optimized chunked backtest with:"
echo "  Config: $CONFIG"
echo "  Data: $DATA"
echo "  Symbol: $SYMBOL"
echo "  Config Symbol: $CONFIG_SYMBOL"
echo "  Workers: $WORKERS (0 = auto)"
echo "  Chunk Size: $CHUNK_SIZE"
echo "  Chunk Type: $CHUNK_TYPE"
echo "  Caching: $CACHE"
echo "  Compression: $COMPRESS"
if [ ! -z "$SAMPLE" ]; then
  echo "  Sample size: $SAMPLE"
fi
echo ""

# Create required directories if they don't exist
mkdir -p logs
mkdir -p backtest_results
mkdir -p backtest_cache

# Build the command
CMD="python -m scripts.backtest_optimized --config $CONFIG --data $DATA --symbol $SYMBOL --config-symbol $CONFIG_SYMBOL --workers $WORKERS --chunk-size $CHUNK_SIZE --chunk-type $CHUNK_TYPE"

# Add optional parameters
if [ "$CACHE" = false ]; then
  CMD="$CMD --no-cache"
fi

if [ "$COMPRESS" = false ]; then
  CMD="$CMD --no-compress"
fi

if [ ! -z "$SAMPLE" ]; then
  CMD="$CMD --sample $SAMPLE"
fi

# Execute the backtest
echo "Executing: $CMD"
$CMD

exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "Backtest failed with exit code $exit_code"
  exit $exit_code
fi

echo "Backtest completed successfully" 