#!/bin/bash

# Get path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Set default parameters
CONFIG="config/v1_optimized_strategy.yaml"
DATA="backtest_data/merged_futures_data_quarter.csv"
SYMBOL="BTCUSDT"
CONFIG_SYMBOL="BTC_USDT"
SAMPLE="1.0"
CACHE_OPTS=""
CHUNK_OPTS=""

# Parse command line options
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
    --sample)
      SAMPLE="$2"
      shift 2
      ;;
    --quick)
      # Quick test with 10% of data
      SAMPLE="0.1"
      shift
      ;;
    --medium)
      # Medium test with 25% of data
      SAMPLE="0.25"
      shift
      ;;
    --no-cache)
      CACHE_OPTS="--no-cache"
      shift
      ;;
    --clear-cache)
      CACHE_OPTS="--clear-cache"
      shift
      ;;
    --chunk-time)
      CHUNK_OPTS="--chunk-mode time --chunk-size ${2:-7}"
      shift 2
      ;;
    --chunk-rows)
      CHUNK_OPTS="--chunk-mode rows --chunk-size ${2:-100000}"
      shift 2
      ;;
    --chunk-size)
      # Use with previous chunk-mode option
      CHUNK_SIZE="$2"
      if [[ -n "$CHUNK_OPTS" ]]; then
        CHUNK_OPTS="${CHUNK_OPTS/--chunk-size *$/--chunk-size $CHUNK_SIZE}"
      fi
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --config FILE          Configuration file (default: $CONFIG)"
      echo "  --data FILE            Data file (default: $DATA)"
      echo "  --symbol SYMBOL        Trading symbol (default: $SYMBOL)"
      echo "  --config-symbol SYM    Config symbol (default: $CONFIG_SYMBOL)"
      echo "  --sample VALUE         Data sample size 0.0-1.0 (default: $SAMPLE)"
      echo "  --quick                Quick test with 10% of data"
      echo "  --medium               Medium test with 25% of data"
      echo "  --no-cache             Disable data caching"
      echo "  --clear-cache          Clear cache before running"
      echo "  --chunk-time DAYS      Process in time-based chunks of DAYS days (default: 7)"
      echo "  --chunk-rows N         Process in row-based chunks of N rows (default: 100000)"
      echo "  --chunk-size N         Override chunk size for previously set chunk mode"
      echo "  --help, -h             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running optimized backtest with:"
echo "  Config: $CONFIG"
echo "  Data: $DATA"
echo "  Symbol: $SYMBOL"
echo "  Config Symbol: $CONFIG_SYMBOL"
echo "  Sample size: $SAMPLE"
if [[ -n "$CACHE_OPTS" ]]; then
  echo "  Cache options: $CACHE_OPTS"
fi
if [[ -n "$CHUNK_OPTS" ]]; then
  echo "  Chunking: $CHUNK_OPTS"
fi
echo ""

# Set Python path to include project root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run the backtest
python -m scripts.backtest_optimized \
  --config "$CONFIG" \
  --data "$DATA" \
  --symbol "$SYMBOL" \
  --config-symbol "$CONFIG_SYMBOL" \
  --sample "$SAMPLE" \
  $CACHE_OPTS \
  $CHUNK_OPTS 