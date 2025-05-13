#!/usr/bin/env python3
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import asyncio
from decimal import Decimal
from pathlib import Path
import tempfile
import shutil

from scripts.backtest_optimized import OptimizedBacktester, run_optimized_backtest
from src.models import Kline, TradeDirection

# Create test fixtures
@pytest.fixture
def sample_klines():
    """Generate a sample of kline data for testing"""
    klines = []
    base_time = int(datetime(2025, 1, 1).timestamp() * 1000)
    
    # Create 100 minutes of 1-minute candles
    for i in range(300):
        timestamp = base_time + (i * 60 * 1000)  # Add i minutes
        
        # Create some price movement for testing SMA crossovers
        # We'll create data that has a clear SMA crossover
        base_price = 30000 + (i * 10) + (np.sin(i/10) * 500)
        
        kline = Kline(
            timestamp=timestamp,
            open=base_price - 5,
            high=base_price + 20,
            low=base_price - 20,
            close=base_price + 5,
            volume=1000 + (i * 100),
            quote_asset_volume=0,
            number_of_trades=100,
            is_closed=True,
            symbol="BTCUSDT",
            interval="1m"
        )
        klines.append(kline)
    
    return klines

@pytest.fixture
def test_config_file():
    """Create a temporary config file for testing"""
    config_content = """
# Test Configuration
engine_config:
  module: "src.signal_engine"
  class: "SignalEngineV1"

global_settings:
  v1_strategy:
    sma_short_period: 20
    sma_long_period: 50
    min_signal_interval_minutes: 10
    default_margin_usdt: 50.0
    default_leverage: 10
    margin_mode: "ISOLATED"
    indicator_timeframes: ["1m"]
    
    filters:
      buffer_time:
        enabled: false
      volume:
        enabled: false
      volatility:
        enabled: false
      trend:
        enabled: false
    
    sl_options:
      primary_type: "percentage"
      percentage_settings:
        value: 0.02
      enable_fallback: false
    
    tp_options:
      primary_type: "rr_ratio"
      rr_ratio_settings:
        value: 2.0

pairs:
  BTC_USDT:
    enabled: true
    contract_type: "USDT_M"
    
backtesting:
  initial_balance_usdt: 1000.0
  fee_percent: 0.02
  slippage_percent: 0.0
"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create the config file
    config_path = os.path.join(temp_dir, "test_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    yield config_path
    
    # Clean up
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with indicators for testing"""
    # Create a sample DataFrame with 100 rows
    base_time = int(datetime(2025, 1, 1).timestamp() * 1000)
    data = []
    
    for i in range(100):
        timestamp = base_time + (i * 60 * 1000)  # Add i minutes
        
        # Create a pattern that will generate crossovers
        if i < 40:
            sma_short = 100 + i
            sma_long = 150 - (i/2)
        elif i < 60:
            sma_short = 140 - (i-40)
            sma_long = 130 - (i-40)/2
        else:
            sma_short = 120 + (i-60)
            sma_long = 110 + (i-60)/2
        
        data.append({
            'timestamp': timestamp,
            'open': 100 + i,
            'high': 110 + i,
            'low': 90 + i,
            'close': 105 + i,
            'volume': 1000 + i*100,
            'is_closed': True,
            'sma_short': sma_short,
            'sma_long': sma_long,
            'atr': 5.0
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

# Tests
@pytest.mark.asyncio
async def test_generate_signals_optimized(sample_dataframe, test_config_file):
    """Test the optimized signal generation algorithm"""
    # Initialize the backtester
    backtester = OptimizedBacktester(test_config_file, "dummy_data_file.csv")
    
    # Generate signals using optimized approach
    signals = await backtester.generate_signals_optimized(sample_dataframe, warmup_period=10)
    
    # Check that signals were generated
    assert len(signals) > 0, "No signals were generated"
    
    # Verify signal properties
    for signal in signals:
        assert signal.symbol == "BTCUSDT"
        assert signal.config_symbol == "BTC_USDT"
        assert signal.direction in [TradeDirection.LONG, TradeDirection.SHORT]
        assert signal.entry_price > 0
        assert signal.stop_loss_price > 0
        assert signal.take_profit_price > 0
        
        # Validate LONG signal
        if signal.direction == TradeDirection.LONG:
            assert signal.entry_price > signal.stop_loss_price
            assert signal.take_profit_price > signal.entry_price
        
        # Validate SHORT signal
        if signal.direction == TradeDirection.SHORT:
            assert signal.entry_price < signal.stop_loss_price
            assert signal.take_profit_price < signal.entry_price

@pytest.mark.asyncio
async def test_progress_tracking(sample_dataframe, test_config_file):
    """Test that progress tracking is working properly"""
    # Initialize the backtester
    backtester = OptimizedBacktester(test_config_file, "dummy_data_file.csv")
    
    # Setup progress tracking
    total_steps = 100
    backtester.setup_progress_tracking(total_steps)
    
    # Update progress
    backtester.update_progress(10)
    assert backtester.current_step == 10
    
    backtester.update_progress(20)
    assert backtester.current_step == 30
    
    # Force update
    backtester.update_progress(10, force_log=True)
    assert backtester.current_step == 40

@pytest.mark.asyncio
async def test_simulate_trading(sample_dataframe, test_config_file, sample_klines):
    """Test trading simulation with predefined signals"""
    # Initialize the backtester
    backtester = OptimizedBacktester(test_config_file, "dummy_data_file.csv")
    
    # Generate signals first
    signals = await backtester.generate_signals_optimized(sample_dataframe, warmup_period=10)
    
    # Only continue if we have signals
    if len(signals) > 0:
        # Simulate trading
        positions, equity_curve = backtester.simulate_trading(signals, sample_dataframe)
        
        # Check that positions were created
        assert len(positions) > 0, "No positions were created during simulation"
        
        # Verify equity curve
        assert len(equity_curve) > 0, "No equity curve data was generated"
        assert equity_curve[0][1] == float(backtester.initial_balance), "Initial equity should match initial balance"
        
        # Check that positions have P&L calculated
        for pos in positions:
            assert pos.status in ["CLOSED_TP", "CLOSED_SL", "CLOSED_EXIT"], f"Position not properly closed: {pos.status}"
            assert pos.profit_loss is not None, "Position profit/loss not calculated"
            assert pos.profit_loss_percent is not None, "Position profit/loss percentage not calculated"

if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__]) 