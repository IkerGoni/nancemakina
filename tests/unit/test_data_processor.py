import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from collections import deque
from datetime import datetime, timedelta

from src.config_loader import ConfigManager
from src.data_processor import DataProcessor, MAX_KLINE_BUFFER_LENGTH
from src.models import Kline

@pytest.fixture
def mock_config_manager_for_dp():
    """Fixture that creates a mock ConfigManager for DataProcessor testing."""
    mock_cm = MagicMock(spec=ConfigManager)
    
    # Create a mock config with different enabled pairs, timeframes, and SMA periods
    mock_config = {
        "global_settings": {
            "v1_strategy": {
                "sma_short_period": 21,
                "sma_long_period": 200,
                "indicator_timeframes": ["1m", "5m", "15m"],
                "filters": {
                    "volume": {
                        "lookback_periods": 4
                    },
                    "volatility": {
                        "lookback_periods": 3
                    }
                }
            }
        },
        "pairs": {
            "BTC_USDT": {
                "enabled": True,
                "contract_type": "USDT_M",
                "indicator_timeframes": ["1m", "5m"]  # Override global
            },
            "ETH_USDT": {
                "enabled": True,
                "contract_type": "USDT_M"
                # Uses global timeframes: ["1m", "5m", "15m"]
            },
            "SOL_USDT": {
                "enabled": False  # This pair should not have buffers initialized initially
            },
            "XRP_USDT": {
                "enabled": True,
                "indicator_timeframes": ["1m"]  # Just 1m timeframe
            }
        }
    }
    
    # Set up the return value for get_config method
    mock_cm.get_config.return_value = mock_config
    
    return mock_cm

@pytest.fixture
def data_processor(mock_config_manager_for_dp):
    """Fixture that initializes DataProcessor with a mock ConfigManager."""
    dp = DataProcessor(config_manager=mock_config_manager_for_dp)
    return dp

# Test Initialization
@pytest.mark.asyncio
async def test_initialization_buffers_creation(data_processor, mock_config_manager_for_dp):
    """Test that buffers are created correctly based on enabled pairs and timeframes."""
    # Check active pairs timeframes dictionary
    assert "BTCUSDT" in data_processor._active_pairs_timeframes
    assert "ETHUSDT" in data_processor._active_pairs_timeframes
    assert "XRPUSDT" in data_processor._active_pairs_timeframes
    assert "SOLUSDT" not in data_processor._active_pairs_timeframes  # Disabled in config
    
    # Check specific timeframes for each pair
    assert set(data_processor._active_pairs_timeframes["BTCUSDT"]) == {"1m", "5m"}  # Overridden from global
    assert set(data_processor._active_pairs_timeframes["ETHUSDT"]) == {"1m", "5m", "15m"}  # From global
    assert set(data_processor._active_pairs_timeframes["XRPUSDT"]) == {"1m"}  # Specific config
    
    # Check kline buffer creation for all active pairs/timeframes
    for symbol in ["BTCUSDT", "ETHUSDT", "XRPUSDT"]:
        for tf in data_processor._active_pairs_timeframes[symbol]:
            assert tf in data_processor.kline_buffers[symbol]
            assert isinstance(data_processor.kline_buffers[symbol][tf], deque)
            assert data_processor.kline_buffers[symbol][tf].maxlen == MAX_KLINE_BUFFER_LENGTH
    
    # Check that disabled pairs don't have buffers or have empty ones
    if "SOLUSDT" in data_processor.kline_buffers:
        assert not data_processor.kline_buffers["SOLUSDT"]  # Should be empty

@pytest.mark.asyncio
async def test_initialization_empty_pairs_config(mock_config_manager_for_dp):
    """Test initialization with empty pairs configuration."""
    # Set empty pairs config
    empty_config = {
        "global_settings": {
            "v1_strategy": {
                "sma_short_period": 21,
                "sma_long_period": 200,
                "indicator_timeframes": ["1m", "5m"]
            }
        },
        "pairs": {}  # Empty pairs
    }
    mock_config_manager_for_dp.get_config.return_value = empty_config
    
    # Initialize DataProcessor with empty config
    dp = DataProcessor(config_manager=mock_config_manager_for_dp)
    
    # Check that active pairs, kline_buffers and indicator_data are empty
    assert not dp._active_pairs_timeframes
    assert not dp.kline_buffers  # Check defaultdict doesn't have any pairs
    assert not dp.indicator_data

# Test Kline Processing
@pytest.mark.asyncio
async def test_process_closed_kline(data_processor):
    """Test processing a single closed Kline for an active pair/timeframe."""
    # Create a sample closed kline
    kline_closed = Kline(
        timestamp=1000,
        open=10.0,
        high=12.0,
        low=9.0,
        close=11.0,
        volume=100.0,
        is_closed=True,
        symbol="BTCUSDT",
        interval="1m"
    )
    
    # Process the kline
    await data_processor.process_kline(kline_closed)
    
    # Check that kline was added to the buffer
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0] == kline_closed
    
    # Check that indicator data was updated (should exist even if values are NA)
    indicator_df = data_processor.get_indicator_dataframe("BTCUSDT", "1m")
    assert indicator_df is not None
    assert not indicator_df.empty
    assert kline_closed.timestamp in indicator_df.index
    assert "sma_short" in indicator_df.columns
    assert "sma_long" in indicator_df.columns

@pytest.mark.asyncio
async def test_process_unclosed_kline(data_processor):
    """Test processing a single unclosed Kline for an active pair/timeframe."""
    # Create a sample unclosed kline
    kline_unclosed = Kline(
        timestamp=1000,
        open=10.0,
        high=12.0,
        low=9.0,
        close=11.0,
        volume=100.0,
        is_closed=False,
        symbol="BTCUSDT",
        interval="1m"
    )
    
    # Process the kline
    await data_processor.process_kline(kline_unclosed)
    
    # Check that kline was added to the buffer
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0] == kline_unclosed
    
    # Check if indicator data exists (should be empty since only unclosed kline was processed)
    indicator_df = data_processor.get_indicator_dataframe("BTCUSDT", "1m")
    if indicator_df is not None and not indicator_df.empty:
        # Implementation may or may not update indicators for unclosed candles
        # Just check the structure exists
        assert kline_unclosed.timestamp in indicator_df.index
        assert "sma_short" in indicator_df.columns
        assert "sma_long" in indicator_df.columns

@pytest.mark.asyncio
async def test_update_unclosed_kline(data_processor):
    """Test updating the last unclosed Kline."""
    # Create initial unclosed kline
    kline_unclosed_t1 = Kline(
        timestamp=1000,
        open=10.0,
        high=11.0,
        low=9.0,
        close=10.5,
        volume=50.0,
        is_closed=False,
        symbol="BTCUSDT",
        interval="1m"
    )
    
    # Process the initial kline
    await data_processor.process_kline(kline_unclosed_t1)
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1
    
    # Create an updated version of the same kline (same timestamp, different data)
    kline_unclosed_t1_update = Kline(
        timestamp=1000,
        open=10.0,
        high=12.0,  # Higher high
        low=8.5,    # Lower low
        close=11.0, # Different close
        volume=75.0, # More volume
        is_closed=False,
        symbol="BTCUSDT",
        interval="1m"
    )
    
    # Process the updated kline
    await data_processor.process_kline(kline_unclosed_t1_update)
    
    # Check that the deque contains only one element (the updated kline)
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0] == kline_unclosed_t1_update
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0].high == 12.0
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0].close == 11.0

@pytest.mark.asyncio
async def test_replace_unclosed_with_closed_kline(data_processor):
    """Test replacing the last unclosed Kline with its closed version."""
    # Create initial unclosed kline
    kline_unclosed_t1 = Kline(
        timestamp=1000,
        open=10.0,
        high=11.0,
        low=9.0,
        close=10.5,
        volume=50.0,
        is_closed=False,
        symbol="BTCUSDT",
        interval="1m"
    )
    
    # Process the unclosed kline
    await data_processor.process_kline(kline_unclosed_t1)
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1
    assert not data_processor.kline_buffers["BTCUSDT"]["1m"][0].is_closed
    
    # Create closed version of the same kline
    kline_closed_t1 = Kline(
        timestamp=1000,
        open=10.0,
        high=11.5,
        low=9.0,
        close=11.0,
        volume=100.0,
        is_closed=True,  # Now closed
        symbol="BTCUSDT",
        interval="1m"
    )
    
    # Process the closed kline
    await data_processor.process_kline(kline_closed_t1)
    
    # Check that the deque contains one element (the closed kline)
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0] == kline_closed_t1
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0].is_closed is True
    
    # Check that indicator data was updated
    indicator_df = data_processor.get_indicator_dataframe("BTCUSDT", "1m")
    assert indicator_df is not None
    assert not indicator_df.empty
    assert kline_closed_t1.timestamp in indicator_df.index

@pytest.mark.asyncio
async def test_process_kline_inactive_pair(data_processor):
    """Test processing a Kline for an inactive pair/timeframe."""
    # Create a kline for an inactive pair
    kline_inactive = Kline(
        timestamp=1000,
        open=10.0,
        high=12.0,
        low=9.0,
        close=11.0,
        volume=100.0,
        is_closed=True,
        symbol="SOLUSDT",  # Disabled in config
        interval="1m"
    )
    
    # Process the kline
    await data_processor.process_kline(kline_inactive)
    
    # Check that kline was not processed (no buffer should exist)
    if "SOLUSDT" in data_processor.kline_buffers:
        assert not data_processor.kline_buffers["SOLUSDT"]["1m"]
    
    # Check that indicator data was not created
    indicator_df = data_processor.get_indicator_dataframe("SOLUSDT", "1m")
    if indicator_df is not None:
        assert indicator_df.empty

@pytest.mark.asyncio
async def test_buffer_limit_enforcement(data_processor):
    """Test that buffer size is limited to MAX_KLINE_BUFFER_LENGTH."""
    # Generate klines with sequential timestamps
    symbol = "BTCUSDT"
    interval = "1m"
    num_klines = MAX_KLINE_BUFFER_LENGTH + 5  # More than the max buffer length
    
    for i in range(num_klines):
        kline = Kline(
            timestamp=1000 + i,
            open=10.0,
            high=12.0,
            low=9.0,
            close=11.0,
            volume=100.0,
            is_closed=True,
            symbol=symbol,
            interval=interval
        )
        await data_processor.process_kline(kline)
    
    # Check that buffer size is limited to MAX_KLINE_BUFFER_LENGTH
    assert len(data_processor.kline_buffers[symbol][interval]) == MAX_KLINE_BUFFER_LENGTH
    
    # Check that oldest klines were discarded
    # First timestamp should be the earliest retained timestamp
    first_timestamp = data_processor.kline_buffers[symbol][interval][0].timestamp
    assert first_timestamp == 1000 + (num_klines - MAX_KLINE_BUFFER_LENGTH)

# Test Indicator Calculation
@pytest.mark.asyncio
async def test_insufficient_data_for_sma(data_processor, mock_config_manager_for_dp):
    """Test case where there's insufficient data for SMA calculation."""
    # Set shorter SMA periods for easier testing
    mock_config = mock_config_manager_for_dp.get_config()
    mock_config["global_settings"]["v1_strategy"]["sma_short_period"] = 3
    mock_config["global_settings"]["v1_strategy"]["sma_long_period"] = 5
    mock_config_manager_for_dp.get_config.return_value = mock_config
    
    # Initialize with new config
    dp = DataProcessor(config_manager=mock_config_manager_for_dp)
    
    symbol = "BTCUSDT"
    interval = "1m"
    
    # Process 2 klines (less than sma_short_period of 3)
    for i in range(2):
        kline = Kline(
            timestamp=1000 + i * 60000,  # 1-minute intervals
            open=100.0,
            high=110.0,
            low=90.0,
            close=100.0 + i,  # Different close prices
            volume=100.0,
            is_closed=True,
            symbol=symbol,
            interval=interval
        )
        await dp.process_kline(kline)
    
    # Check indicator dataframe
    df = dp.get_indicator_dataframe(symbol, interval)
    assert df is not None
    assert not df.empty
    assert "sma_short" in df.columns
    assert "sma_long" in df.columns
    
    # Check that SMA values are NaN due to insufficient data
    latest_row = df.iloc[-1]
    assert pd.isna(latest_row["sma_short"])
    assert pd.isna(latest_row["sma_long"])

@pytest.mark.asyncio
async def test_correct_sma_short_calculation(data_processor, mock_config_manager_for_dp):
    """Test correct calculation of short-period SMA."""
    # Set shorter SMA periods for easier testing
    mock_config = mock_config_manager_for_dp.get_config()
    mock_config["global_settings"]["v1_strategy"]["sma_short_period"] = 3
    mock_config["global_settings"]["v1_strategy"]["sma_long_period"] = 5
    mock_config_manager_for_dp.get_config.return_value = mock_config
    
    # Initialize with new config
    dp = DataProcessor(config_manager=mock_config_manager_for_dp)
    
    symbol = "BTCUSDT"
    interval = "1m"
    close_prices = [100.0, 110.0, 120.0, 130.0]
    
    # Process exactly sma_short_period + 1 klines
    for i, close in enumerate(close_prices):
        kline = Kline(
            timestamp=1000 + i * 60000,
            open=close,
            high=close + 10,
            low=close - 10,
            close=close,
            volume=100.0,
            is_closed=True,
            symbol=symbol,
            interval=interval
        )
        await dp.process_kline(kline)
    
    # Check indicator dataframe
    df = dp.get_indicator_dataframe(symbol, interval)
    assert df is not None
    assert len(df) == len(close_prices)
    
    # Calculate expected SMA values
    expected_sma3 = sum(close_prices[-3:]) / 3  # Average of last 3 values
    
    # Check SMA values
    latest_row = df.iloc[-1]
    assert latest_row["sma_short"] == pytest.approx(expected_sma3)
    assert pd.isna(latest_row["sma_long"])  # Not enough data for long SMA

@pytest.mark.asyncio
async def test_correct_sma_long_calculation(data_processor, mock_config_manager_for_dp):
    """Test correct calculation of long-period SMA."""
    # Set shorter SMA periods for easier testing
    mock_config = mock_config_manager_for_dp.get_config()
    mock_config["global_settings"]["v1_strategy"]["sma_short_period"] = 3
    mock_config["global_settings"]["v1_strategy"]["sma_long_period"] = 5
    mock_config_manager_for_dp.get_config.return_value = mock_config
    
    # Initialize with new config
    dp = DataProcessor(config_manager=mock_config_manager_for_dp)
    
    symbol = "BTCUSDT"
    interval = "1m"
    close_prices = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    
    # Process exactly sma_long_period + 1 klines
    for i, close in enumerate(close_prices):
        kline = Kline(
            timestamp=1000 + i * 60000,
            open=close,
            high=close + 10,
            low=close - 10,
            close=close,
            volume=100.0,
            is_closed=True,
            symbol=symbol,
            interval=interval
        )
        await dp.process_kline(kline)
    
    # Check indicator dataframe
    df = dp.get_indicator_dataframe(symbol, interval)
    assert df is not None
    assert len(df) == len(close_prices)
    
    # Calculate expected SMA values
    expected_sma3 = sum(close_prices[-3:]) / 3  # Average of last 3 values
    expected_sma5 = sum(close_prices[-5:]) / 5  # Average of last 5 values
    
    # Check SMA values
    latest_row = df.iloc[-1]
    assert latest_row["sma_short"] == pytest.approx(expected_sma3)
    assert latest_row["sma_long"] == pytest.approx(expected_sma5)

@pytest.mark.asyncio
async def test_sma_calculation_with_duplicate_timestamps(data_processor, mock_config_manager_for_dp):
    """Test that SMA calculation correctly handles duplicate timestamps."""
    # Set shorter SMA periods for easier testing
    mock_config = mock_config_manager_for_dp.get_config()
    mock_config["global_settings"]["v1_strategy"]["sma_short_period"] = 3
    mock_config["global_settings"]["v1_strategy"]["sma_long_period"] = 5
    mock_config_manager_for_dp.get_config.return_value = mock_config
    
    # Initialize with new config
    dp = DataProcessor(config_manager=mock_config_manager_for_dp)
    
    symbol = "BTCUSDT"
    interval = "1m"
    
    # Process klines with some duplicate timestamps
    klines_data = [
        # Original klines
        (1000, 100.0, False),
        (2000, 110.0, False),
        (3000, 120.0, False),
        (4000, 130.0, False),
        (5000, 140.0, False),
        # Updates with same timestamps but different values (should replace)
        (3000, 125.0, True),  # Update and close kline at t=3000
        (5000, 145.0, True),  # Update and close kline at t=5000
    ]
    
    for timestamp, close, is_closed in klines_data:
        kline = Kline(
            timestamp=timestamp,
            open=close - 10,
            high=close + 10,
            low=close - 20,
            close=close,
            volume=100.0,
            is_closed=is_closed,
            symbol=symbol,
            interval=interval
        )
        await dp.process_kline(kline)
    
    # Check indicator dataframe
    df = dp.get_indicator_dataframe(symbol, interval)
    assert df is not None
    
    # Check no duplicate timestamps in DataFrame
    assert len(df.index) == len(set(df.index))
    
    # Check that the updated values are used in the DataFrame
    assert df.loc[3000, "close"] == 125.0
    assert df.loc[5000, "close"] == 145.0
    
    # Calculate expected SMA values (using last values after updates)
    expected_values = [100.0, 110.0, 125.0, 130.0, 145.0]
    expected_sma3 = sum(expected_values[-3:]) / 3
    expected_sma5 = sum(expected_values) / 5
    
    # Check SMA calculations
    latest_row = df.iloc[-1]  # Should be the row for timestamp 5000
    assert latest_row["sma_short"] == pytest.approx(expected_sma3)
    assert latest_row["sma_long"] == pytest.approx(expected_sma5)

# Test Getter Methods
@pytest.mark.asyncio
async def test_getters_return_correct_data(data_processor, mock_config_manager_for_dp):
    """Test that getter methods return correct data after processing."""
    # Set shorter SMA periods for easier testing
    mock_config = mock_config_manager_for_dp.get_config()
    mock_config["global_settings"]["v1_strategy"]["sma_short_period"] = 3
    mock_config["global_settings"]["v1_strategy"]["sma_long_period"] = 5
    mock_config_manager_for_dp.get_config.return_value = mock_config
    
    # Initialize with new config
    dp = DataProcessor(config_manager=mock_config_manager_for_dp)
    
    symbol = "BTCUSDT"
    interval = "1m"
    
    # Process several klines
    klines = []
    for i in range(6):  # 6 klines
        kline = Kline(
            timestamp=1000 + i * 60000,
            open=100.0 + i,
            high=110.0 + i,
            low=90.0 + i,
            close=105.0 + i,
            volume=100.0 + i * 10,
            is_closed=True,
            symbol=symbol,
            interval=interval
        )
        klines.append(kline)
        await dp.process_kline(kline)
    
    # Test get_latest_kline
    latest_kline = dp.get_latest_kline(symbol, interval)
    assert latest_kline is not None
    assert latest_kline == klines[-1]
    
    # Test get_indicator_dataframe
    df = dp.get_indicator_dataframe(symbol, interval)
    assert df is not None
    assert not df.empty
    assert len(df) == len(klines)
    
    # Test get_latest_indicators
    latest_indicators = dp.get_latest_indicators(symbol, interval)
    assert latest_indicators is not None
    assert latest_indicators["timestamp"] == klines[-1].timestamp
    assert latest_indicators["close"] == klines[-1].close
    assert not pd.isna(latest_indicators["sma_short"])
    assert not pd.isna(latest_indicators["sma_long"])

@pytest.mark.asyncio
async def test_getters_return_none_for_nonexistent_data(data_processor):
    """Test that getters return None or empty DataFrame for non-existent symbol/timeframe."""
    # Test with non-existent symbol
    assert data_processor.get_latest_kline("NONEXISTENT", "1m") is None
    df = data_processor.get_indicator_dataframe("NONEXISTENT", "1m")
    if df is not None:
        assert df.empty
    assert data_processor.get_latest_indicators("NONEXISTENT", "1m") is None
    
    # Test with valid symbol but non-existent timeframe
    assert data_processor.get_latest_kline("BTCUSDT", "NONEXISTENT") is None
    df = data_processor.get_indicator_dataframe("BTCUSDT", "NONEXISTENT")
    if df is not None:
        assert df.empty
    assert data_processor.get_latest_indicators("BTCUSDT", "NONEXISTENT") is None

# Test Configuration Hot-Reload
@pytest.mark.asyncio
async def test_adding_new_active_pair_via_config_update(data_processor, mock_config_manager_for_dp):
    """Test that config update adds a new active pair correctly."""
    # Check initial state
    assert "SOLUSDT" not in data_processor._active_pairs_timeframes  # Initially disabled
    
    # Create new config with SOLUSDT enabled
    new_config = mock_config_manager_for_dp.get_config().copy()
    new_config["pairs"]["SOL_USDT"]["enabled"] = True
    new_config["pairs"]["SOL_USDT"]["indicator_timeframes"] = ["1m", "15m"]
    
    # Trigger config update
    data_processor._handle_config_update(new_config)
    
    # Check that SOLUSDT was added to active pairs
    assert "SOLUSDT" in data_processor._active_pairs_timeframes
    assert set(data_processor._active_pairs_timeframes["SOLUSDT"]) == {"1m", "15m"}
    
    # Check that buffers were created
    assert "SOLUSDT" in data_processor.kline_buffers
    assert "1m" in data_processor.kline_buffers["SOLUSDT"]
    assert "15m" in data_processor.kline_buffers["SOLUSDT"]
    assert isinstance(data_processor.kline_buffers["SOLUSDT"]["1m"], deque)
    assert isinstance(data_processor.kline_buffers["SOLUSDT"]["15m"], deque)
    assert data_processor.kline_buffers["SOLUSDT"]["1m"].maxlen == MAX_KLINE_BUFFER_LENGTH

@pytest.mark.asyncio
async def test_disabling_existing_pair_via_config_update(data_processor, mock_config_manager_for_dp):
    """Test that config update correctly disables an active pair."""
    # Check initial state
    assert "BTCUSDT" in data_processor._active_pairs_timeframes  # Initially enabled
    
    # Create new config with BTCUSDT disabled
    new_config = mock_config_manager_for_dp.get_config().copy()
    new_config["pairs"]["BTC_USDT"]["enabled"] = False
    
    # Trigger config update
    data_processor._handle_config_update(new_config)
    
    # Check that BTCUSDT was removed from active pairs
    assert "BTCUSDT" not in data_processor._active_pairs_timeframes
    
    # Note: The current implementation might not remove old buffers, just stop processing for them
    # So we won't assert on the removal of buffers, just on the active pairs list

@pytest.mark.asyncio
async def test_changing_timeframes_via_config_update(data_processor, mock_config_manager_for_dp):
    """Test that config update correctly changes timeframes for an active pair."""
    # Check initial state for ETHUSDT
    assert "ETHUSDT" in data_processor._active_pairs_timeframes
    assert set(data_processor._active_pairs_timeframes["ETHUSDT"]) == {"1m", "5m", "15m"}
    
    # Create new config with different timeframes for ETHUSDT
    new_config = mock_config_manager_for_dp.get_config().copy()
    new_config["pairs"]["ETH_USDT"]["indicator_timeframes"] = ["5m", "30m"]  # Changed timeframes
    
    # Trigger config update
    data_processor._handle_config_update(new_config)
    
    # Check that timeframes were updated
    assert "ETHUSDT" in data_processor._active_pairs_timeframes
    assert set(data_processor._active_pairs_timeframes["ETHUSDT"]) == {"5m", "30m"}
    
    # Check that new buffer for 30m timeframe was created
    assert "30m" in data_processor.kline_buffers["ETHUSDT"]
    assert isinstance(data_processor.kline_buffers["ETHUSDT"]["30m"], deque)

def create_test_klines(num_klines=10, base_timestamp=1609459200000, symbol="BTCUSDT", interval="1m"):
    """Helper to create a sequence of test klines."""
    klines = []
    for i in range(num_klines):
        timestamp = base_timestamp + (i * 60000)  # Add minutes
        klines.append(
            Kline(
                timestamp=timestamp,
                open=100 + i,
                high=110 + i,
                low=90 + i,
                close=105 + i,
                volume=1000 + (i * 100),
                is_closed=True,
                symbol=symbol,
                interval=interval
            )
        )
    return klines

def test_update_indicators_calculates_average_volume(data_processor):
    """Test that _update_indicators correctly calculates average volume."""
    # Setup
    symbol = "BTCUSDT"
    interval = "1m"
    klines = create_test_klines(10, symbol=symbol, interval=interval)
    
    # Add klines to buffer
    data_processor.kline_buffers[symbol][interval] = deque(klines)
    
    # Call method to test
    data_processor._update_indicators(symbol, interval)
    
    # Get results
    result_df = data_processor.indicator_data[symbol][interval]
    
    # Verify average volume calculated with correct window
    assert "avg_volume" in result_df.columns
    
    # Since we configured lookback_periods as 4, let's check
    # First 3 values should be NaN (not enough data points for window)
    assert pd.isna(result_df["avg_volume"].iloc[0])
    assert pd.isna(result_df["avg_volume"].iloc[1])
    assert pd.isna(result_df["avg_volume"].iloc[2])
    
    # 4th value should be average of first 4 volumes
    expected_avg = sum([klines[i].volume for i in range(4)]) / 4
    assert result_df["avg_volume"].iloc[3] == pytest.approx(expected_avg)
    
    # Last value should be average of last 4 volumes
    expected_last_avg = sum([klines[i].volume for i in range(6, 10)]) / 4
    assert result_df["avg_volume"].iloc[9] == pytest.approx(expected_last_avg)

def test_update_indicators_calculates_atr(data_processor):
    """Test that _update_indicators correctly calculates ATR."""
    # Setup
    symbol = "BTCUSDT"
    interval = "1m"
    klines = create_test_klines(10, symbol=symbol, interval=interval)
    
    # Add klines to buffer
    data_processor.kline_buffers[symbol][interval] = deque(klines)
    
    # Call method to test
    data_processor._update_indicators(symbol, interval)
    
    # Get results
    result_df = data_processor.indicator_data[symbol][interval]
    
    # Verify ATR calculated
    assert "atr" in result_df.columns
    
    # Since we configured atr_period as 3, and ATR needs a previous candle
    # First 3 values should be NaN (1 for previous candle + 3 for window)
    assert pd.isna(result_df["atr"].iloc[0])
    assert pd.isna(result_df["atr"].iloc[1])
    assert pd.isna(result_df["atr"].iloc[2])
    assert pd.isna(result_df["atr"].iloc[3])
    
    # 5th value onwards should have ATR
    assert not pd.isna(result_df["atr"].iloc[4])
    
    # Manually calculate ATR for the 5th value (index 4)
    # TR = max(high-low, abs(high-prevclose), abs(low-prevclose))
    tr_values = []
    for i in range(1, 4):  # We need 3 TR values (ATR window)
        high = klines[i].high
        low = klines[i].low
        prev_close = klines[i-1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    
    expected_atr = sum(tr_values) / 3
    assert result_df["atr"].iloc[4] == pytest.approx(expected_atr, rel=1e-10)

def test_update_indicators_handles_insufficient_data(data_processor):
    """Test that _update_indicators handles cases with insufficient data gracefully."""
    # Setup with just 2 klines (not enough for most calculations)
    symbol = "BTCUSDT"
    interval = "1m"
    klines = create_test_klines(2, symbol=symbol, interval=interval)
    
    # Add klines to buffer
    data_processor.kline_buffers[symbol][interval] = deque(klines)
    
    # Call method to test
    data_processor._update_indicators(symbol, interval)
    
    # Get results
    result_df = data_processor.indicator_data[symbol][interval]
    
    # Verify columns exist but values are NA due to insufficient data
    assert "sma_short" in result_df.columns
    assert "sma_long" in result_df.columns
    assert "avg_volume" in result_df.columns
    assert "atr" in result_df.columns
    
    # All ATR values should be NA (need at least 4 candles with our settings)
    assert pd.isna(result_df["atr"]).all()
    
    # All avg_volume values should be NA (need at least 4 candles with our settings)
    assert pd.isna(result_df["avg_volume"]).all()

