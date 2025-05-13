import pytest
import asyncio
import pandas as pd
import time
from unittest.mock import MagicMock, AsyncMock, patch

from src.config_loader import ConfigManager
from src.data_processor import DataProcessor
from src.signal_engine import SignalEngineV1
from src.models import TradeSignal, TradeDirection, Kline

@pytest.fixture
def mock_config_manager_for_se():
    cm = MagicMock(spec=ConfigManager)
    cm.get_config.return_value = {
        "global_settings": {
            "v1_strategy": {
                "sma_short_period": 3,
                "sma_long_period": 5,
                "min_signal_interval_minutes": 15,
                "tp_sl_ratio": 2.0
            }
        },
        "pairs": {
            "BTC_USDT": {"enabled": True, "contract_type": "USDT_M"},
            "ETH_USDT": {
                "enabled": True, 
                "contract_type": "USDT_M",
                "min_signal_interval_minutes": 0  # For testing without wait
            },
            "SOL_USDT": {
                "enabled": True,
                "contract_type": "USDT_M",
                "tp_sl_ratio": 3.5  # Custom R:R ratio for testing
            }
        }
    }
    return cm

@pytest.fixture
def mock_data_processor_for_se():
    dp = MagicMock(spec=DataProcessor)
    return dp

@pytest.fixture
def signal_engine_v1(mock_config_manager_for_se, mock_data_processor_for_se):
    return SignalEngineV1(config_manager=mock_config_manager_for_se, data_processor=mock_data_processor_for_se)

# Helper to create a DataFrame for mock_data_processor
def create_mock_df(sma_short_prev, sma_long_prev, sma_short_curr, sma_long_curr, close_curr=100, low_lookback=[90,91,89], high_lookback=[110,109,111]):
    # Ensure lookback arrays are long enough for pivot logic (default pivot_lookback=30)
    # Create enough past data for pivot calculation
    total_rows = 32  # Total number of rows in the DataFrame (31 for past data + 1 for current)
    pivot_data_rows = 30  # Number of rows needed for pivot calculation (excluding the last 2 rows)
    
    # Ensure low_lookback and high_lookback are lists
    if not isinstance(low_lookback, list):
        low_lookback = [low_lookback]
    if not isinstance(high_lookback, list):
        high_lookback = [high_lookback]
    
    # Repeat the lookback arrays to make them at least pivot_data_rows long
    past_lows = (low_lookback * (pivot_data_rows // len(low_lookback) + 1))[:pivot_data_rows]
    past_highs = (high_lookback * (pivot_data_rows // len(high_lookback) + 1))[:pivot_data_rows]
    
    # Create data arrays of consistent length
    timestamps = list(range(1000, 1000 + total_rows * 1000, 1000))  # timestamps from 1000 to 32000, step 1000
    
    # Build arrays for all columns, ensuring they're all the same length
    opens = [close_curr - 1] * pivot_data_rows + [close_curr - 1, close_curr - 0.5]
    highs = past_highs + [max(high_lookback), close_curr + 2]
    lows = past_lows + [min(low_lookback), close_curr - 2]
    closes = [close_curr - 1] * pivot_data_rows + [close_curr - 1, close_curr]
    volumes = [100] * total_rows
    is_closed = [True] * total_rows
    
    # SMA values - past values, previous, and current
    sma_shorts = [sma_short_prev - 1] * pivot_data_rows + [sma_short_prev, sma_short_curr]
    sma_longs = [sma_long_prev - 1] * pivot_data_rows + [sma_long_prev, sma_long_curr]
    
    # Validate all arrays have the same length
    assert len(timestamps) == total_rows
    assert len(opens) == total_rows
    assert len(highs) == total_rows
    assert len(lows) == total_rows
    assert len(closes) == total_rows
    assert len(volumes) == total_rows
    assert len(is_closed) == total_rows
    assert len(sma_shorts) == total_rows
    assert len(sma_longs) == total_rows
    
    data = {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "is_closed": is_closed,
        "sma_short": sma_shorts,
        "sma_long": sma_longs,
    }
    
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df

# Helper to create a shorter DataFrame with specific values
def create_limited_df(rows=5, with_na=False):
    """Creates a small DataFrame with limited rows - useful for testing insufficient data scenarios"""
    data = {
        "timestamp": list(range(1000, 1000 + rows * 1000, 1000)),
        "open": [99] * rows,
        "high": [101] * rows,
        "low": [98] * rows,
        "close": [100] * rows,
        "volume": [100] * rows,
        "is_closed": [True] * rows,
        "sma_short": [99] * rows,
        "sma_long": [100] * rows
    }
    
    # Set some values to NA if requested
    if with_na:
        data["sma_short"][-1] = pd.NA
    
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df

# 1. Test Signal Generation Logic
@pytest.mark.asyncio
async def test_se_no_signal_if_not_enough_data(signal_engine_v1, mock_data_processor_for_se):
    # Test with empty DataFrame
    mock_data_processor_for_se.get_indicator_dataframe.return_value = pd.DataFrame()  # Empty DF
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

    # Test with None DataFrame
    mock_data_processor_for_se.get_indicator_dataframe.return_value = None
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

    # Test with only one row
    mock_data_processor_for_se.get_indicator_dataframe.return_value = create_mock_df(10,20,11,19)[:1]  # Only one row
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

@pytest.mark.asyncio
async def test_se_no_signal_if_sma_na(signal_engine_v1, mock_data_processor_for_se):
    # Test with NA in latest SMA short
    df_with_na = create_mock_df(10, 20, 11, 19)
    df_with_na.loc[df_with_na.index[-1], "sma_short"] = pd.NA
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df_with_na
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

    # Test with NA in latest SMA long
    df_with_na = create_mock_df(10, 20, 11, 19)
    df_with_na.loc[df_with_na.index[-1], "sma_long"] = pd.NA
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df_with_na
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

    # Test with NA in previous SMA short
    df_with_na = create_mock_df(10, 20, 11, 19)
    df_with_na.loc[df_with_na.index[-2], "sma_short"] = pd.NA
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df_with_na
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

    # Test with NA in previous SMA long
    df_with_na = create_mock_df(10, 20, 11, 19)
    df_with_na.loc[df_with_na.index[-2], "sma_long"] = pd.NA
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df_with_na
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

@pytest.mark.asyncio
async def test_se_no_signal_if_no_crossover(signal_engine_v1, mock_data_processor_for_se):
    # Test when short > long, and still short > long (no crossover)
    mock_df = create_mock_df(sma_short_prev=101, sma_long_prev=100, sma_short_curr=102, sma_long_curr=99)
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is None

    # Test when short < long, and still short < long (no crossover)
    mock_df = create_mock_df(sma_short_prev=99, sma_long_prev=100, sma_short_curr=98, sma_long_curr=101)
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is None

    # Test when short = long, and still short = long (no crossover)
    mock_df = create_mock_df(sma_short_prev=100, sma_long_prev=100, sma_short_curr=100, sma_long_curr=100)
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is None

@pytest.mark.asyncio
async def test_se_long_signal_sma_crossover(signal_engine_v1, mock_data_processor_for_se):
    # prev: short <= long, curr: short > long
    mock_df = create_mock_df(sma_short_prev=99, sma_long_prev=100, sma_short_curr=101, sma_long_curr=100, close_curr=100.5, low_lookback=[90,88,89])
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")  # ETH_USDT has min_interval 0
    assert signal is not None
    assert signal.direction == TradeDirection.LONG
    assert signal.symbol == "ETHUSDT"
    assert signal.config_symbol == "ETH_USDT"
    assert signal.contract_type == "USDT_M"
    assert signal.entry_price == 100.5  # latest close
    assert signal.stop_loss_price == 88  # min of low_lookback
    # Risk = 100.5 - 88 = 12.5. TP = 100.5 + (12.5 * 2.0) = 100.5 + 25 = 125.5
    assert signal.take_profit_price == 125.5
    assert signal.strategy_name == "V1_SMA_Crossover"
    assert signal.signal_kline is not None
    assert signal.signal_kline.close == 100.5
    assert signal.signal_kline.timestamp == mock_df.index[-1]
    assert signal.details is not None
    assert signal.details["sma_short_at_signal"] == 101
    assert signal.details["sma_long_at_signal"] == 100
    assert signal.details["sma_short_previous"] == 99
    assert signal.details["sma_long_previous"] == 100
    assert signal.details["pivot_used_for_sl"] == 88

@pytest.mark.asyncio
async def test_se_short_signal_sma_crossover(signal_engine_v1, mock_data_processor_for_se):
    # prev: short >= long, curr: short < long
    mock_df = create_mock_df(sma_short_prev=101, sma_long_prev=100, sma_short_curr=99, sma_long_curr=100, close_curr=99.5, high_lookback=[110,112,111])
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df

    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is not None
    assert signal.direction == TradeDirection.SHORT
    assert signal.symbol == "ETHUSDT"
    assert signal.config_symbol == "ETH_USDT"
    assert signal.entry_price == 99.5
    assert signal.stop_loss_price == 112  # max of high_lookback
    # Risk = 112 - 99.5 = 12.5. TP = 99.5 - (12.5 * 2.0) = 99.5 - 25 = 74.5
    assert signal.take_profit_price == 74.5
    assert signal.signal_kline is not None
    assert signal.signal_kline.close == 99.5
    assert signal.details is not None
    assert signal.details["sma_short_at_signal"] == 99
    assert signal.details["sma_long_at_signal"] == 100
    assert signal.details["sma_short_previous"] == 101
    assert signal.details["sma_long_previous"] == 100
    assert signal.details["pivot_used_for_sl"] == 112

# 2. Test SL/TP Calculation
@pytest.mark.asyncio
async def test_se_sl_tp_calculation_long(signal_engine_v1, mock_data_processor_for_se):
    # Test a more complex case with specific pivot lows
    low_lookback = [90, 88, 89, 85, 87]  # The pivot low should be 85
    mock_df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5, 
        low_lookback=low_lookback
    )
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is not None
    assert signal.direction == TradeDirection.LONG
    assert signal.stop_loss_price == 85  # min of low_lookback
    # Risk = 100.5 - 85 = 15.5. TP = 100.5 + (15.5 * 2.0) = 100.5 + 31 = 131.5
    assert signal.take_profit_price == 131.5

@pytest.mark.asyncio
async def test_se_sl_tp_calculation_short(signal_engine_v1, mock_data_processor_for_se):
    # Test a more complex case with specific pivot highs
    high_lookback = [110, 112, 115, 111, 113]  # The pivot high should be 115
    mock_df = create_mock_df(
        sma_short_prev=101, sma_long_prev=100, 
        sma_short_curr=99, sma_long_curr=100, 
        close_curr=99.5, 
        high_lookback=high_lookback
    )
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is not None
    assert signal.direction == TradeDirection.SHORT
    assert signal.stop_loss_price == 115  # max of high_lookback
    # Risk = 115 - 99.5 = 15.5. TP = 99.5 - (15.5 * 2.0) = 99.5 - 31 = 68.5
    assert signal.take_profit_price == 68.5

@pytest.mark.asyncio
async def test_se_custom_tp_sl_ratio(signal_engine_v1, mock_data_processor_for_se):
    # Test TP calculation with a custom R:R ratio for SOL_USDT (3.5)
    mock_df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5, 
        low_lookback=[90, 88, 89]
    )
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    
    signal = await signal_engine_v1.check_signal("SOLUSDT", "SOL_USDT")
    assert signal is not None
    assert signal.direction == TradeDirection.LONG
    assert signal.stop_loss_price == 88
    # Risk = 100.5 - 88 = 12.5. TP = 100.5 + (12.5 * 3.5) = 100.5 + 43.75 = 144.25
    assert signal.take_profit_price == 144.25

@pytest.mark.asyncio
async def test_se_no_signal_if_insufficient_pivot_data(signal_engine_v1, mock_data_processor_for_se):
    # Create a DataFrame that's too small for proper pivot calculation (< 3 rows)
    small_df = create_limited_df(rows=2)
    mock_data_processor_for_se.get_indicator_dataframe.return_value = small_df
    
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is None

@pytest.mark.asyncio
async def test_se_sl_tp_calculation_invalid_risk(signal_engine_v1, mock_data_processor_for_se):
    # SL is worse than entry for LONG (pivot low > entry price)
    mock_df_bad_sl_long = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=90, 
        low_lookback=[95, 96, 97]  # min low is 95, which is > entry price 90
    )
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df_bad_sl_long
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is None

    # SL is worse than entry for SHORT (pivot high < entry price)
    mock_df_bad_sl_short = create_mock_df(
        sma_short_prev=101, sma_long_prev=100, 
        sma_short_curr=99, sma_long_curr=100, 
        close_curr=110, 
        high_lookback=[105, 106, 104]  # max high is 106, which is < entry price 110
    )
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df_bad_sl_short
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is None

@pytest.mark.asyncio
async def test_se_long_signal_signal_kline_content(signal_engine_v1, mock_data_processor_for_se):
    # Test that signal_kline contains the correct data from the last DataFrame row
    mock_df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5
    )
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is not None
    assert signal.signal_kline is not None
    assert signal.signal_kline.timestamp == mock_df.index[-1]
    assert signal.signal_kline.open == mock_df.iloc[-1]["open"]
    assert signal.signal_kline.high == mock_df.iloc[-1]["high"]
    assert signal.signal_kline.low == mock_df.iloc[-1]["low"]
    assert signal.signal_kline.close == mock_df.iloc[-1]["close"]
    assert signal.signal_kline.volume == mock_df.iloc[-1]["volume"]
    assert signal.signal_kline.is_closed == mock_df.iloc[-1]["is_closed"]
    assert signal.signal_kline.symbol == "ETHUSDT"
    assert signal.signal_kline.interval == "1m"  # The default interval used by SignalEngineV1

# 3. Test Filters
@pytest.mark.asyncio
async def test_se_signal_interval_filter(signal_engine_v1, mock_data_processor_for_se):
    # BTC_USDT has min_interval_minutes = 15
    mock_df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5
    )
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df

    # First signal should pass
    signal1 = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal1 is not None
    assert signal_engine_v1.last_signal_time["BTCUSDT"] > 0

    # Second signal immediately after should be filtered
    signal2 = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal2 is None

    # Simulate time passing (less than interval)
    signal_engine_v1.last_signal_time["BTCUSDT"] = time.time() - (10 * 60)  # 10 minutes ago
    signal3 = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal3 is None 

    # Simulate time passing (more than interval)
    signal_engine_v1.last_signal_time["BTCUSDT"] = time.time() - (20 * 60)  # 20 minutes ago
    signal4 = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal4 is not None

@pytest.mark.asyncio
async def test_se_signal_interval_is_pair_specific(signal_engine_v1, mock_data_processor_for_se):
    # Setup: BTC_USDT has interval=15, ETH_USDT has interval=0
    mock_df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5
    )
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    
    # Generate signals for both pairs
    btc_signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    eth_signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    
    # Both should work the first time
    assert btc_signal is not None
    assert eth_signal is not None
    
    # Try immediate second signals
    btc_signal2 = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    eth_signal2 = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    
    # BTC should be filtered, ETH should still generate (interval=0)
    assert btc_signal2 is None
    assert eth_signal2 is not None

@pytest.mark.asyncio
async def test_se_volume_filter_passes_with_sufficient_volume(signal_engine_v1, mock_data_processor_for_se):
    """Test that volume filter passes when volume exceeds threshold."""
    # Create DataFrame with current volume > avg_volume * threshold
    df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5
    )
    # Add volume data to test the filter
    df["volume"] = 2000  # Current volume
    df["avg_volume"] = 1000  # Average volume
    
    # Configure the filter to require 1.5x average volume
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df
    
    # Enable volume filter in config
    mock_config = signal_engine_v1.config_manager.get_config.return_value
    mock_config["pairs"]["ETH_USDT"]["filters"] = {
        "volume": {"enabled": True, "min_threshold": 1.5}
    }
    
    # Test signal generation
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    
    # Signal should be generated as volume filter passes (2000 > 1000 * 1.5)
    assert signal is not None
    assert signal.symbol == "ETHUSDT"
    assert signal.direction == TradeDirection.LONG

@pytest.mark.asyncio
async def test_se_volume_filter_blocks_with_insufficient_volume(signal_engine_v1, mock_data_processor_for_se):
    """Test that volume filter blocks signals when volume is insufficient."""
    # Create DataFrame with current volume < avg_volume * threshold
    df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5
    )
    # Add volume data to test the filter
    df["volume"] = 1200  # Current volume
    df["avg_volume"] = 1000  # Average volume
    
    # Configure the filter to require 1.5x average volume
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df
    
    # Enable volume filter in config
    mock_config = signal_engine_v1.config_manager.get_config.return_value
    mock_config["pairs"]["ETH_USDT"]["filters"] = {
        "volume": {"enabled": True, "min_threshold": 1.5}
    }
    
    # Test signal generation
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    
    # Signal should be blocked as volume filter fails (1200 < 1000 * 1.5)
    assert signal is None

@pytest.mark.asyncio
async def test_se_volume_filter_handles_missing_data(signal_engine_v1, mock_data_processor_for_se):
    """Test that volume filter gracefully handles missing avg_volume data."""
    # Create DataFrame with missing avg_volume
    df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5
    )
    # Add volume but set avg_volume to NA
    df["volume"] = 1000
    df["avg_volume"] = pd.NA
    
    # Configure the filter
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df
    
    # Enable volume filter in config
    mock_config = signal_engine_v1.config_manager.get_config.return_value
    mock_config["pairs"]["ETH_USDT"]["filters"] = {
        "volume": {"enabled": True, "min_threshold": 1.5}
    }
    
    # Test signal generation
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    
    # Signal should still be generated despite missing avg_volume (graceful handling)
    assert signal is not None
    assert signal.symbol == "ETHUSDT"
    assert signal.direction == TradeDirection.LONG

@pytest.mark.asyncio
async def test_se_volatility_filter_passes_within_range(signal_engine_v1, mock_data_processor_for_se):
    """Test that volatility filter passes when ATR is within desired range."""
    # Create DataFrame with ATR in acceptable range
    df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.0
    )
    # Add ATR data - 3% of price (within min 1% and max 5%)
    df["atr"] = 3.0  # 3% of price (100)
    
    # Configure the filter
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df
    
    # Enable volatility filter in config
    mock_config = signal_engine_v1.config_manager.get_config.return_value
    mock_config["pairs"]["ETH_USDT"]["filters"] = {
        "volatility": {
            "enabled": True, 
            "indicator": "atr",
            "min_threshold": 1.0,  # 1%
            "max_threshold": 5.0   # 5%
        }
    }
    
    # Test signal generation
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    
    # Signal should be generated as ATR is within range
    assert signal is not None
    assert signal.symbol == "ETHUSDT"

@pytest.mark.asyncio
async def test_se_volatility_filter_blocks_below_min(signal_engine_v1, mock_data_processor_for_se):
    """Test that volatility filter blocks when ATR is below minimum threshold."""
    # Create DataFrame with ATR below min threshold
    df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.0
    )
    # Add ATR data - 0.5% of price (below min 1%)
    df["atr"] = 0.5  # 0.5% of price (100)
    
    # Configure the filter
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df
    
    # Enable volatility filter in config
    mock_config = signal_engine_v1.config_manager.get_config.return_value
    mock_config["pairs"]["ETH_USDT"]["filters"] = {
        "volatility": {
            "enabled": True, 
            "indicator": "atr",
            "min_threshold": 1.0,  # 1%
            "max_threshold": 5.0   # 5%
        }
    }
    
    # Test signal generation
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    
    # Signal should be blocked as ATR is below min threshold
    assert signal is None

@pytest.mark.asyncio
async def test_se_volatility_filter_blocks_above_max(signal_engine_v1, mock_data_processor_for_se):
    """Test that volatility filter blocks when ATR is above maximum threshold."""
    # Create DataFrame with ATR above max threshold
    df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.0
    )
    # Add ATR data - 6% of price (above max 5%)
    df["atr"] = 6.0  # 6% of price (100)
    
    # Configure the filter
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df
    
    # Enable volatility filter in config
    mock_config = signal_engine_v1.config_manager.get_config.return_value
    mock_config["pairs"]["ETH_USDT"]["filters"] = {
        "volatility": {
            "enabled": True, 
            "indicator": "atr",
            "min_threshold": 1.0,  # 1%
            "max_threshold": 5.0   # 5%
        }
    }
    
    # Test signal generation
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    
    # Signal should be blocked as ATR is above max threshold
    assert signal is None

@pytest.mark.asyncio
async def test_se_volatility_filter_handles_missing_data(signal_engine_v1, mock_data_processor_for_se):
    """Test that volatility filter gracefully handles missing ATR data."""
    # Create DataFrame with missing ATR
    df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.0
    )
    # Set ATR to NA
    df["atr"] = pd.NA
    
    # Configure the filter
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df
    
    # Enable volatility filter in config
    mock_config = signal_engine_v1.config_manager.get_config.return_value
    mock_config["pairs"]["ETH_USDT"]["filters"] = {
        "volatility": {
            "enabled": True, 
            "indicator": "atr",
            "min_threshold": 1.0,
            "max_threshold": 5.0
        }
    }
    
    # Test signal generation
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    
    # Signal should still be generated despite missing ATR (graceful handling)
    assert signal is not None
    assert signal.symbol == "ETHUSDT"

# 4. Test Helper Methods
@pytest.mark.asyncio
async def test_se_get_pair_specific_config(signal_engine_v1):
    # Test getting config for BTC_USDT (uses global values)
    btc_config = signal_engine_v1._get_pair_specific_config("BTC_USDT")
    assert btc_config["sma_short_period"] == 3
    assert btc_config["sma_long_period"] == 5
    assert btc_config["min_signal_interval_minutes"] == 15
    assert btc_config["tp_sl_ratio"] == 2.0
    assert btc_config["contract_type"] == "USDT_M"
    
    # Test getting config for ETH_USDT (has custom min_signal_interval_minutes)
    eth_config = signal_engine_v1._get_pair_specific_config("ETH_USDT")
    assert eth_config["sma_short_period"] == 3
    assert eth_config["sma_long_period"] == 5
    assert eth_config["min_signal_interval_minutes"] == 0  # Overridden
    assert eth_config["tp_sl_ratio"] == 2.0
    assert eth_config["contract_type"] == "USDT_M"
    
    # Test getting config for SOL_USDT (has custom tp_sl_ratio)
    sol_config = signal_engine_v1._get_pair_specific_config("SOL_USDT")
    assert sol_config["sma_short_period"] == 3
    assert sol_config["sma_long_period"] == 5
    assert sol_config["min_signal_interval_minutes"] == 15
    assert sol_config["tp_sl_ratio"] == 3.5  # Overridden
    assert sol_config["contract_type"] == "USDT_M"
    
    # Test getting config for unknown pair (should use global values)
    unknown_config = signal_engine_v1._get_pair_specific_config("XRP_USDT")
    assert unknown_config["sma_short_period"] == 3
    assert unknown_config["sma_long_period"] == 5
    assert unknown_config["min_signal_interval_minutes"] == 15
    assert unknown_config["tp_sl_ratio"] == 2.0
    assert unknown_config["contract_type"] == "USDT_M"

@pytest.mark.asyncio
async def test_se_find_recent_pivot_logic(signal_engine_v1):
    # Test _find_recent_pivot directly (it's a protected method, but crucial)
    data = {
        "timestamp": range(10), 
        "low": [10, 8, 9, 7, 10, 6, 8, 9, 7, 11],
        "high": [20, 22, 21, 23, 20, 24, 22, 21, 23, 19],
        "is_closed": [True]*10
    }
    df = pd.DataFrame(data).set_index("timestamp")
    
    # Test for LONG direction with different lookbacks
    # For lookback=5, it should look at indices 4,5,6,7,8 (lows: 10, 6, 8, 9, 7). Min is 6.
    pivot_l_5 = signal_engine_v1._find_recent_pivot(df, lookback=5, direction=TradeDirection.LONG)
    assert pivot_l_5 == 6
    
    # For lookback=3, it should look at indices 6,7,8 (lows: 8, 9, 7). Min is 7.
    pivot_l_3 = signal_engine_v1._find_recent_pivot(df, lookback=3, direction=TradeDirection.LONG)
    assert pivot_l_3 == 7
    
    # Test for SHORT direction with different lookbacks
    # For lookback=5, it should look at indices 4,5,6,7,8 (highs: 20, 24, 22, 21, 23). Max is 24.
    pivot_h_5 = signal_engine_v1._find_recent_pivot(df, lookback=5, direction=TradeDirection.SHORT)
    assert pivot_h_5 == 24
    
    # For lookback=3, it should look at indices 6,7,8 (highs: 22, 21, 23). Max is 23.
    pivot_h_3 = signal_engine_v1._find_recent_pivot(df, lookback=3, direction=TradeDirection.SHORT)
    assert pivot_h_3 == 23
    
    # Test with insufficient data
    assert signal_engine_v1._find_recent_pivot(df[:2], lookback=5, direction=TradeDirection.LONG) is None
    
    # Test with exactly 3 rows
    # According to the implementation in SignalEngineV1._find_recent_pivot,
    # we need at least 3 rows, but they must be closed candles *before* the signal candle
    # When we pass a 3-row DataFrame, there are actually only 2 rows considered for pivot calculation
    # (the signal candle itself is excluded with iloc[-lookback-1:-1])
    three_row_df = df[:3]
    pivot_l_min = signal_engine_v1._find_recent_pivot(three_row_df, lookback=5, direction=TradeDirection.LONG)
    # Since we only have 2 usable rows, which is less than the minimum 3 required,
    # the function should return None, not the min of [10, 8]
    assert pivot_l_min is None
    
    # Test with 4 rows (which gives 3 usable rows - the minimum required)
    four_row_df = df[:4]
    pivot_l_four = signal_engine_v1._find_recent_pivot(four_row_df, lookback=5, direction=TradeDirection.LONG)
    # Now we have 3 usable rows with lows [10, 8, 9], so min is 8
    assert pivot_l_four == 8
    
    # Test with flat values
    flat_data = {
        "timestamp": range(5),
        "low": [10, 10, 10, 10, 10],
        "high": [20, 20, 20, 20, 20],
        "is_closed": [True]*5
    }
    flat_df = pd.DataFrame(flat_data).set_index("timestamp")
    pivot_l_flat = signal_engine_v1._find_recent_pivot(flat_df, lookback=3, direction=TradeDirection.LONG)
    assert pivot_l_flat == 10
    pivot_h_flat = signal_engine_v1._find_recent_pivot(flat_df, lookback=3, direction=TradeDirection.SHORT)
    assert pivot_h_flat == 20

# Tests specific to the new check_signal_with_df method
@pytest.mark.asyncio
async def test_check_signal_with_df_long_signal(signal_engine_v1):
    """Test that check_signal_with_df generates a LONG signal correctly on crossover up"""
    # Create test DataFrame with crossover up (short crosses above long)
    mock_df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5
    )
    
    # Call the new method directly
    signal = await signal_engine_v1.check_signal_with_df("ETHUSDT", "ETH_USDT", mock_df)
    
    # Verify signal was generated correctly
    assert signal is not None
    assert signal.direction == TradeDirection.LONG
    assert signal.symbol == "ETHUSDT"
    assert signal.config_symbol == "ETH_USDT"
    assert signal.entry_price == 100.5  # From close_curr
    assert signal.stop_loss_price is not None
    assert signal.take_profit_price is not None
    
    # Verify the signal interval is independent of min_signal_interval_minutes
    # (which should only be checked in the wrapper check_signal method)
    signal_engine_v1.last_signal_time["ETHUSDT"] = time.time()  # Signal was just generated
    
    # Should still generate a signal (min_signal_interval is not checked in check_signal_with_df)
    signal2 = await signal_engine_v1.check_signal_with_df("ETHUSDT", "ETH_USDT", mock_df)
    assert signal2 is not None

@pytest.mark.asyncio
async def test_check_signal_with_df_short_signal(signal_engine_v1):
    """Test that check_signal_with_df generates a SHORT signal correctly on crossover down"""
    # Create test DataFrame with crossover down (short crosses below long)
    mock_df = create_mock_df(
        sma_short_prev=100, sma_long_prev=99, 
        sma_short_curr=98, sma_long_curr=99, 
        close_curr=100.5
    )
    
    # Call the new method directly
    signal = await signal_engine_v1.check_signal_with_df("ETHUSDT", "ETH_USDT", mock_df)
    
    # Verify signal was generated correctly
    assert signal is not None
    assert signal.direction == TradeDirection.SHORT
    assert signal.symbol == "ETHUSDT"
    assert signal.config_symbol == "ETH_USDT"
    assert signal.entry_price == 100.5  # From close_curr
    assert signal.stop_loss_price is not None
    assert signal.take_profit_price is not None

@pytest.mark.asyncio
async def test_check_signal_with_df_integration_with_check_signal(signal_engine_v1, mock_data_processor_for_se):
    """Test that check_signal properly calls check_signal_with_df with the DataFrame from DataProcessor"""
    # Create test DataFrame with crossover up
    mock_df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5
    )
    
    # Configure DataProcessor mock to return our test DataFrame
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    
    # Patch the check_signal_with_df method to check if it's called correctly
    with patch.object(signal_engine_v1, 'check_signal_with_df', wraps=signal_engine_v1.check_signal_with_df) as wrapped_method:
        # Call check_signal which should internally call check_signal_with_df
        signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
        
        # Verify that check_signal_with_df was called with the correct arguments
        wrapped_method.assert_called_once()
        args, kwargs = wrapped_method.call_args
        assert args[0] == "ETHUSDT"
        assert args[1] == "ETH_USDT"
        assert args[2].equals(mock_df)  # The DataFrame should be passed unchanged
        
        # Verify that check_signal returned the expected signal
        assert signal is not None
        assert signal.direction == TradeDirection.LONG

@pytest.mark.asyncio
async def test_check_signal_with_df_varying_dataframe_size(signal_engine_v1):
    """Test that check_signal_with_df works correctly with different sized DataFrames"""
    # Create base test DataFrame with crossover up
    base_df = create_mock_df(
        sma_short_prev=99, sma_long_prev=100, 
        sma_short_curr=101, sma_long_curr=100, 
        close_curr=100.5
    )
    
    # Add 20 more rows before the crossover to test with a larger historical context
    extended_rows = []
    for i in range(20):
        timestamp = base_df.index[0] - (i+1) * 60000  # 1 minute earlier each time
        row = {
            'timestamp': timestamp,
            'open': 100.0 - i*0.1,
            'high': 101.0 - i*0.1,
            'low': 99.0 - i*0.1,
            'close': 100.0 - i*0.1,
            'volume': 1000.0,
            'sma_short': 98.0 - i*0.1,  # Keep below sma_long
            'sma_long': 99.0 - i*0.1,
            'is_closed': True
        }
        extended_rows.append(row)
    
    extended_df = pd.concat([
        pd.DataFrame(extended_rows).set_index('timestamp'),
        base_df
    ]).sort_index()
    
    # Test with the extended DataFrame
    signal_ext = await signal_engine_v1.check_signal_with_df("ETHUSDT", "ETH_USDT", extended_df)
    
    # Should generate the same signal as with the smaller DataFrame
    assert signal_ext is not None
    assert signal_ext.direction == TradeDirection.LONG
    
    # Verify that the stop loss calculation can use the additional historical data
    # by checking if the SL price is potentially different (more historical lows available)
    signal_base = await signal_engine_v1.check_signal_with_df("ETHUSDT", "ETH_USDT", base_df)
    
    # Both signals should be valid, though SL values might differ due to different pivot calculations
    assert signal_base.stop_loss_price is not None
    assert signal_ext.stop_loss_price is not None

