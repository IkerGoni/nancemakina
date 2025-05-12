import pytest
import asyncio
import pandas as pd
from unittest.mock import MagicMock, patch

from src.config_loader import ConfigManager
from src.data_processor import DataProcessor, MAX_KLINE_BUFFER_LENGTH
from src.models import Kline

@pytest.fixture
def mock_config_manager_for_dp(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_content = """
    global_settings:
      v1_strategy:
        sma_short_period: 3
        sma_long_period: 5
        indicator_timeframes: ["1m", "5m"]
    pairs:
      BTC_USDT:
        enabled: true
        contract_type: "USDT_M"
        indicator_timeframes: ["1m"] # Override global
      ETH_USDT:
        enabled: true
        contract_type: "USDT_M"
        # Uses global timeframes: ["1m", "5m"]
      SOL_USDT:
        enabled: false # This pair should not have buffers initialized initially
        contract_type: "USDT_M"
    """
    with open(config_file, "w") as f:
        f.write(config_content)
    
    if hasattr(ConfigManager, 	_instance	):
        ConfigManager._instance = None
    with patch("src.config_loader.DEFAULT_CONFIG_PATH", str(config_file)):
        cm = ConfigManager(auto_reload=False)
        yield cm
    if hasattr(ConfigManager, 	_instance	):
        ConfigManager._instance = None

@pytest.fixture
def data_processor(mock_config_manager_for_dp):
    dp = DataProcessor(config_manager=mock_config_manager_for_dp)
    return dp

@pytest.mark.asyncio
async def test_dp_initialization_creates_buffers(data_processor):
    assert "BTCUSDT" in data_processor._active_pairs_timeframes
    assert "1m" in data_processor._active_pairs_timeframes["BTCUSDT"]
    assert "5m" not in data_processor._active_pairs_timeframes["BTCUSDT"] # Overridden
    assert "ETHUSDT" in data_processor._active_pairs_timeframes
    assert "1m" in data_processor._active_pairs_timeframes["ETHUSDT"]
    assert "5m" in data_processor._active_pairs_timeframes["ETHUSDT"]
    assert "SOLUSDT" not in data_processor._active_pairs_timeframes # Disabled

    assert "BTCUSDT" in data_processor.kline_buffers
    assert "1m" in data_processor.kline_buffers["BTCUSDT"]
    assert isinstance(data_processor.kline_buffers["BTCUSDT"]["1m"], pd.collections.deque)
    assert data_processor.kline_buffers["BTCUSDT"]["1m"].maxlen == MAX_KLINE_BUFFER_LENGTH

@pytest.mark.asyncio
async def test_dp_process_kline_active_pair(data_processor):
    kline_data = Kline(timestamp=1000, open=10, high=12, low=9, close=11, volume=100, is_closed=True, symbol="BTCUSDT", interval="1m")
    await data_processor.process_kline(kline_data)
    
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0] == kline_data
    
    # Check if indicators were updated (SMA calculation)
    df = data_processor.get_indicator_dataframe("BTCUSDT", "1m")
    assert df is not None
    assert not df.empty
    assert "sma_short" in df.columns
    assert "sma_long" in df.columns

@pytest.mark.asyncio
async def test_dp_process_kline_inactive_pair(data_processor):
    kline_data = Kline(timestamp=1000, open=10, high=12, low=9, close=11, volume=100, is_closed=True, symbol="SOLUSDT", interval="1m")
    await data_processor.process_kline(kline_data)
    # SOLUSDT is disabled in config, so no buffer should be created or kline processed for it
    assert "SOLUSDT" not in data_processor.kline_buffers 
    # Or if defaultdict creates it, it should be empty
    if "SOLUSDT" in data_processor.kline_buffers:
        assert not data_processor.kline_buffers["SOLUSDT"]["1m"]

@pytest.mark.asyncio
async def test_dp_process_kline_update_last_candle(data_processor):
    kline1 = Kline(timestamp=1000, open=10, high=12, low=9, close=11, volume=100, is_closed=False, symbol="BTCUSDT", interval="1m")
    await data_processor.process_kline(kline1)
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0].close == 11

    kline2_update = Kline(timestamp=1000, open=10, high=13, low=9, close=12, volume=150, is_closed=False, symbol="BTCUSDT", interval="1m")
    await data_processor.process_kline(kline2_update)
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1 # Should update, not append
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0].close == 12
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0].high == 13

    kline3_closed = Kline(timestamp=1000, open=10, high=13, low=9, close=12.5, volume=200, is_closed=True, symbol="BTCUSDT", interval="1m")
    await data_processor.process_kline(kline3_closed)
    assert len(data_processor.kline_buffers["BTCUSDT"]["1m"]) == 1 # Should update the non-closed one
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0].close == 12.5
    assert data_processor.kline_buffers["BTCUSDT"]["1m"][0].is_closed is True

@pytest.mark.asyncio
async def test_dp_sma_calculation(data_processor):
    symbol = "ETHUSDT"
    interval = "1m"
    klines_data = [
        Kline(timestamp=1000, open=1, high=1, low=1, close=1, volume=1, is_closed=True, symbol=symbol, interval=interval),
        Kline(timestamp=2000, open=1, high=1, low=1, close=2, volume=1, is_closed=True, symbol=symbol, interval=interval),
        Kline(timestamp=3000, open=1, high=1, low=1, close=3, volume=1, is_closed=True, symbol=symbol, interval=interval),
        Kline(timestamp=4000, open=1, high=1, low=1, close=4, volume=1, is_closed=True, symbol=symbol, interval=interval),
        Kline(timestamp=5000, open=1, high=1, low=1, close=5, volume=1, is_closed=True, symbol=symbol, interval=interval),
        Kline(timestamp=6000, open=1, high=1, low=1, close=6, volume=1, is_closed=True, symbol=symbol, interval=interval),
    ]
    for k in klines_data:
        await data_processor.process_kline(k)

    df = data_processor.get_indicator_dataframe(symbol, interval)
    assert df is not None
    # SMA short period = 3, SMA long period = 5 from mock_config
    # Expected SMA3 at t=3000 (close=3): (1+2+3)/3 = 2.0
    # Expected SMA5 at t=5000 (close=5): (1+2+3+4+5)/5 = 3.0
    # Expected SMA3 at t=6000 (close=6): (4+5+6)/3 = 5.0
    # Expected SMA5 at t=6000 (close=6): (2+3+4+5+6)/5 = 4.0
    
    latest_indicators = data_processor.get_latest_indicators(symbol, interval)
    assert latest_indicators is not None
    pd.testing.assert_series_equal(latest_indicators[["sma_short", "sma_long"]], pd.Series([5.0, 4.0], index=["sma_short", "sma_long"]), check_dtype=False)
    
    assert df.loc[3000, "sma_short"] == 2.0
    assert pd.isna(df.loc[3000, "sma_long"]) # Not enough data for SMA5
    assert df.loc[5000, "sma_long"] == 3.0

@pytest.mark.asyncio
async def test_dp_get_latest_kline_and_indicators(data_processor):
    k1 = Kline(timestamp=1000, open=1, high=1, low=1, close=1, volume=1, is_closed=True, symbol="BTCUSDT", interval="1m")
    k2 = Kline(timestamp=2000, open=1, high=1, low=1, close=2, volume=1, is_closed=False, symbol="BTCUSDT", interval="1m")
    await data_processor.process_kline(k1)
    await data_processor.process_kline(k2)

    latest_k = data_processor.get_latest_kline("BTCUSDT", "1m")
    assert latest_k == k2

    latest_ind = data_processor.get_latest_indicators("BTCUSDT", "1m")
    assert latest_ind is not None
    assert latest_ind["close"] == 2
    assert latest_ind["is_closed"] is False # Indicators are based on latest kline in buffer

    assert data_processor.get_latest_kline("NONEXISTENT", "1m") is None
    assert data_processor.get_latest_indicators("NONEXISTENT", "1m") is None

@pytest.mark.asyncio
async def test_dp_config_hot_reload_updates_active_pairs(mock_config_manager_for_dp, data_processor):
    # Initial state: BTC_USDT and ETH_USDT are active
    assert "BTCUSDT" in data_processor._active_pairs_timeframes
    assert "XRPUSDT" not in data_processor._active_pairs_timeframes

    # Simulate config change by directly calling the callback (easier than file ops here)
    new_config_data = mock_config_manager_for_dp.get_config()
    new_config_data["pairs"]["BTC_USDT"]["enabled"] = False
    new_config_data["pairs"]["XRP_USDT"] = {"enabled": True, "contract_type": "USDT_M", "indicator_timeframes": ["1m"]}
    
    # Manually trigger the config update handler in DataProcessor
    data_processor._handle_config_update(new_config_data)

    assert "BTCUSDT" not in data_processor._active_pairs_timeframes # Should be removed or marked inactive
    assert "XRPUSDT" in data_processor._active_pairs_timeframes
    assert "1m" in data_processor._active_pairs_timeframes["XRPUSDT"]

    # Ensure new buffers are ready for XRPUSDT
    assert "XRPUSDT" in data_processor.kline_buffers
    assert "1m" in data_processor.kline_buffers["XRPUSDT"]

