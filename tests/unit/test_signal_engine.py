import pytest
import asyncio
import pandas as pd
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
                "min_signal_interval_minutes": 0 # For testing without wait
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
    # Ensure lookback arrays are long enough for pivot logic if needed
    # Create enough past data for pivot calculation (e.g., 30 candles)
    past_lows = low_lookback * 10 # Repeat to make it 30 long
    past_highs = high_lookback * 10
    past_closes = [close_curr - 1] * 30 # Dummy past closes

    data = {
        "timestamp": list(range(1000, 32000, 1000)), # 31 points for previous, 1 for current
        "open": [close_curr-1]*30 + [close_curr-1, close_curr-0.5],
        "high": past_highs + [max(high_lookback), close_curr + 2],
        "low": past_lows + [min(low_lookback), close_curr - 2],
        "close": past_closes + [close_curr-1, close_curr],
        "volume": [100]*32,
        "is_closed": [True]*31 + [True], # Current candle is also closed for signal generation
        "sma_short": [sma_short_prev -1]*30 + [sma_short_prev, sma_short_curr],
        "sma_long": [sma_long_prev-1]*30 + [sma_long_prev, sma_long_curr],
    }
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df

@pytest.mark.asyncio
async def test_se_no_signal_if_not_enough_data(signal_engine_v1, mock_data_processor_for_se):
    mock_data_processor_for_se.get_indicator_dataframe.return_value = pd.DataFrame() # Empty DF
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

    mock_data_processor_for_se.get_indicator_dataframe.return_value = create_mock_df(10,20,11,19)[:1] # Only one row
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

@pytest.mark.asyncio
async def test_se_no_signal_if_sma_na(signal_engine_v1, mock_data_processor_for_se):
    df_with_na = create_mock_df(10,20,11,19)
    df_with_na.loc[df_with_na.index[-1], "sma_short"] = pd.NA
    mock_data_processor_for_se.get_indicator_dataframe.return_value = df_with_na
    signal = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal is None

@pytest.mark.asyncio
async def test_se_long_signal_sma_crossover(signal_engine_v1, mock_data_processor_for_se):
    # prev: short <= long, curr: short > long
    mock_df = create_mock_df(sma_short_prev=99, sma_long_prev=100, sma_short_curr=101, sma_long_curr=100, close_curr=100.5, low_lookback=[90,88,89])
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT") # ETH_USDT has min_interval 0
    assert signal is not None
    assert signal.direction == TradeDirection.LONG
    assert signal.symbol == "ETHUSDT"
    assert signal.entry_price == 100.5 # latest close
    assert signal.stop_loss_price == 88 # min of low_lookback
    # Risk = 100.5 - 88 = 12.5. TP = 100.5 + (12.5 * 2.0) = 100.5 + 25 = 125.5
    assert signal.take_profit_price == 125.5
    assert signal.strategy_name == "V1_SMA_Crossover"
    assert signal.signal_kline is not None
    assert signal.signal_kline.close == 100.5

@pytest.mark.asyncio
async def test_se_short_signal_sma_crossover(signal_engine_v1, mock_data_processor_for_se):
    # prev: short >= long, curr: short < long
    mock_df = create_mock_df(sma_short_prev=101, sma_long_prev=100, sma_short_curr=99, sma_long_curr=100, close_curr=99.5, high_lookback=[110,112,111])
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df

    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is not None
    assert signal.direction == TradeDirection.SHORT
    assert signal.symbol == "ETHUSDT"
    assert signal.entry_price == 99.5
    assert signal.stop_loss_price == 112 # max of high_lookback
    # Risk = 112 - 99.5 = 12.5. TP = 99.5 - (12.5 * 2.0) = 99.5 - 25 = 74.5
    assert signal.take_profit_price == 74.5

@pytest.mark.asyncio
async def test_se_no_signal_if_no_crossover(signal_engine_v1, mock_data_processor_for_se):
    # short > long, and still short > long
    mock_df = create_mock_df(sma_short_prev=101, sma_long_prev=100, sma_short_curr=102, sma_long_curr=99)
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is None

@pytest.mark.asyncio
async def test_se_signal_interval_filter(signal_engine_v1, mock_data_processor_for_se, mock_config_manager_for_se):
    # BTC_USDT has min_interval_minutes = 15
    mock_df = create_mock_df(sma_short_prev=99, sma_long_prev=100, sma_short_curr=101, sma_long_curr=100, close_curr=100.5)
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df

    # First signal should pass
    signal1 = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal1 is not None
    assert signal_engine_v1.last_signal_time["BTCUSDT"] > 0

    # Second signal immediately after should be filtered
    signal2 = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal2 is None

    # Simulate time passing (less than interval)
    signal_engine_v1.last_signal_time["BTCUSDT"] = time.time() - (10 * 60) # 10 minutes ago
    signal3 = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal3 is None 

    # Simulate time passing (more than interval)
    signal_engine_v1.last_signal_time["BTCUSDT"] = time.time() - (20 * 60) # 20 minutes ago
    signal4 = await signal_engine_v1.check_signal("BTCUSDT", "BTC_USDT")
    assert signal4 is not None

@pytest.mark.asyncio
async def test_se_sl_tp_calculation_invalid_risk(signal_engine_v1, mock_data_processor_for_se):
    # SL is worse than entry for LONG
    mock_df_bad_sl_long = create_mock_df(99,100,101,100, close_curr=90, low_lookback=[95,96,97]) # entry 90, pivot_low 95
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df_bad_sl_long
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is None

    # SL is worse than entry for SHORT
    mock_df_bad_sl_short = create_mock_df(101,100,99,100, close_curr=110, high_lookback=[105,106,104]) # entry 110, pivot_high 106
    mock_data_processor_for_se.get_indicator_dataframe.return_value = mock_df_bad_sl_short
    signal = await signal_engine_v1.check_signal("ETHUSDT", "ETH_USDT")
    assert signal is None

@pytest.mark.asyncio
async def test_se_find_recent_pivot_logic(signal_engine_v1):
    # Test _find_recent_pivot directly (it is a protected method, but crucial)
    data = {
        "timestamp": range(10), 
        "low": [10, 8, 9, 7, 10, 6, 8, 9, 7, 11],
        "high": [20, 22, 21, 23, 20, 24, 22, 21, 23, 19],
        "is_closed": [True]*10
    }
    df = pd.DataFrame(data).set_index("timestamp")
    
    # For LONG, look for min low in lookback (excluding last candle)
    # If df is passed as klines_df, it looks at df.iloc[-lookback-1:-1]
    # If df has 10 rows, and lookback is 5. It looks at rows from index 3 up to 8 (exclusive of 9)
    # df.iloc[10-5-1 : -1] = df.iloc[4:-1] = indices 4,5,6,7,8
    # lows: 10, 6, 8, 9, 7. Min is 6.
    pivot_l = signal_engine_v1._find_recent_pivot(df, lookback=5, direction=TradeDirection.LONG)
    assert pivot_l == 6 

    # For SHORT, look for max high in lookback
    # highs: 20, 24, 22, 21, 23. Max is 24.
    pivot_h = signal_engine_v1._find_recent_pivot(df, lookback=5, direction=TradeDirection.SHORT)
    assert pivot_h == 24

    assert signal_engine_v1._find_recent_pivot(df[:2], lookback=5, direction=TradeDirection.LONG) is None # Not enough data

