import pytest
import asyncio
import pandas as pd
import time
from unittest.mock import AsyncMock, MagicMock, patch
import os
import yaml

# Ensure src is in path for tests if running pytest from root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.config_loader import ConfigManager
from src.connectors import BinanceRESTClient, BinanceWebSocketConnector, convert_symbol_to_api_format
from src.data_processor import DataProcessor
from src.signal_engine import SignalEngineV1
from src.order_manager import OrderManager
from src.position_manager import PositionManager
from src.models import Kline, TradeSignal, TradeDirection, Order, OrderStatus, Position
from src.main import TradingBot # To test overall orchestration if needed

# --- Test Fixtures --- 
@pytest.fixture(scope="module") # Module scope for heavier setup
def event_loop():
    # Pytest-asyncio provides an event loop by default, but can be explicit if needed
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_config_file_for_integration(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_content = """
    api:
      binance_api_key: "INTEGRATION_TEST_KEY" # Use distinct keys for integration tests
      binance_api_secret: "INTEGRATION_TEST_SECRET"
    global_settings:
      v1_strategy:
        sma_short_period: 3
        sma_long_period: 5
        min_signal_interval_minutes: 0 # No wait for testing
        tp_sl_ratio: 1.5
        default_margin_usdt: 20.0
        default_leverage: 5
        indicator_timeframes: ["1m"]
      risk_management:
        dynamic_sizing_enabled: false
    pairs:
      TEST_USDT:
        enabled: true
        contract_type: "USDT_M"
        # margin_usdt: 20 # Uses global
        # leverage: 5   # Uses global
    logging:
      level: "DEBUG"
    monitoring:
      prometheus_port: 8001 # Different port for tests
    """
    with open(config_file, "w") as f:
        f.write(config_content)
    return str(config_file)

@pytest.fixture
def integration_config_manager(temp_config_file_for_integration):
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None # Reset singleton
    with patch("src.config_loader.DEFAULT_CONFIG_PATH", temp_config_file_for_integration):
        cm = ConfigManager(auto_reload=False)
        yield cm
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None

@pytest.fixture
def mock_binance_rest_client_integration(integration_config_manager):
    client = BinanceRESTClient(config_manager=integration_config_manager)
    # Mock actual API calls to avoid hitting Binance
    client.exchange = AsyncMock() # Mock the ccxt exchange object itself
    client.exchange.load_markets = AsyncMock(return_value={
        "TESTUSDT": {
            "symbol": "TESTUSDT", "base": "TEST", "quote": "USDT", "type": "future",
            "precision": {"amount": "0.001", "price": "0.01"},
            "limits": {"amount": {"min": "0.001"}, "cost": {"min": "5.0"}}
        }
    })
    
    # Create a proper AsyncMock for create_order so call_count is accessible
    create_order_mock = AsyncMock()
    create_order_mock.side_effect = lambda symbol, order_type, side, amount, price=None, params=None: {
        "id": f"mock_order_{int(time.time()*1000)}",
        "symbol": symbol, 
        "type": order_type, 
        "side": side, 
        "amount": amount, 
        "price": price,
        "status": "NEW", # Simulate NEW, then assume it fills for test flow
        "avgPrice": str(price or params.get("stopPrice") or 100.0), # Mock fill price
        "filled": amount # Assume full fill for market orders
    }
    client.create_order = create_order_mock
    
    client.exchange.set_leverage = AsyncMock(return_value=True)
    client.exchange.set_margin_mode = AsyncMock(return_value=True)
    client.exchange.fetch_balance = AsyncMock(return_value={"USDT": {"free": 1000.0, "total": 1000.0}})
    client.exchange.close = AsyncMock()
    return client

# --- Integration Tests --- 

@pytest.mark.asyncio
async def test_kline_to_signal_to_order_flow(integration_config_manager, mock_binance_rest_client_integration):
    """ Test the flow: Kline -> DataProcessor -> SignalEngine -> OrderManager """
    
    # 1. Setup components
    data_processor = DataProcessor(config_manager=integration_config_manager)
    signal_engine = SignalEngineV1(config_manager=integration_config_manager, data_processor=data_processor)
    order_manager = OrderManager(config_manager=integration_config_manager, rest_client=mock_binance_rest_client_integration)
    # PositionManager could be added if we test its interaction too

    api_symbol = "TESTUSDT"
    config_symbol = "TEST_USDT"
    interval = "1m"

    # 2. Simulate incoming Kline data that should generate a signal
    # Data for a LONG signal (SMA3 crosses above SMA5)
    # SMA3 periods: 3, SMA5 periods: 5
    # Prices: 90, 92, 94 (SMA3=92), 96 (SMA3=94), 98 (SMA3=96)
    #         SMA5 = (90+92+94+96+98)/5 = 94
    # Next candle: close=100. SMA3=(96+98+100)/3 = 98. SMA5=(92+94+96+98+100)/5 = 96.
    # Crossover: Prev SMA3(96) > Prev SMA5(94). Curr SMA3(98) > Curr SMA5(96). This is not a crossover from below.
    # Let's make prev short < long, curr short > long
    # Prices: 90, 92, 90 (SMA3=90.67), 92 (SMA3=91.33), 90 (SMA3=90.67)
    #         SMA5 = (90+92+90+92+90)/5 = 90.8
    # Prev (idx 4, close 90): SMA3=90.67, SMA5=90.8 (short < long)
    # Curr (idx 5, close 98): SMA3=(90+92+98)/3=93.33. SMA5=(92+90+92+90+98)/5=92.4 (short > long) -> BUY
    klines_to_process = [
        Kline(timestamp=1000, open=90, high=91, low=89, close=90, volume=10, is_closed=True, symbol=api_symbol, interval=interval),
        Kline(timestamp=2000, open=90, high=93, low=91, close=92, volume=10, is_closed=True, symbol=api_symbol, interval=interval),
        Kline(timestamp=3000, open=92, high=92, low=89, close=90, volume=10, is_closed=True, symbol=api_symbol, interval=interval),
        Kline(timestamp=4000, open=90, high=93, low=91, close=92, volume=10, is_closed=True, symbol=api_symbol, interval=interval),
        Kline(timestamp=5000, open=92, high=92, low=89, close=90, volume=10, is_closed=True, symbol=api_symbol, interval=interval), # Prev state: SMA3=90.67, SMA5=90.8
        Kline(timestamp=6000, open=90, high=99, low=89, close=98, volume=10, is_closed=True, symbol=api_symbol, interval=interval)  # Curr state: SMA3=93.33, SMA5=92.4 -> BUY
    ]

    for kline in klines_to_process:
        await data_processor.process_kline(kline)
    
    # 3. Check for signal
    # The signal check would typically happen in the main loop or after kline processing
    # For this test, we call it directly.
    signal = await signal_engine.check_signal(api_symbol=api_symbol, config_symbol=config_symbol)
    
    assert signal is not None, "Signal should have been generated"
    assert signal.direction == TradeDirection.LONG
    assert signal.symbol == api_symbol
    assert signal.entry_price == 98.0 # Close of the signal candle
    # SL: min low of lookback. DataProcessor buffer has these klines.
    # Pivots are calculated on `df` inside signal_engine. `df` comes from data_processor.
    # Lows in buffer: 89,91,89,91,89,89. Min low for SL should be 89.
    assert signal.stop_loss_price == 89.0
    # TP: entry + (entry - SL) * tp_sl_ratio = 98 + (98 - 89) * 1.5 = 98 + 9 * 1.5 = 98 + 13.5 = 111.5
    assert signal.take_profit_price == 111.5

    # 4. Handle signal with OrderManager
    # Ensure exchange info is primed for the mock client
    await order_manager._get_exchange_info_for_symbol(api_symbol) 
    
    await order_manager.handle_trade_signal(signal)

    # 5. Verify orders were placed (mocked)
    # Expected quantity: (20 USDT margin * 5x leverage) / 98 entry = 1.0204...
    # Adjusted by step 0.001: floor(1.0204... / 0.001)*0.001 = 1.020
    expected_quantity = 1.020

    assert mock_binance_rest_client_integration.create_order.call_count == 3
    calls = mock_binance_rest_client_integration.create_order.call_args_list
    
    # Entry order
    assert calls[0][1]["symbol"] == api_symbol
    assert calls[0][1]["order_type"] == "MARKET"
    assert calls[0][1]["side"] == "BUY"
    assert calls[0][1]["amount"] == expected_quantity
    
    # SL order (price adjusted to 0.01 tick)
    assert calls[1][1]["params"]["stopPrice"] == 89.00
    assert calls[1][1]["amount"] == expected_quantity

    # TP order (price adjusted to 0.01 tick)
    assert calls[2][1]["params"]["stopPrice"] == 111.50
    assert calls[2][1]["amount"] == expected_quantity

@pytest.mark.asyncio
async def test_full_bot_cycle_with_mock_dependencies(temp_config_file_for_integration):
    """ Test a simplified integration between components without using TradingBot """
    
    # Reset ConfigManager singleton
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None
    
    with patch("src.config_loader.DEFAULT_CONFIG_PATH", temp_config_file_for_integration):
        # Create a real config manager using the temp file
        config_manager = ConfigManager(auto_reload=False)
        
        # Setup components
        data_processor = DataProcessor(config_manager=config_manager)
        signal_engine = SignalEngineV1(config_manager=config_manager, data_processor=data_processor)
        
        # Setup REST client mock
        rest_client = AsyncMock()
        rest_client.fetch_exchange_info.return_value = {
            "TESTUSDT": {
                "symbol": "TESTUSDT", "precision": {"amount": "0.001", "price": "0.01"},
                "limits": {"cost": {"min": "5.0"}}
            }
        }
        
        # Mock create_order with an AsyncMock that we can track
        rest_client.create_order = AsyncMock(return_value={
            "id": "test_order_id",
            "status": "NEW"
        })
        
        # Create OrderManager with our mock
        order_manager = OrderManager(config_manager=config_manager, rest_client=rest_client)
        
        # Test data
        api_symbol = "TESTUSDT"
        config_symbol = "TEST_USDT"
        interval = "1m"
        
        # Create klines that will trigger a buy signal
        klines_to_process = [
            Kline(timestamp=1000, open=90, high=91, low=89, close=90, volume=10, is_closed=True, symbol=api_symbol, interval=interval),
            Kline(timestamp=2000, open=90, high=93, low=91, close=92, volume=10, is_closed=True, symbol=api_symbol, interval=interval),
            Kline(timestamp=3000, open=92, high=92, low=89, close=90, volume=10, is_closed=True, symbol=api_symbol, interval=interval),
            Kline(timestamp=4000, open=90, high=93, low=91, close=92, volume=10, is_closed=True, symbol=api_symbol, interval=interval),
            Kline(timestamp=5000, open=92, high=92, low=89, close=90, volume=10, is_closed=True, symbol=api_symbol, interval=interval),
            Kline(timestamp=6000, open=90, high=99, low=89, close=98, volume=10, is_closed=True, symbol=api_symbol, interval=interval)
        ]
        
        # Process klines
        for kline in klines_to_process:
            await data_processor.process_kline(kline)
        
        # Get signal
        signal = await signal_engine.check_signal(api_symbol=api_symbol, config_symbol=config_symbol)
        
        # Verify signal
        assert signal is not None, "Signal should have been generated"
        assert signal.direction == TradeDirection.LONG
        assert signal.symbol == api_symbol
        
        # Handle signal with OrderManager
        await order_manager.handle_trade_signal(signal)
        
        # Verify orders were placed
        assert rest_client.create_order.call_count == 3, "Should have created entry, SL, and TP orders"
        
        # Verify that config contains our test pair
        config = config_manager.get_config()
        assert "TEST_USDT" in config.get("pairs", {}), "Test pair should be in config"

    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None # Cleanup singleton

