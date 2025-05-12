import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.config_loader import ConfigManager
from src.connectors import BinanceRESTClient, BinanceWebSocketConnector, convert_symbol_to_ws_format, convert_symbol_to_api_format
from src.models import Kline

@pytest.fixture
def mock_config_manager_for_connectors(tmp_path):
    # Create a dummy config file for tests
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_content = """
    api:
      binance_api_key: "TEST_KEY"
      binance_api_secret: "TEST_SECRET"
    global_settings:
      v1_strategy:
        indicator_timeframes: ["1m", "5m"]
    pairs:
      BTC_USDT:
        enabled: true
        contract_type: "USDT_M"
      ETH_USDT:
        enabled: true
        contract_type: "USDT_M"
        indicator_timeframes: ["1m"]
      BTCUSD_PERP:
        enabled: true
        contract_type: "COIN_M"
        indicator_timeframes: ["1m"]
    logging:
      level: "DEBUG"
    """
    with open(config_file, "w") as f:
        f.write(config_content)
    
    # Patch ConfigManager to use this temp config file
    # and ensure singleton is reset for these tests
    if hasattr(ConfigManager, 	_instance	):
        ConfigManager._instance = None

    with patch("src.config_loader.DEFAULT_CONFIG_PATH", str(config_file)):
        cm = ConfigManager(auto_reload=False)
        yield cm
    
    # Cleanup singleton after test
    if hasattr(ConfigManager, 	_instance	):
        ConfigManager._instance = None

# --- Test Symbol Conversion --- 
def test_convert_symbol_to_ws_format():
    assert convert_symbol_to_ws_format("BTC_USDT") == "btcusdt"
    assert convert_symbol_to_ws_format("ETH_BTC") == "ethbtc"
    assert convert_symbol_to_ws_format("BTCUSD_PERP") == "btcusd_perp"
    assert convert_symbol_to_ws_format("btcusdt") == "btcusdt" # Already formatted

def test_convert_symbol_to_api_format():
    assert convert_symbol_to_api_format("BTC_USDT") == "BTCUSDT"
    assert convert_symbol_to_api_format("ETH_BTC") == "ETHBTC"
    assert convert_symbol_to_api_format("BTCUSD_PERP") == "BTCUSD_PERP"
    assert convert_symbol_to_api_format("btcusdt") == "BTCUSDT"

# --- Test BinanceRESTClient --- 
@pytest.mark.asyncio
async def test_rest_client_initialization(mock_config_manager_for_connectors):
    rest_client = BinanceRESTClient(config_manager=mock_config_manager_for_connectors)
    assert rest_client.exchange is not None # Should initialize ccxt exchange
    await rest_client.close_exchange()

@pytest.mark.asyncio
async def test_rest_client_initialization_no_api_keys(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_content = """
    api:
      binance_api_key: "YOUR_BINANCE_API_KEY"
      binance_api_secret: "YOUR_BINANCE_API_SECRET"
    """
    with open(config_file, "w") as f:
        f.write(config_content)
    if hasattr(ConfigManager, 	_instance	):
        ConfigManager._instance = None
    with patch("src.config_loader.DEFAULT_CONFIG_PATH", str(config_file)):
        cm = ConfigManager(auto_reload=False)
        rest_client = BinanceRESTClient(config_manager=cm)
        assert rest_client.exchange is None # Should be None if keys are placeholders
        # Test a call, should simulate or return default
        order = await rest_client.create_order("BTCUSDT", "MARKET", "BUY", 1.0)
        assert order["id"] == "simulated_order_id"
        await rest_client.close_exchange()
    if hasattr(ConfigManager, 	_instance	):
        ConfigManager._instance = None

@pytest.mark.asyncio
async def test_rest_client_fetch_balance(mock_config_manager_for_connectors):
    rest_client = BinanceRESTClient(config_manager=mock_config_manager_for_connectors)
    with patch.object(rest_client.exchange, "fetch_balance", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"USDT": {"free": 1000, "total": 1000}}
        balance = await rest_client.fetch_balance()
        assert balance["USDT"]["free"] == 1000
        mock_fetch.assert_called_once()
    await rest_client.close_exchange()

@pytest.mark.asyncio
async def test_rest_client_create_order(mock_config_manager_for_connectors):
    rest_client = BinanceRESTClient(config_manager=mock_config_manager_for_connectors)
    with patch.object(rest_client.exchange, "create_order", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = {"id": "123", "symbol": "BTCUSDT", "status": "NEW"}
        order = await rest_client.create_order("BTC_USDT", "MARKET", "BUY", 0.01)
        assert order["id"] == "123"
        mock_create.assert_called_once_with("BTCUSDT", "MARKET", "BUY", 0.01, None, {})
    await rest_client.close_exchange()

# --- Test BinanceWebSocketConnector --- 
@pytest.fixture
def mock_kline_callback():
    return AsyncMock()

@pytest.mark.asyncio
async def test_ws_connector_get_active_pairs_and_intervals(mock_config_manager_for_connectors, mock_kline_callback):
    ws_connector = BinanceWebSocketConnector(config_manager=mock_config_manager_for_connectors, kline_callback=mock_kline_callback)
    active_streams = ws_connector._get_active_pairs_and_intervals()
    assert "btcusdt@kline_1m" in active_streams["USDT_M"]
    assert "btcusdt@kline_5m" in active_streams["USDT_M"]
    assert "ethusdt@kline_1m" in active_streams["USDT_M"]
    assert "btcusd_perp@kline_1m" in active_streams["COIN_M"]
    assert "ethusdt@kline_5m" not in active_streams["USDT_M"] # Overridden to 1m only

@pytest.mark.asyncio
async def test_ws_connector_subscribe_unsubscribe(mock_kline_callback):
    # This test requires a more involved setup with a mock WebSocket server
    # or deeper patching of the `websockets.connect` call.
    # For simplicity, we focus on the logic of forming subscription messages.
    
    # Mock config that enables only one stream for easier assertion
    class MinimalMockConfigManager:
        def get_config(self):
            return {
                "global_settings": {"v1_strategy": {"indicator_timeframes": ["1m"]}},
                "pairs": {"BTC_USDT": {"enabled": True, "contract_type": "USDT_M"}}
            }
        def register_callback(self, cb): pass
        def stop_watcher(self): pass

    ws_connector = BinanceWebSocketConnector(config_manager=MinimalMockConfigManager(), kline_callback=mock_kline_callback)
    
    mock_ws_connection = AsyncMock(spec=websockets.WebSocketClientProtocol)
    mock_ws_connection.open = True # Simulate open connection

    streams_to_sub = ["btcusdt@kline_1m"]
    await ws_connector._subscribe(mock_ws_connection, streams_to_sub)
    call_args = json.loads(mock_ws_connection.send.call_args[0][0])
    assert call_args["method"] == "SUBSCRIBE"
    assert call_args["params"] == streams_to_sub

    await ws_connector._unsubscribe(mock_ws_connection, streams_to_sub)
    call_args_unsub = json.loads(mock_ws_connection.send.call_args[0][0])
    assert call_args_unsub["method"] == "UNSUBSCRIBE"
    assert call_args_unsub["params"] == streams_to_sub

@pytest.mark.asyncio
async def test_ws_connector_process_kline_message(mock_config_manager_for_connectors, mock_kline_callback):
    ws_connector = BinanceWebSocketConnector(config_manager=mock_config_manager_for_connectors, kline_callback=mock_kline_callback)
    kline_msg_str = 	{	
        "e": "kline",
        "E": 123456789,
        "s": "BTCUSDT",
        "k": {
            "t": 123400000,
            "T": 123459999,
            "s": "BTCUSDT",
            "i": "1m",
            "o": "0.0010",
            "c": "0.0020",
            "h": "0.0025",
            "l": "0.0015",
            "v": "1000",
            "n": 100,
            "x": False, # Is kline closed?
            "q": "1.0000",
            "V": "500",
            "Q": "0.500",
            "B": "123456"
        }
    }	
    await ws_connector._process_message(json.dumps(kline_msg_str), "USDT_M")
    mock_kline_callback.assert_called_once()
    called_kline_arg = mock_kline_callback.call_args[0][0]
    assert isinstance(called_kline_arg, Kline)
    assert called_kline_arg.symbol == "BTCUSDT"
    assert called_kline_arg.interval == "1m"
    assert called_kline_arg.close == 0.0020
    assert called_kline_arg.is_closed is False

@pytest.mark.asyncio
async def test_ws_connector_start_stop_and_config_update_flow(mock_config_manager_for_connectors, mock_kline_callback):
    ws_connector = BinanceWebSocketConnector(config_manager=mock_config_manager_for_connectors, kline_callback=mock_kline_callback)
    
    # Patch the actual connection handler to prevent real WS connections during this unit test
    with patch.object(ws_connector, 	_handle_connection	, new_callable=AsyncMock) as mock_handler:
        await ws_connector.start()
        assert ws_connector._is_running
        # _handle_config_update should have been called, leading to _handle_connection calls
        # Check if tasks for USDT_M and COIN_M were created (based on config)
        await asyncio.sleep(0.1) # Allow tasks to be scheduled
        assert mock_handler.call_count >= 1 # Should be called for USDT_M and COIN_M if enabled
        
        # Simulate a config update that disables a pair
        new_config_data = mock_config_manager_for_connectors.get_config()
        new_config_data["pairs"]["BTC_USDT"]["enabled"] = False
        
        # Mock the _unsubscribe and _subscribe methods on an active connection if one existed
        # For this test, we mainly check if _handle_config_update is called and updates subscriptions
        mock_ws_conn_usdm = AsyncMock(spec=websockets.WebSocketClientProtocol)
        mock_ws_conn_usdm.open = True
        ws_connector._ws_connections["USDT_M"] = mock_ws_conn_usdm
        ws_connector._active_subscriptions["USDT_M"] = ["btcusdt@kline_1m", "btcusdt@kline_5m"]

        await ws_connector._handle_config_update(new_config_data) # Manually trigger for test control
        
        # Check if btcusdt streams were unsubscribed
        # This requires inspecting calls to _unsubscribe or checking _active_subscriptions
        assert "btcusdt@kline_1m" not in ws_connector._active_subscriptions["USDT_M"]
        assert "btcusdt@kline_5m" not in ws_connector._active_subscriptions["USDT_M"]
        # And that ethusdt@kline_1m is still there
        assert "ethusdt@kline_1m" in ws_connector._active_subscriptions["USDT_M"]

        await ws_connector.stop()
        assert not ws_connector._is_running
        mock_handler.reset_mock() # Reset for next potential calls if any

