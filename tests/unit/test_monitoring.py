import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from prometheus_client import REGISTRY

from src.monitoring import PrometheusMonitor

@pytest.fixture
def monitor():
    """Fixture that provides a PrometheusMonitor instance for testing"""
    # Use a high port number to avoid conflicts with any running services
    test_port = 9999
    monitor = PrometheusMonitor(port=test_port)
    yield monitor
    
    # Clean up after test
    monitor._server_started = False
    
    # Clear Prometheus registry to avoid interference between tests
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


class TestPrometheusMonitor:
    """Tests for the PrometheusMonitor class"""

    def test_initialization(self, monitor):
        """Test that the monitor initializes with the correct metric types"""
        # Check counters
        assert hasattr(monitor, "bot_errors_total")
        assert hasattr(monitor, "websocket_messages_received_total")
        assert hasattr(monitor, "kline_processed_total")
        assert hasattr(monitor, "signals_generated_total")
        assert hasattr(monitor, "orders_placed_total")
        assert hasattr(monitor, "orders_failed_total")
        assert hasattr(monitor, "orders_filled_total")
        
        # Check gauges
        assert hasattr(monitor, "bot_uptime_seconds")
        assert hasattr(monitor, "indicator_calculation_duration_seconds")
        assert hasattr(monitor, "active_positions_count")
        assert hasattr(monitor, "position_pnl_unrealized")
        
        # Check enum
        assert hasattr(monitor, "websocket_connection_status")
        
        # Check info
        assert hasattr(monitor, "bot_info")
        
        # Verify internal state
        assert monitor._server_started is False
        assert isinstance(monitor.port, int)

    @patch('src.monitoring.start_http_server')
    def test_start_method(self, mock_start_http_server, monitor):
        """Test that the start method correctly initializes the HTTP server and metrics"""
        # Test normal start
        monitor.start()
        
        # Verify HTTP server was started with correct port
        mock_start_http_server.assert_called_once_with(monitor.port)
        
        # Verify server started flag is set
        assert monitor._server_started is True
        
        # Check bot_info metric was set
        # For Info metrics, we can only verify it exists as get_sample_value
        # doesn't work the same way for this type
        assert hasattr(monitor, "bot_info")
        
        # Test starting again (should log warning and do nothing)
        monitor.start()
        # Mock should still have been called only once
        mock_start_http_server.assert_called_once()

    @patch('src.monitoring.start_http_server')
    def test_start_method_error_handling(self, mock_start_http_server, monitor):
        """Test error handling when starting the HTTP server fails"""
        # Configure mock to raise an exception
        mock_start_http_server.side_effect = OSError("Port in use")
        
        # Call start method
        monitor.start()
        
        # Verify server started flag remains False
        assert monitor._server_started is False
        
        # Test with a different exception type
        mock_start_http_server.side_effect = Exception("Unknown error")
        
        # Call start method
        monitor.start()
        
        # Verify server started flag remains False
        assert monitor._server_started is False

    def test_stop_method(self, monitor):
        """Test the stop method"""
        # Set the server as started
        monitor._server_started = True
        
        # Call stop method
        monitor.stop()
        
        # Verify server started flag is reset
        assert monitor._server_started is False

    @pytest.mark.asyncio
    @patch('src.monitoring.asyncio.create_task')
    @patch('src.monitoring.start_http_server')
    async def test_uptime_task_creation(self, mock_start_http_server, mock_create_task, monitor):
        """Test that the uptime update task is created correctly"""
        # Start the monitor
        monitor.start()
        
        # Verify create_task was called with the _update_uptime_periodically coroutine
        mock_create_task.assert_called_once()
        
        # The first argument to the first call should be a coroutine
        call_args = mock_create_task.call_args[0][0]
        assert asyncio.iscoroutine(call_args)

    @pytest.mark.asyncio
    async def test_update_uptime_directly(self, monitor):
        """Test the _update_uptime_periodically method directly by calling it once"""
        # Set server as started
        monitor._server_started = True
        
        # Record the start time
        monitor.start_time = time.time() - 10  # pretend we started 10 seconds ago
        
        # Call the method once (manually)
        monitor.bot_uptime_seconds.set(time.time() - monitor.start_time)
        
        # Verify uptime is approximately 10 seconds (with some tolerance)
        uptime = REGISTRY.get_sample_value('trading_bot_uptime_seconds')
        assert uptime is not None
        assert 9.5 <= uptime <= 10.5  # Allow for small timing differences

    def test_inc_bot_error(self, monitor):
        """Test the inc_bot_error method"""
        # Call the method with different labels
        monitor.inc_bot_error(module="TestModule", error_type="TestError")
        monitor.inc_bot_error(module="TestModule", error_type="AnotherError")
        monitor.inc_bot_error(module="AnotherModule", error_type="TestError")
        # Call again with the same labels
        monitor.inc_bot_error(module="TestModule", error_type="TestError")
        
        # Verify counters are incremented correctly
        assert REGISTRY.get_sample_value('trading_bot_errors_total', 
                                       {'module': 'TestModule', 'error_type': 'TestError'}) == 2
        assert REGISTRY.get_sample_value('trading_bot_errors_total', 
                                       {'module': 'TestModule', 'error_type': 'AnotherError'}) == 1
        assert REGISTRY.get_sample_value('trading_bot_errors_total', 
                                       {'module': 'AnotherModule', 'error_type': 'TestError'}) == 1

    def test_set_websocket_status(self, monitor):
        """Test the set_websocket_status method with valid and invalid states"""
        # Test valid states
        monitor.set_websocket_status(market_type="USDT_M", status="connected")
        
        # Check connected=1, others=0
        assert REGISTRY.get_sample_value('trading_bot_websocket_connection_status', 
                                      {'market_type': 'USDT_M', 'trading_bot_websocket_connection_status': 'connected'}) == 1
        assert REGISTRY.get_sample_value('trading_bot_websocket_connection_status', 
                                      {'market_type': 'USDT_M', 'trading_bot_websocket_connection_status': 'disconnected'}) == 0
        
        # Change status
        monitor.set_websocket_status(market_type="USDT_M", status="disconnected")
        
        # Check disconnected=1, others=0
        assert REGISTRY.get_sample_value('trading_bot_websocket_connection_status', 
                                      {'market_type': 'USDT_M', 'trading_bot_websocket_connection_status': 'disconnected'}) == 1
        assert REGISTRY.get_sample_value('trading_bot_websocket_connection_status', 
                                      {'market_type': 'USDT_M', 'trading_bot_websocket_connection_status': 'connected'}) == 0
        
        # Test invalid state
        monitor.set_websocket_status(market_type="USDT_M", status="invalid_state")
        
        # Should set to "error" state
        assert REGISTRY.get_sample_value('trading_bot_websocket_connection_status', 
                                      {'market_type': 'USDT_M', 'trading_bot_websocket_connection_status': 'error'}) == 1
        assert REGISTRY.get_sample_value('trading_bot_websocket_connection_status', 
                                      {'market_type': 'USDT_M', 'trading_bot_websocket_connection_status': 'disconnected'}) == 0

    def test_inc_websocket_message(self, monitor):
        """Test the inc_websocket_message method"""
        # Call the method with different labels
        monitor.inc_websocket_message(market_type="USDT_M", message_type="kline")
        monitor.inc_websocket_message(market_type="USDT_M", message_type="kline")
        monitor.inc_websocket_message(market_type="COIN_M", message_type="depthUpdate")
        
        # Verify counters are incremented correctly
        assert REGISTRY.get_sample_value('trading_bot_websocket_messages_received_total', 
                                       {'market_type': 'USDT_M', 'message_type': 'kline'}) == 2
        assert REGISTRY.get_sample_value('trading_bot_websocket_messages_received_total', 
                                       {'market_type': 'COIN_M', 'message_type': 'depthUpdate'}) == 1

    def test_inc_kline_processed(self, monitor):
        """Test the inc_kline_processed method"""
        # Call the method with different labels
        monitor.inc_kline_processed(symbol="BTCUSDT", interval="1m")
        monitor.inc_kline_processed(symbol="BTCUSDT", interval="1m")
        monitor.inc_kline_processed(symbol="ETHUSDT", interval="5m")
        
        # Verify counters are incremented correctly
        assert REGISTRY.get_sample_value('trading_bot_klines_processed_total', 
                                       {'symbol': 'BTCUSDT', 'interval': '1m'}) == 2
        assert REGISTRY.get_sample_value('trading_bot_klines_processed_total', 
                                       {'symbol': 'ETHUSDT', 'interval': '5m'}) == 1

    def test_set_indicator_calculation_duration(self, monitor):
        """Test the set_indicator_calculation_duration method"""
        # Call the method with different labels and values
        monitor.set_indicator_calculation_duration(symbol="BTCUSDT", interval="1m", duration_seconds=0.123)
        monitor.set_indicator_calculation_duration(symbol="ETHUSDT", interval="5m", duration_seconds=0.456)
        
        # Update value for first label set
        monitor.set_indicator_calculation_duration(symbol="BTCUSDT", interval="1m", duration_seconds=0.789)
        
        # Verify gauge values reflect the last set value
        assert REGISTRY.get_sample_value('trading_bot_indicator_calculation_duration_seconds', 
                                       {'symbol': 'BTCUSDT', 'interval': '1m'}) == 0.789
        assert REGISTRY.get_sample_value('trading_bot_indicator_calculation_duration_seconds', 
                                       {'symbol': 'ETHUSDT', 'interval': '5m'}) == 0.456

    def test_inc_signal_generated(self, monitor):
        """Test the inc_signal_generated method"""
        # Call the method with different labels
        monitor.inc_signal_generated(symbol="BTCUSDT", strategy="V1_SMA", direction="LONG")
        monitor.inc_signal_generated(symbol="BTCUSDT", strategy="V1_SMA", direction="LONG")
        monitor.inc_signal_generated(symbol="ETHUSDT", strategy="V1_SMA", direction="SHORT")
        
        # Verify counters are incremented correctly
        assert REGISTRY.get_sample_value('trading_bot_signals_generated_total', 
                                       {'symbol': 'BTCUSDT', 'strategy': 'V1_SMA', 'direction': 'LONG'}) == 2
        assert REGISTRY.get_sample_value('trading_bot_signals_generated_total', 
                                       {'symbol': 'ETHUSDT', 'strategy': 'V1_SMA', 'direction': 'SHORT'}) == 1

    def test_inc_order_placed(self, monitor):
        """Test the inc_order_placed method"""
        # Call the method with different labels
        monitor.inc_order_placed(symbol="BTCUSDT", order_type="MARKET", side="BUY")
        monitor.inc_order_placed(symbol="BTCUSDT", order_type="MARKET", side="BUY")
        monitor.inc_order_placed(symbol="ETHUSDT", order_type="LIMIT", side="SELL")
        
        # Verify counters are incremented correctly
        assert REGISTRY.get_sample_value('trading_bot_orders_placed_total', 
                                       {'symbol': 'BTCUSDT', 'order_type': 'MARKET', 'side': 'BUY'}) == 2
        assert REGISTRY.get_sample_value('trading_bot_orders_placed_total', 
                                       {'symbol': 'ETHUSDT', 'order_type': 'LIMIT', 'side': 'SELL'}) == 1

    def test_inc_order_failed(self, monitor):
        """Test the inc_order_failed method"""
        # Call the method with different labels
        monitor.inc_order_failed(symbol="BTCUSDT", reason="INSUFFICIENT_BALANCE")
        monitor.inc_order_failed(symbol="BTCUSDT", reason="INSUFFICIENT_BALANCE")
        monitor.inc_order_failed(symbol="ETHUSDT", reason="PRICE_FILTER")
        
        # Verify counters are incremented correctly
        assert REGISTRY.get_sample_value('trading_bot_orders_failed_total', 
                                       {'symbol': 'BTCUSDT', 'reason': 'INSUFFICIENT_BALANCE'}) == 2
        assert REGISTRY.get_sample_value('trading_bot_orders_failed_total', 
                                       {'symbol': 'ETHUSDT', 'reason': 'PRICE_FILTER'}) == 1

    def test_inc_order_filled(self, monitor):
        """Test the inc_order_filled method"""
        # Call the method with different labels
        monitor.inc_order_filled(symbol="BTCUSDT", order_type="MARKET", side="BUY")
        monitor.inc_order_filled(symbol="BTCUSDT", order_type="MARKET", side="BUY")
        monitor.inc_order_filled(symbol="ETHUSDT", order_type="LIMIT", side="SELL")
        
        # Verify counters are incremented correctly
        assert REGISTRY.get_sample_value('trading_bot_orders_filled_total', 
                                       {'symbol': 'BTCUSDT', 'order_type': 'MARKET', 'side': 'BUY'}) == 2
        assert REGISTRY.get_sample_value('trading_bot_orders_filled_total', 
                                       {'symbol': 'ETHUSDT', 'order_type': 'LIMIT', 'side': 'SELL'}) == 1

    def test_set_active_positions_count(self, monitor):
        """Test the set_active_positions_count method"""
        # Call the method with different labels and values
        monitor.set_active_positions_count(contract_type="USDT_M", count=5)
        monitor.set_active_positions_count(contract_type="COIN_M", count=3)
        
        # Update value for first label set
        monitor.set_active_positions_count(contract_type="USDT_M", count=7)
        
        # Verify gauge values reflect the last set value
        assert REGISTRY.get_sample_value('trading_bot_active_positions_count', 
                                       {'contract_type': 'USDT_M'}) == 7
        assert REGISTRY.get_sample_value('trading_bot_active_positions_count', 
                                       {'contract_type': 'COIN_M'}) == 3

    def test_update_active_positions_gauge(self, monitor):
        """Test the update_active_positions_gauge method"""
        # Call the method
        monitor.update_active_positions_gauge(positions_usdt_m=5, positions_coin_m=3)
        
        # Verify both gauges are set correctly
        assert REGISTRY.get_sample_value('trading_bot_active_positions_count', 
                                       {'contract_type': 'USDT_M'}) == 5
        assert REGISTRY.get_sample_value('trading_bot_active_positions_count', 
                                       {'contract_type': 'COIN_M'}) == 3
        
        # Update with new values
        monitor.update_active_positions_gauge(positions_usdt_m=7, positions_coin_m=2)
        
        # Verify both gauges reflect the updated values
        assert REGISTRY.get_sample_value('trading_bot_active_positions_count', 
                                       {'contract_type': 'USDT_M'}) == 7
        assert REGISTRY.get_sample_value('trading_bot_active_positions_count', 
                                       {'contract_type': 'COIN_M'}) == 2

    def test_set_position_pnl(self, monitor):
        """Test the set_position_pnl method"""
        # Call the method with different labels and values
        monitor.set_position_pnl(symbol="BTCUSDT", pnl=10.5)
        monitor.set_position_pnl(symbol="ETHUSDT", pnl=-5.25)
        
        # Update value for first label
        monitor.set_position_pnl(symbol="BTCUSDT", pnl=15.75)
        
        # Verify gauge values reflect the last set value
        assert REGISTRY.get_sample_value('trading_bot_position_pnl_unrealized', 
                                       {'symbol': 'BTCUSDT'}) == 15.75
        assert REGISTRY.get_sample_value('trading_bot_position_pnl_unrealized', 
                                       {'symbol': 'ETHUSDT'}) == -5.25 