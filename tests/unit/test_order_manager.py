import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from math import isclose

from src.config_loader import ConfigManager
from src.connectors import BinanceRESTClient
from src.order_manager import OrderManager
from src.models import TradeSignal, TradeDirection, OrderType, OrderSide, Position, OrderStatus

@pytest.fixture
def mock_config_manager_for_om():
    cm = MagicMock(spec=ConfigManager)
    cm.get_config.return_value = {
        "api": {"binance_api_key": "TEST_KEY", "binance_api_secret": "TEST_SECRET"},
        "global_settings": {
            "v1_strategy": {
                "default_margin_usdt": 50.0,
                "default_leverage": 10,
                "margin_mode": "ISOLATED"
            }
        },
        "pairs": {
            "BTC_USDT": {
                "enabled": True, 
                "contract_type": "USDT_M", 
                "margin_usdt": 100.0, # Override global
                "leverage": 20 # Override global
            },
            "ETHUSD_PERP": {
                "enabled": True, 
                "contract_type": "COIN_M", 
                "margin_coin": 0.1, # Interpreted as num_contracts for MVP test
                "leverage": 5
            }
        }
    }
    return cm

@pytest.fixture
def mock_rest_client_for_om():
    rest_client = AsyncMock(spec=BinanceRESTClient)
    # Mock exchange info for precision and limits
    rest_client.fetch_exchange_info.return_value = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "precision": {"amount": "0.00001", "price": "0.01"},
            "limits": {"cost": {"min": "5.0"}, "amount": {"min": "0.00001"}}
        },
        "ETHUSD_PERP": {
            "symbol": "ETHUSD_PERP",
            "precision": {"amount": "0.001", "price": "0.01"}, # COIN-M contracts might be integers or decimals
            "limits": {"cost": {"min": "5.0"}, "amount": {"min": "0.001"}}
        }
    }
    
    # Properly mock create_order to return a dictionary directly
    async def mock_create_order(symbol, order_type, side, amount, price=None, params={}):
        return {
            "id": f"mock_order_{symbol}_{int(asyncio.get_event_loop().time()*1000)}",
            "symbol": symbol,
            "status": "NEW", 
            "avgPrice": str(price or params.get("stopPrice") or (50000.0 if "BTC" in symbol else 3000.0)),
            "type": order_type
        }
    
    rest_client.create_order.side_effect = mock_create_order
    return rest_client

@pytest.fixture
def order_manager(mock_config_manager_for_om, mock_rest_client_for_om):
    om = OrderManager(config_manager=mock_config_manager_for_om, rest_client=mock_rest_client_for_om)
    # Prime the cache for exchange info to avoid repeated calls in tests unless specifically testing cache miss
    async def prime_cache():
        # Make sure the exchange_info_cache is properly set with the values we expect
        om.exchange_info_cache = {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "precision": {"amount": "0.00001", "price": "0.01"},
                "limits": {"cost": {"min": "5.0"}, "amount": {"min": "0.00001"}}
            },
            "ETHUSD_PERP": {
                "symbol": "ETHUSD_PERP",
                "precision": {"amount": "0.001", "price": "0.01"},
                "limits": {"cost": {"min": "5.0"}, "amount": {"min": "0.001"}}
            }
        }
    asyncio.run(prime_cache())
    return om

# Tests for Precision and Filter Logic
@pytest.mark.asyncio
async def test_om_adjust_quantity_to_precision(order_manager):
    """Test quantity adjustment to precision based on step size."""
    # Use almost equal for floating point comparisons
    
    result = order_manager._adjust_quantity_to_precision(0.123456, 0.001)
    assert isclose(result, 0.123, rel_tol=1e-10), f"Expected 0.123, got {result}"
    
    result = order_manager._adjust_quantity_to_precision(0.123, 0.0001)
    assert isclose(result, 0.123, rel_tol=1e-10), f"Expected 0.123, got {result}"
    
    result = order_manager._adjust_quantity_to_precision(123.456, 1.0)
    assert isclose(result, 123.0, rel_tol=1e-10), f"Expected 123.0, got {result}"
    
    # Expected to be 0.99999 (multiple of 0.00001)
    result = order_manager._adjust_quantity_to_precision(0.99999, 0.00001)
    assert isclose(result, 0.99999, rel_tol=1e-10), f"Expected 0.99999, got {result}"
    
    # Floor behavior - 0.000005 is less than 0.00001
    result = order_manager._adjust_quantity_to_precision(0.000005, 0.00001)
    assert isclose(result, 0.0, rel_tol=1e-10), f"Expected 0.0, got {result}"
    
    # Test with zero step size (should return original value)
    result = order_manager._adjust_quantity_to_precision(123.456, 0)
    assert isclose(result, 123.456, rel_tol=1e-10), f"Expected 123.456, got {result}"

@pytest.mark.asyncio
async def test_om_adjust_price_to_precision(order_manager):
    """Test price adjustment to precision based on tick size."""
    assert order_manager._adjust_price_to_precision(123.456, 0.01) == 123.46 # Rounds
    assert order_manager._adjust_price_to_precision(123.454, 0.01) == 123.45
    assert order_manager._adjust_price_to_precision(123.5, 1.0) == 124.0
    assert order_manager._adjust_price_to_precision(50000.557, 0.01) == 50000.56
    # Test with zero tick size (should return original value)
    assert order_manager._adjust_price_to_precision(123.456, 0) == 123.456

@pytest.mark.asyncio
async def test_om_get_symbol_filters(order_manager, mock_rest_client_for_om):
    """Test retrieving correct symbol filters from exchange info."""
    # Test for BTCUSDT
    lot_step, price_tick, min_notional = await order_manager._get_symbol_filters("BTCUSDT")
    assert lot_step == 0.00001
    assert price_tick == 0.01
    assert min_notional == 5.0
    
    # Test for ETHUSD_PERP
    lot_step, price_tick, min_notional = await order_manager._get_symbol_filters("ETHUSD_PERP")
    assert lot_step == 0.001
    assert price_tick == 0.01
    assert min_notional == 5.0
    
    # Verify cache is used (fetch_exchange_info should not be called again)
    mock_rest_client_for_om.fetch_exchange_info.reset_mock()
    await order_manager._get_symbol_filters("BTCUSDT")
    mock_rest_client_for_om.fetch_exchange_info.assert_not_called()

@pytest.mark.asyncio
async def test_om_get_symbol_filters_with_missing_data(order_manager, mock_rest_client_for_om):
    """Test handling missing filter data gracefully."""
    # Clear the cache to force a new fetch
    order_manager.exchange_info_cache = {}
    
    # Mock the response with missing data
    mock_rest_client_for_om.fetch_exchange_info.return_value = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            # Missing precision and limits data
        }
    }
    
    lot_step, price_tick, min_notional = await order_manager._get_symbol_filters("BTCUSDT")
    assert lot_step is None
    assert price_tick is None
    assert min_notional is None

@pytest.mark.asyncio
async def test_om_get_symbol_filters_fetch_failure(order_manager, mock_rest_client_for_om):
    """Test handling exchange info fetch failure."""
    # Clear the cache to force a new fetch
    order_manager.exchange_info_cache = {}
    
    # Mock fetch_exchange_info to return None (failure)
    mock_rest_client_for_om.fetch_exchange_info.return_value = None
    
    lot_step, price_tick, min_notional = await order_manager._get_symbol_filters("BTCUSDT")
    assert lot_step is None
    assert price_tick is None
    assert min_notional is None

# Tests for Position Sizing
@pytest.mark.asyncio
async def test_om_quantity_calculation_usdt_m(order_manager, mock_rest_client_for_om):
    """Test correct quantity calculation for USDT-M pair."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # Expected quantity: (100 USDT margin * 20x leverage) / 50000 entry = 0.04 BTC
    expected_quantity = 0.04
    
    assert mock_rest_client_for_om.create_order.call_count == 3
    entry_call = mock_rest_client_for_om.create_order.call_args_list[0]
    assert entry_call[1]["amount"] == expected_quantity

@pytest.mark.asyncio
async def test_om_quantity_calculation_coin_m(order_manager, mock_rest_client_for_om):
    """Test correct quantity calculation for COIN-M pair (MVP: margin_coin = number of contracts)."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    signal = TradeSignal(
        symbol="ETHUSD_PERP", config_symbol="ETHUSD_PERP", contract_type="COIN_M",
        direction=TradeDirection.LONG, entry_price=3000.0,
        stop_loss_price=2900.0, take_profit_price=3300.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # For COIN-M in MVP, margin_coin (0.1) is used directly as the quantity
    expected_quantity = 0.1
    
    assert mock_rest_client_for_om.create_order.call_count == 3
    entry_call = mock_rest_client_for_om.create_order.call_args_list[0]
    assert entry_call[1]["amount"] == expected_quantity

@pytest.mark.asyncio
async def test_om_missing_margin_configuration(order_manager, mock_rest_client_for_om):
    """Test handling missing margin configuration."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Create a new config with missing margin settings for BTC_USDT
    config = order_manager.config_manager.get_config.return_value.copy()
    config["pairs"]["BTC_USDT"].pop("margin_usdt")  # Remove margin_usdt
    config["global_settings"]["v1_strategy"].pop("default_margin_usdt")  # Remove default margin
    order_manager.config_manager.get_config.return_value = config
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # No orders should be placed due to missing margin configuration
    assert mock_rest_client_for_om.create_order.call_count == 0

@pytest.mark.asyncio
async def test_om_zero_entry_price(order_manager, mock_rest_client_for_om):
    """Test handling zero or invalid entry price."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=0.0,  # Zero entry price
        stop_loss_price=0.0, take_profit_price=0.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # No orders should be placed due to invalid entry price
    assert mock_rest_client_for_om.create_order.call_count == 0

# Tests for Order Placement Logic
@pytest.mark.asyncio
async def test_om_handle_trade_signal_usdt_m_long(order_manager, mock_rest_client_for_om):
    """Test correct order parameters for USDT-M LONG trade."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # Expected quantity: (100 USDT margin * 20x leverage) / 50000 entry = 0.04 BTC
    expected_quantity = 0.04
    
    # Check create_order calls
    assert mock_rest_client_for_om.create_order.call_count == 3
    
    # Check entry order parameters
    entry_call = mock_rest_client_for_om.create_order.call_args_list[0]
    assert entry_call[1]["symbol"] == "BTCUSDT"
    assert entry_call[1]["order_type"] == OrderType.MARKET
    assert entry_call[1]["side"] == OrderSide.BUY
    assert entry_call[1]["amount"] == expected_quantity
    
    # Check SL order parameters
    sl_call = mock_rest_client_for_om.create_order.call_args_list[1]
    assert sl_call[1]["symbol"] == "BTCUSDT"
    assert sl_call[1]["order_type"] == OrderType.STOP_MARKET
    assert sl_call[1]["side"] == OrderSide.SELL
    assert sl_call[1]["amount"] == expected_quantity
    assert sl_call[1]["params"]["stopPrice"] == 49000.00  # Adjusted to 0.01 tick
    assert sl_call[1]["params"]["reduceOnly"] == "true"
    
    # Check TP order parameters
    tp_call = mock_rest_client_for_om.create_order.call_args_list[2]
    assert tp_call[1]["symbol"] == "BTCUSDT"
    assert tp_call[1]["order_type"] == OrderType.TAKE_PROFIT_MARKET
    assert tp_call[1]["side"] == OrderSide.SELL
    assert tp_call[1]["amount"] == expected_quantity
    assert tp_call[1]["params"]["stopPrice"] == 53000.00  # Adjusted to 0.01 tick
    assert tp_call[1]["params"]["reduceOnly"] == "true"

@pytest.mark.asyncio
async def test_om_handle_trade_signal_usdt_m_short(order_manager, mock_rest_client_for_om):
    """Test correct order parameters for USDT-M SHORT trade."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.SHORT, entry_price=50000.0,
        stop_loss_price=51000.0, take_profit_price=47000.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # Expected quantity: (100 USDT margin * 20x leverage) / 50000 entry = 0.04 BTC
    expected_quantity = 0.04
    
    # Check create_order calls
    assert mock_rest_client_for_om.create_order.call_count == 3
    
    # Check entry order parameters
    entry_call = mock_rest_client_for_om.create_order.call_args_list[0]
    assert entry_call[1]["symbol"] == "BTCUSDT"
    assert entry_call[1]["order_type"] == OrderType.MARKET
    assert entry_call[1]["side"] == OrderSide.SELL
    assert entry_call[1]["amount"] == expected_quantity
    
    # Check SL order parameters
    sl_call = mock_rest_client_for_om.create_order.call_args_list[1]
    assert sl_call[1]["symbol"] == "BTCUSDT"
    assert sl_call[1]["order_type"] == OrderType.STOP_MARKET
    assert sl_call[1]["side"] == OrderSide.BUY
    assert sl_call[1]["amount"] == expected_quantity
    assert sl_call[1]["params"]["stopPrice"] == 51000.00  # Adjusted to 0.01 tick
    assert sl_call[1]["params"]["reduceOnly"] == "true"
    
    # Check TP order parameters
    tp_call = mock_rest_client_for_om.create_order.call_args_list[2]
    assert tp_call[1]["symbol"] == "BTCUSDT"
    assert tp_call[1]["order_type"] == OrderType.TAKE_PROFIT_MARKET
    assert tp_call[1]["side"] == OrderSide.BUY
    assert tp_call[1]["amount"] == expected_quantity
    assert tp_call[1]["params"]["stopPrice"] == 47000.00  # Adjusted to 0.01 tick
    assert tp_call[1]["params"]["reduceOnly"] == "true"

@pytest.mark.asyncio
async def test_om_handle_trade_signal_coin_m_short(order_manager, mock_rest_client_for_om):
    """Test correct order parameters for COIN-M SHORT trade."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    signal = TradeSignal(
        symbol="ETHUSD_PERP", config_symbol="ETHUSD_PERP", contract_type="COIN_M",
        direction=TradeDirection.SHORT, entry_price=3000.0,
        stop_loss_price=3100.0, take_profit_price=2700.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # Expected quantity for COIN-M (margin_coin=0.1 is used as num_contracts for MVP test)
    expected_quantity = 0.1
    
    assert mock_rest_client_for_om.create_order.call_count == 3
    
    # Check entry order parameters
    entry_call = mock_rest_client_for_om.create_order.call_args_list[0]
    assert entry_call[1]["symbol"] == "ETHUSD_PERP"
    assert entry_call[1]["order_type"] == OrderType.MARKET
    assert entry_call[1]["side"] == OrderSide.SELL
    assert entry_call[1]["amount"] == expected_quantity
    
    # Check SL order parameters
    sl_call = mock_rest_client_for_om.create_order.call_args_list[1]
    assert sl_call[1]["symbol"] == "ETHUSD_PERP"
    assert sl_call[1]["order_type"] == OrderType.STOP_MARKET
    assert sl_call[1]["side"] == OrderSide.BUY
    assert sl_call[1]["amount"] == expected_quantity
    assert sl_call[1]["params"]["stopPrice"] == 3100.00
    assert sl_call[1]["params"]["reduceOnly"] == "true"
    
    # Check TP order parameters
    tp_call = mock_rest_client_for_om.create_order.call_args_list[2]
    assert tp_call[1]["symbol"] == "ETHUSD_PERP"
    assert tp_call[1]["order_type"] == OrderType.TAKE_PROFIT_MARKET
    assert tp_call[1]["side"] == OrderSide.BUY
    assert tp_call[1]["amount"] == expected_quantity
    assert tp_call[1]["params"]["stopPrice"] == 2700.00
    assert tp_call[1]["params"]["reduceOnly"] == "true"

@pytest.mark.asyncio
async def test_om_min_notional_check_usdt_m(order_manager, mock_rest_client_for_om):
    """Test minimum notional value check for USDT-M."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Test case that passes min notional check
    signal_high_notional = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=1.0,
        stop_loss_price=0.9, take_profit_price=1.3
    )
    await order_manager.handle_trade_signal(signal_high_notional)
    assert mock_rest_client_for_om.create_order.call_count == 3  # Orders should be placed
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Test case that fails min notional check by modifying config
    # Set margin_usdt and leverage so that margin_usdt * leverage < min_notional
    order_manager.config_manager.get_config.return_value["pairs"]["BTC_USDT"]["margin_usdt"] = 0.1
    order_manager.config_manager.get_config.return_value["pairs"]["BTC_USDT"]["leverage"] = 10
    # Now notional is 0.1 * 10 = 1.0 USDT. Min notional is 5.0. Should fail.
    
    signal_low_notional = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    await order_manager.handle_trade_signal(signal_low_notional)
    
    # No orders should be placed due to min notional check failure
    assert mock_rest_client_for_om.create_order.call_count == 0
    
    # Reset config for other tests
    order_manager.config_manager.get_config.return_value["pairs"]["BTC_USDT"]["margin_usdt"] = 100.0
    order_manager.config_manager.get_config.return_value["pairs"]["BTC_USDT"]["leverage"] = 20

@pytest.mark.asyncio
async def test_om_handle_trade_signal_zero_quantity(order_manager, mock_rest_client_for_om):
    """Test handling zero calculated quantity."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Mock _get_symbol_filters to return a larger step_size that will cause quantity to be floored to zero
    with patch.object(order_manager, '_get_symbol_filters', new_callable=AsyncMock) as mock_filters:
        mock_filters.return_value = (0.001, 0.01, 5.0)  # Lot_step = 0.001
        
        signal = TradeSignal(
            symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
            direction=TradeDirection.LONG, entry_price=10000000.0,  # Very high price
            stop_loss_price=9000000.0, take_profit_price=13000000.0
        )
        
        # Expected raw quantity: (100 * 20) / 10000000 = 0.0002
        # Adjusted with step 0.001: floor(0.0002 / 0.001) * 0.001 = 0.0
        await order_manager.handle_trade_signal(signal)
        
        # No orders should be placed due to zero quantity
        assert mock_rest_client_for_om.create_order.call_count == 0

# Tests for Error Handling
@pytest.mark.asyncio
async def test_om_exchange_info_fetch_failure(order_manager, mock_rest_client_for_om):
    """Test handling failure to fetch exchange info."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Clear the cache to force a new fetch
    order_manager.exchange_info_cache = {}
    
    # Mock exchange_info fetch to return None (failure)
    mock_rest_client_for_om.fetch_exchange_info.return_value = None
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # No orders should be placed due to exchange info fetch failure
    assert mock_rest_client_for_om.create_order.call_count == 0

@pytest.mark.asyncio
async def test_om_entry_order_placement_failure(order_manager, mock_rest_client_for_om):
    """Test handling entry order placement failure."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Mock the first call to create_order (entry order) to return None (failure)
    mock_rest_client_for_om.create_order.side_effect = [
        None,  # Entry order fails
        {"id": "mock_sl_order_id"},  # SL order (shouldn't be called)
        {"id": "mock_tp_order_id"}   # TP order (shouldn't be called)
    ]
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # Only entry order should be attempted, and it fails
    assert mock_rest_client_for_om.create_order.call_count == 1

@pytest.mark.asyncio
async def test_om_entry_order_rejected(order_manager, mock_rest_client_for_om):
    """Test handling entry order rejection."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Mock the first call to create_order (entry order) to return a rejected status
    mock_rest_client_for_om.create_order.side_effect = [
        {"id": "mock_entry_order_id", "status": OrderStatus.REJECTED},  # Entry order rejected
        {"id": "mock_sl_order_id"},  # SL order (shouldn't be called)
        {"id": "mock_tp_order_id"}   # TP order (shouldn't be called)
    ]
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # Only entry order should be attempted, but SL/TP are not placed due to rejected entry
    assert mock_rest_client_for_om.create_order.call_count == 1

@pytest.mark.asyncio
async def test_om_sl_order_placement_failure(order_manager, mock_rest_client_for_om):
    """Test handling SL order placement failure."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Mock create_order to succeed for entry but fail for SL
    mock_rest_client_for_om.create_order.side_effect = [
        {"id": "mock_entry_order_id", "status": "NEW"},  # Entry order succeeds
        None,  # SL order fails
        {"id": "mock_tp_order_id"}   # TP order should still be attempted
    ]
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # Entry order succeeds, SL fails, but TP is still attempted
    assert mock_rest_client_for_om.create_order.call_count == 3

@pytest.mark.asyncio
async def test_om_tp_order_placement_failure(order_manager, mock_rest_client_for_om):
    """Test handling TP order placement failure."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Mock create_order to succeed for entry and SL but fail for TP
    mock_rest_client_for_om.create_order.side_effect = [
        {"id": "mock_entry_order_id", "status": "NEW"},  # Entry order succeeds
        {"id": "mock_sl_order_id"},   # SL order succeeds
        None  # TP order fails
    ]
    
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    
    await order_manager.handle_trade_signal(signal)
    
    # All three orders should be attempted, but TP fails
    assert mock_rest_client_for_om.create_order.call_count == 3

@pytest.mark.asyncio
async def test_om_unknown_contract_type(order_manager, mock_rest_client_for_om):
    """Test handling unknown contract type."""
    mock_rest_client_for_om.create_order.reset_mock()
    
    # Use a valid contract type but modify handle_trade_signal behavior
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    
    # Patch the signal's contract_type attribute after creation to simulate an unknown type
    # This avoids the Pydantic validation error
    with patch.object(signal, 'contract_type', "UNKNOWN_TYPE"):
        await order_manager.handle_trade_signal(signal)
    
    # No orders should be placed due to unknown contract type
    assert mock_rest_client_for_om.create_order.call_count == 0

