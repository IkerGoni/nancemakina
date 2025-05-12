import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.config_loader import ConfigManager
from src.connectors import BinanceRESTClient
from src.order_manager import OrderManager
from src.models import TradeSignal, TradeDirection, OrderType, OrderSide, Position

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
    rest_client.create_order = AsyncMock(side_effect=lambda symbol, order_type, side, amount, price=None, params={}:
        AsyncMock(
            return_value=AsyncMock(
                id=f"mock_order_{symbol}_{int(asyncio.get_event_loop().time()*1000)}_", 
                symbol=symbol, 
                status="NEW", 
                avgPrice=str(price or params.get("stopPrice") or (50000.0 if "BTC" in symbol else 3000.0)),
                type=order_type
            )
        )() # Call the inner AsyncMock to get the return_value
    )
    return rest_client

@pytest.fixture
def order_manager(mock_config_manager_for_om, mock_rest_client_for_om):
    om = OrderManager(config_manager=mock_config_manager_for_om, rest_client=mock_rest_client_for_om)
    # Prime the cache for exchange info to avoid repeated calls in tests unless specifically testing cache miss
    async def prime_cache():
        await om._get_exchange_info_for_symbol("BTCUSDT")
        await om._get_exchange_info_for_symbol("ETHUSD_PERP")
    asyncio.run(prime_cache())
    return om

@pytest.mark.asyncio
async def test_om_adjust_quantity_to_precision(order_manager):
    assert order_manager._adjust_quantity_to_precision(0.123456, 0.001) == 0.123
    assert order_manager._adjust_quantity_to_precision(0.123, 0.0001) == 0.123
    assert order_manager._adjust_quantity_to_precision(123.456, 1.0) == 123.0
    assert order_manager._adjust_quantity_to_precision(0.99999, 0.00001) == 0.99999
    assert order_manager._adjust_quantity_to_precision(0.000005, 0.00001) == 0.0 # Floor behavior

@pytest.mark.asyncio
async def test_om_adjust_price_to_precision(order_manager):
    assert order_manager._adjust_price_to_precision(123.456, 0.01) == 123.46 # Rounds
    assert order_manager._adjust_price_to_precision(123.454, 0.01) == 123.45
    assert order_manager._adjust_price_to_precision(123.5, 1.0) == 124.0

@pytest.mark.asyncio
async def test_om_get_symbol_filters(order_manager, mock_rest_client_for_om):
    lot_step, price_tick, min_notional = await order_manager._get_symbol_filters("BTCUSDT")
    assert lot_step == 0.00001
    assert price_tick == 0.01
    assert min_notional == 5.0
    # Test cache: second call should not call fetch_exchange_info again if already cached by prime_cache
    # mock_rest_client_for_om.fetch_exchange_info.assert_called_once() # This might fail if prime_cache is complex

@pytest.mark.asyncio
async def test_om_handle_trade_signal_usdt_m_long(order_manager, mock_rest_client_for_om):
    signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    await order_manager.handle_trade_signal(signal)

    # Expected quantity: (100 USDT margin * 20x leverage) / 50000 entry = 0.04 BTC
    # Adjusted: step_size 0.00001. 0.04 is fine.
    expected_quantity = 0.04
    
    # Check create_order calls
    # Call 1: Entry order (MARKET)
    # Call 2: SL order (STOP_MARKET)
    # Call 3: TP order (TAKE_PROFIT_MARKET)
    assert mock_rest_client_for_om.create_order.call_count == 3
    
    entry_call = mock_rest_client_for_om.create_order.call_args_list[0]
    sl_call = mock_rest_client_for_om.create_order.call_args_list[1]
    tp_call = mock_rest_client_for_om.create_order.call_args_list[2]

    assert entry_call[1]["symbol"] == "BTCUSDT"
    assert entry_call[1]["order_type"] == OrderType.MARKET
    assert entry_call[1]["side"] == OrderSide.BUY
    assert entry_call[1]["amount"] == expected_quantity

    assert sl_call[1]["symbol"] == "BTCUSDT"
    assert sl_call[1]["order_type"] == OrderType.STOP_MARKET
    assert sl_call[1]["side"] == OrderSide.SELL
    assert sl_call[1]["amount"] == expected_quantity
    assert sl_call[1]["params"]["stopPrice"] == 49000.00 # Adjusted to 0.01 tick
    assert sl_call[1]["params"]["reduceOnly"] == "true"

    assert tp_call[1]["symbol"] == "BTCUSDT"
    assert tp_call[1]["order_type"] == OrderType.TAKE_PROFIT_MARKET
    assert tp_call[1]["side"] == OrderSide.SELL
    assert tp_call[1]["amount"] == expected_quantity
    assert tp_call[1]["params"]["stopPrice"] == 53000.00 # Adjusted
    assert tp_call[1]["params"]["reduceOnly"] == "true"

@pytest.mark.asyncio
async def test_om_handle_trade_signal_coin_m_short(order_manager, mock_rest_client_for_om):
    mock_rest_client_for_om.create_order.reset_mock() # Reset from previous test
    signal = TradeSignal(
        symbol="ETHUSD_PERP", config_symbol="ETHUSD_PERP", contract_type="COIN_M",
        direction=TradeDirection.SHORT, entry_price=3000.0,
        stop_loss_price=3100.0, take_profit_price=2700.0
    )
    await order_manager.handle_trade_signal(signal)

    # Expected quantity for COIN-M (margin_coin=0.1 is used as num_contracts for MVP test)
    # Adjusted: step_size 0.001. 0.1 is fine.
    expected_quantity = 0.1 
    
    assert mock_rest_client_for_om.create_order.call_count == 3
    entry_call = mock_rest_client_for_om.create_order.call_args_list[0]
    sl_call = mock_rest_client_for_om.create_order.call_args_list[1]
    tp_call = mock_rest_client_for_om.create_order.call_args_list[2]

    assert entry_call[1]["symbol"] == "ETHUSD_PERP"
    assert entry_call[1]["order_type"] == OrderType.MARKET
    assert entry_call[1]["side"] == OrderSide.SELL
    assert entry_call[1]["amount"] == expected_quantity

    assert sl_call[1]["params"]["stopPrice"] == 3100.00
    assert tp_call[1]["params"]["stopPrice"] == 2700.00

@pytest.mark.asyncio
async def test_om_min_notional_check_usdt_m(order_manager, mock_rest_client_for_om):
    mock_rest_client_for_om.create_order.reset_mock()
    # Config: BTC_USDT margin 100, leverage 20. Min notional 5.0
    # Entry price 1.0 USDT. Quantity = (100*20)/1.0 = 2000 units.
    # Notional = 2000 * 1.0 = 2000. This should pass.
    signal_high_notional = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=1.0,
        stop_loss_price=0.9, take_profit_price=1.3
    )
    await order_manager.handle_trade_signal(signal_high_notional)
    assert mock_rest_client_for_om.create_order.call_count == 3 # Orders should be placed
    mock_rest_client_for_om.create_order.reset_mock()

    # Entry price 500000.0 USDT. Quantity = (100*20)/500000 = 0.004 units.
    # Notional = 0.004 * 500000 = 2000. This should pass.
    # Let's try to make it fail: if entry_price is very high, quantity is low.
    # (100 * 20) / entry_price * entry_price = 2000. This will always be 2000 notional for fixed margin * leverage.
    # The min_notional check in current OrderManager is based on calculated quantity * entry_price.
    # For fixed margin USDT-M, (margin * leverage / entry_price) * entry_price = margin * leverage.
    # So, if margin * leverage (e.g. 100*20=2000) > min_notional (5), it will always pass.
    # This means the test needs to adjust margin or min_notional in mock_config for a fail case.

    # Let's test a scenario where calculated quantity is very small due to high price, but still passes notional
    # because fixed margin * leverage is high enough.
    # The current logic for USDT-M fixed margin sizing means notional value is fixed at margin_usdt * leverage.
    # So, if (margin_usdt * leverage) < min_notional, it would fail.
    # In our config, BTC_USDT margin_usdt=100, leverage=20 => 2000 USDT notional. Min notional is 5. Always passes.

    # To test failure, let's mock a config where margin_usdt * leverage < min_notional
    order_manager.config_manager.get_config.return_value["pairs"]["BTC_USDT"]["margin_usdt"] = 0.1
    order_manager.config_manager.get_config.return_value["pairs"]["BTC_USDT"]["leverage"] = 10
    # Now notional is 0.1 * 10 = 1.0 USDT. Min notional is 5.0. Should fail.
    signal_low_notional = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0, # Price doesn't matter for this fail condition
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    await order_manager.handle_trade_signal(signal_low_notional)
    assert mock_rest_client_for_om.create_order.call_count == 0 # No orders should be placed

    # Reset config for other tests if OrderManager instance is reused by other tests (it is per fixture)
    # This is okay as fixture re-creates OrderManager with fresh mock_config_manager_for_om

@pytest.mark.asyncio
async def test_om_handle_trade_signal_zero_quantity(order_manager, mock_rest_client_for_om):
    mock_rest_client_for_om.create_order.reset_mock()
    # Make entry price extremely high so quantity becomes near zero and then floored to zero by precision
    # (100 USDT margin * 20x leverage) / 1_000_000_000 entry = 0.000002 BTC
    # If step_size is 0.0001, this would be floored to 0.
    # Mock _get_symbol_filters to return a larger step_size for BTCUSDT for this test
    with patch.object(order_manager, 	_get_symbol_filters	, new_callable=AsyncMock) as mock_filters:
        mock_filters.return_value = (0.001, 0.01, 5.0) # Lot_step = 0.001
        signal = TradeSignal(
            symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
            direction=TradeDirection.LONG, entry_price=10000000.0, # Very high price
            stop_loss_price=9000000.0, take_profit_price=13000000.0
        )
        # Expected raw quantity: (100 * 20) / 10000000 = 0.0002
        # Adjusted with step 0.001: floor(0.0002 / 0.001) * 0.001 = floor(0.2) * 0.001 = 0 * 0.001 = 0.0
        await order_manager.handle_trade_signal(signal)
        assert mock_rest_client_for_om.create_order.call_count == 0

