import pytest
import time
from pydantic import ValidationError

from src.models import Kline, PairConfig, TradeSignal, TradeDirection, Order, OrderStatus, OrderType, OrderSide, Position

def test_kline_creation():
    ts = int(time.time() * 1000)
    kline = Kline(
        timestamp=ts, open=100, high=110, low=90, close=105, volume=1000,
        is_closed=True, symbol="BTCUSDT", interval="1m"
    )
    assert kline.timestamp == ts
    assert kline.close == 105
    assert kline.symbol == "BTCUSDT"
    assert kline.is_closed is True

def test_pair_config_creation():
    pc = PairConfig(symbol="BTC_USDT", enabled=True, leverage=20, contract_type="USDT_M")
    assert pc.symbol == "BTC_USDT"
    assert pc.leverage == 20
    assert pc.contract_type == "USDT_M"

    with pytest.raises(ValidationError):
        PairConfig(symbol="BTCUSDT", enabled=True) # Invalid symbol format for config
    
    pc_coinm = PairConfig(symbol="BTCUSD_PERP", enabled=True, contract_type="COIN_M")
    assert pc_coinm.symbol == "BTCUSD_PERP"
    assert pc_coinm.contract_type == "COIN_M"

def test_trade_signal_creation():
    ts = int(time.time() * 1000)
    signal = TradeSignal(
        timestamp=ts, symbol="ETHUSDT", config_symbol="ETH_USDT", contract_type="USDT_M",
        direction=TradeDirection.SHORT, entry_price=2000, stop_loss_price=2100, take_profit_price=1800,
        strategy_name="TestStrategy"
    )
    assert signal.symbol == "ETHUSDT"
    assert signal.direction == TradeDirection.SHORT
    assert signal.take_profit_price == 1800

def test_order_creation():
    ts = int(time.time() * 1000)
    order = Order(
        order_id="ord123", symbol="LINKUSDT", type=OrderType.LIMIT, side=OrderSide.BUY,
        quantity=100, price=15.0, status=OrderStatus.NEW, timestamp=ts
    )
    assert order.order_id == "ord123"
    assert order.type == OrderType.LIMIT
    assert order.status == OrderStatus.NEW

    market_order = Order(
        order_id="ord124", symbol="ADAUSDT", type=OrderType.MARKET, side=OrderSide.SELL,
        quantity=1000, status=OrderStatus.FILLED, timestamp=ts, avg_fill_price=0.45, filled_quantity=1000
    )
    assert market_order.type == OrderType.MARKET
    assert market_order.avg_fill_price == 0.45

def test_position_creation():
    ts = int(time.time() * 1000)
    position = Position(
        symbol="SOLUSDT", contract_type="USDT_M", side=TradeDirection.LONG, entry_price=40.0,
        quantity=10, leverage=10, margin_type="ISOLATED", entry_timestamp=ts,
        entry_order_id="entry_sol_1", sl_order_id="sl_sol_1", tp_order_id="tp_sol_1",
        current_sl_price=38.0, current_tp_price=45.0
    )
    assert position.symbol == "SOLUSDT"
    assert position.side == TradeDirection.LONG
    assert position.current_sl_price == 38.0

# Test enums
def test_trade_direction_values():
    assert TradeDirection.LONG.value == "BUY"
    assert TradeDirection.SHORT.value == "SELL"

def test_order_status_values():
    assert OrderStatus.NEW.value == "NEW"
    assert OrderStatus.FILLED.value == "FILLED"
    # ... add more checks if needed

def test_order_type_values():
    assert OrderType.MARKET.value == "MARKET"
    assert OrderType.LIMIT.value == "LIMIT"
    # ... add more checks if needed

def test_order_side_values():
    assert OrderSide.BUY.value == "BUY"
    assert OrderSide.SELL.value == "SELL"

# Test validation if any complex validators were added
# Example: Kline timestamp must be positive
@pytest.mark.parametrize("invalid_ts", [-100, 0])
def test_kline_invalid_timestamp(invalid_ts):
    # Assuming Pydantic models implicitly validate types, but explicit validators can be tested.
    # If Kline had a validator for timestamp > 0:
    # with pytest.raises(ValidationError):
    #     Kline(timestamp=invalid_ts, open=1, high=2, low=0, close=1, volume=10, is_closed=True)
    pass # No explicit validator for this in current model, Pydantic handles type (int)

@pytest.mark.parametrize(
    "symbol_input, expected_output, should_raise",
    [
        ("BTC_USDT", "BTC_USDT", False),
        ("ETH_USD", "ETH_USD", False),
        ("BTCUSD_PERP", "BTCUSD_PERP", False),
        ("BTCUSDT", "BTCUSDT", True), # Fails because _ is missing for USDT_M style
        ("btcusd_perp", "btcusd_perp", False), # Passes as it ends with PERP
    ]
)
def test_pair_config_symbol_validation(symbol_input, expected_output, should_raise):
    if should_raise:
        with pytest.raises(ValidationError):
            PairConfig(symbol=symbol_input, enabled=True)
    else:
        pc = PairConfig(symbol=symbol_input, enabled=True)
        assert pc.symbol == expected_output

