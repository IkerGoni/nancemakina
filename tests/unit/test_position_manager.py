import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from threading import RLock

from src.config_loader import ConfigManager
from src.position_manager import PositionManager
from src.models import Position, Order, OrderStatus, TradeDirection, OrderType, OrderSide

@pytest.fixture
def mock_config_manager_for_pm():
    cm = MagicMock(spec=ConfigManager)
    cm.get_config.return_value = {}  # Empty config is fine for PositionManager tests
    return cm

@pytest.fixture
def position_manager(mock_config_manager_for_pm):
    return PositionManager(config_manager=mock_config_manager_for_pm)

@pytest.mark.asyncio
async def test_pm_empty_state(position_manager):
    """Test that a new PositionManager returns correct values when empty."""
    # Verify getters return correct values when empty
    assert position_manager.get_position("BTCUSDT") is None
    assert position_manager.get_all_positions() == []
    assert position_manager.has_open_position("BTCUSDT") is False
    assert position_manager.has_open_position("ETHUSDT") is False

@pytest.mark.asyncio
async def test_pm_add_get_position(position_manager):
    # Test adding a position
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1",
        sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Test retrieving the position
    retrieved_pos = position_manager.get_position("BTCUSDT")
    assert retrieved_pos is not None
    assert retrieved_pos.symbol == "BTCUSDT"
    assert retrieved_pos.entry_price == 50000.0
    assert retrieved_pos.quantity == 0.01
    assert retrieved_pos.sl_order_id == "sl1"
    
    # Test has_open_position
    assert position_manager.has_open_position("BTCUSDT") is True
    assert position_manager.has_open_position("ETHUSDT") is False
    
    # Test get_all_positions
    all_positions = position_manager.get_all_positions()
    assert len(all_positions) == 1
    assert all_positions[0].symbol == "BTCUSDT"

@pytest.mark.asyncio
async def test_pm_update_position(position_manager):
    # Add initial position
    pos_data = Position(
        symbol="ETHUSDT", contract_type="USDT_M", side=TradeDirection.SHORT,
        entry_price=3000.0, quantity=0.1, entry_order_id="entry2",
        sl_order_id="sl2", tp_order_id="tp2"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Update the position
    updated_pos_data = Position(
        symbol="ETHUSDT", contract_type="USDT_M", side=TradeDirection.SHORT,
        entry_price=3000.0, quantity=0.1, entry_order_id="entry2",
        sl_order_id="sl2_new", tp_order_id="tp2", # Changed SL order ID
        current_sl_price=3100.0, # Added SL price
        unrealized_pnl=-50.0 # Added PnL
    )
    await position_manager.add_or_update_position(updated_pos_data)
    
    # Verify the update
    retrieved_pos = position_manager.get_position("ETHUSDT")
    assert retrieved_pos.sl_order_id == "sl2_new"
    assert retrieved_pos.current_sl_price == 3100.0
    assert retrieved_pos.unrealized_pnl == -50.0

@pytest.mark.asyncio
async def test_pm_remove_position(position_manager):
    # Add a position
    pos_data = Position(
        symbol="SOLUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=100.0, quantity=1.0, entry_order_id="entry3",
        sl_order_id="sl3", tp_order_id="tp3"
    )
    await position_manager.add_or_update_position(pos_data)
    assert position_manager.has_open_position("SOLUSDT") is True
    
    # Remove the position
    await position_manager.remove_position("SOLUSDT")
    assert position_manager.has_open_position("SOLUSDT") is False
    assert position_manager.get_position("SOLUSDT") is None
    
    # Test removing a non-existent position (should not raise error)
    await position_manager.remove_position("NONEXISTENT")

@pytest.mark.asyncio
async def test_pm_update_position_on_entry_order_fill(position_manager):
    # Add a position
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1",
        sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Simulate entry order fill with different fill price
    entry_order_fill = Order(
        order_id="entry1", symbol="BTCUSDT", type=OrderType.MARKET, side=OrderSide.BUY,
        quantity=0.01, status=OrderStatus.FILLED, timestamp=1234567890,
        avg_fill_price=50100.0, filled_quantity=0.01
    )
    await position_manager.update_position_on_order_update(entry_order_fill)
    
    # Verify entry price was updated
    updated_pos = position_manager.get_position("BTCUSDT")
    assert updated_pos.entry_price == 50100.0

@pytest.mark.asyncio
async def test_pm_update_position_on_sl_order_fill(position_manager):
    # Add a position
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1",
        sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Simulate SL order fill
    sl_order_fill = Order(
        order_id="sl1", symbol="BTCUSDT", type=OrderType.STOP_MARKET, side=OrderSide.SELL,
        quantity=0.01, status=OrderStatus.FILLED, timestamp=1234567890,
        avg_fill_price=49000.0, filled_quantity=0.01
    )
    await position_manager.update_position_on_order_update(sl_order_fill)
    
    # Verify position was removed
    assert position_manager.has_open_position("BTCUSDT") is False

@pytest.mark.asyncio
async def test_pm_update_position_on_tp_order_fill(position_manager):
    # Add a position
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1",
        sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Simulate TP order fill
    tp_order_fill = Order(
        order_id="tp1", symbol="BTCUSDT", type=OrderType.TAKE_PROFIT_MARKET, side=OrderSide.SELL,
        quantity=0.01, status=OrderStatus.FILLED, timestamp=1234567890,
        avg_fill_price=52000.0, filled_quantity=0.01
    )
    await position_manager.update_position_on_order_update(tp_order_fill)
    
    # Verify position was removed
    assert position_manager.has_open_position("BTCUSDT") is False

@pytest.mark.asyncio
async def test_pm_update_position_on_order_cancel(position_manager):
    # Add a position
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1",
        sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Simulate SL order cancellation
    sl_order_cancel = Order(
        order_id="sl1", symbol="BTCUSDT", type=OrderType.STOP_MARKET, side=OrderSide.SELL,
        quantity=0.01, status=OrderStatus.CANCELED, timestamp=1234567890
    )
    await position_manager.update_position_on_order_update(sl_order_cancel)
    
    # Verify SL order ID was cleared
    updated_pos = position_manager.get_position("BTCUSDT")
    assert updated_pos.sl_order_id is None
    assert updated_pos.tp_order_id == "tp1"  # TP order should still be there

@pytest.mark.asyncio
async def test_pm_update_position_on_entry_order_cancel(position_manager):
    # Add a position
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1",
        sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Simulate entry order cancellation
    entry_order_cancel = Order(
        order_id="entry1", symbol="BTCUSDT", type=OrderType.MARKET, side=OrderSide.BUY,
        quantity=0.01, status=OrderStatus.CANCELED, timestamp=1234567890
    )
    await position_manager.update_position_on_order_update(entry_order_cancel)
    
    # Verify position was removed
    assert position_manager.has_open_position("BTCUSDT") is False

@pytest.mark.asyncio
async def test_pm_update_position_on_order_reject(position_manager):
    # Add a position
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1",
        sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Simulate TP order rejection
    tp_order_reject = Order(
        order_id="tp1", symbol="BTCUSDT", type=OrderType.TAKE_PROFIT_MARKET, side=OrderSide.SELL,
        quantity=0.01, status=OrderStatus.REJECTED, timestamp=1234567890
    )
    await position_manager.update_position_on_order_update(tp_order_reject)
    
    # Verify TP order ID was cleared
    updated_pos = position_manager.get_position("BTCUSDT")
    assert updated_pos.tp_order_id is None
    assert updated_pos.sl_order_id == "sl1"  # SL order should still be there

@pytest.mark.asyncio
async def test_pm_update_position_on_unrelated_order(position_manager):
    """Test that unrelated order updates don't affect positions."""
    # Add a position
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1",
        sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Create a snapshot of the position before the update
    original_pos = position_manager.get_position("BTCUSDT")
    
    # Simulate an order update with an unrelated order ID
    unrelated_order = Order(
        order_id="unrelated123", symbol="BTCUSDT", type=OrderType.MARKET, side=OrderSide.BUY,
        quantity=0.01, status=OrderStatus.FILLED, timestamp=1234567890
    )
    await position_manager.update_position_on_order_update(unrelated_order)
    
    # Verify position state remains unchanged
    updated_pos = position_manager.get_position("BTCUSDT")
    assert updated_pos.entry_order_id == original_pos.entry_order_id
    assert updated_pos.sl_order_id == original_pos.sl_order_id
    assert updated_pos.tp_order_id == original_pos.tp_order_id
    assert updated_pos.entry_price == original_pos.entry_price

@pytest.mark.asyncio
async def test_pm_update_position_on_order_for_nonexistent_position(position_manager):
    """Test handling order updates for symbols with no tracked position."""
    # Ensure no position exists
    assert position_manager.has_open_position("XYZUSDT") is False
    
    # Simulate an order update for a symbol with no position
    order = Order(
        order_id="xyz123", symbol="XYZUSDT", type=OrderType.MARKET, side=OrderSide.BUY,
        quantity=0.01, status=OrderStatus.FILLED, timestamp=1234567890
    )
    # This should not raise an error
    await position_manager.update_position_on_order_update(order)
    
    # Verify no position was created
    assert position_manager.has_open_position("XYZUSDT") is False

@pytest.mark.asyncio
async def test_pm_reconcile_positions_with_exchange(position_manager):
    # Mock REST client
    mock_rest_client = AsyncMock()
    mock_rest_client.fetch_positions.return_value = [
        {
            "symbol": "BTCUSDT",
            "contracts": 0.01,
            "entryPrice": 50000.0,
            "marginType": "ISOLATED",
            "leverage": 20,
            "unrealizedPnl": 100.0,
            "liquidationPrice": 45000.0,
            "markPrice": 51000.0
        },
        {
            "symbol": "ETHUSDT",
            "contracts": -0.1,  # Negative for SHORT
            "entryPrice": 3000.0,
            "marginType": "ISOLATED",
            "leverage": 10,
            "unrealizedPnl": -50.0,
            "liquidationPrice": 3300.0,
            "markPrice": 3050.0
        }
    ]
    
    # Add a position that doesn't exist on exchange
    pos_data = Position(
        symbol="SOLUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=100.0, quantity=1.0, entry_order_id="entry3",
        sl_order_id="sl3", tp_order_id="tp3"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Reconcile
    await position_manager.reconcile_positions_with_exchange(mock_rest_client)
    
    # Verify positions from exchange were added
    btc_pos = position_manager.get_position("BTCUSDT")
    assert btc_pos is not None
    assert btc_pos.side == TradeDirection.LONG
    assert btc_pos.entry_price == 50000.0
    assert btc_pos.quantity == 0.01
    assert btc_pos.unrealized_pnl == 100.0
    
    eth_pos = position_manager.get_position("ETHUSDT")
    assert eth_pos is not None
    assert eth_pos.side == TradeDirection.SHORT
    assert eth_pos.entry_price == 3000.0
    assert eth_pos.quantity == 0.1
    assert eth_pos.unrealized_pnl == -50.0
    
    # Verify position not on exchange was removed
    assert position_manager.has_open_position("SOLUSDT") is False

@pytest.mark.asyncio
async def test_pm_reconcile_positions_with_zero_quantity(position_manager):
    """Test reconciling with an exchange position that has zero quantity."""
    # Mock REST client with a zero-quantity position
    mock_rest_client = AsyncMock()
    mock_rest_client.fetch_positions.return_value = [
        {
            "symbol": "BTCUSDT",
            "contracts": 0,  # Zero quantity position
            "entryPrice": 50000.0,
            "marginType": "ISOLATED",
            "leverage": 20
        }
    ]
    
    # Add a position for BTCUSDT in manager
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1"
    )
    await position_manager.add_or_update_position(pos_data)
    assert position_manager.has_open_position("BTCUSDT") is True
    
    # Reconcile
    await position_manager.reconcile_positions_with_exchange(mock_rest_client)
    
    # Verify position was removed (because exchange shows zero quantity)
    assert position_manager.has_open_position("BTCUSDT") is False

@pytest.mark.asyncio
async def test_pm_reconcile_positions_with_exchange_error(position_manager):
    """Test reconciliation when fetch_positions returns None or raises an exception."""
    # Add a position that should remain untouched if fetch_positions fails
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1"
    )
    await position_manager.add_or_update_position(pos_data)
    
    # Case 1: fetch_positions returns None
    mock_rest_client1 = AsyncMock()
    mock_rest_client1.fetch_positions.return_value = None
    
    # Reconcile - should not modify positions
    await position_manager.reconcile_positions_with_exchange(mock_rest_client1)
    assert position_manager.has_open_position("BTCUSDT") is True
    
    # Case 2: fetch_positions raises an exception
    mock_rest_client2 = AsyncMock()
    mock_rest_client2.fetch_positions.side_effect = Exception("API Error")
    
    # Reconcile - should not modify positions
    await position_manager.reconcile_positions_with_exchange(mock_rest_client2)
    assert position_manager.has_open_position("BTCUSDT") is True

@pytest.mark.asyncio
async def test_pm_thread_safety(position_manager):
    # This test is more conceptual than practical in a unit test environment
    # In a real multi-threaded environment, we'd need to test with actual threads
    
    # Verify the lock exists and is a RLock
    assert hasattr(position_manager, "_lock")
    assert isinstance(position_manager._lock, RLock)
    
    # Test that methods use the lock (we can't directly test this without mocking the lock)
    # But we can verify the methods work correctly
    pos_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG,
        entry_price=50000.0, quantity=0.01, entry_order_id="entry1",
        sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos_data)
    assert position_manager.get_position("BTCUSDT") is not None
    
    # Simulate concurrent operations
    await asyncio.gather(
        position_manager.add_or_update_position(pos_data),
        position_manager.get_position("BTCUSDT"),
        position_manager.get_all_positions()
    )
    # If no exceptions, the lock is working as expected
