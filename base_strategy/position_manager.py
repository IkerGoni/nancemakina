import logging
from typing import Dict, Optional, List
from threading import RLock # For thread-safe access to positions dictionary

from src.models import Position, Order, OrderStatus, TradeDirection
from src.config_loader import ConfigManager # For potential config needs

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        # self.positions: Dict[str, Position] = {}  # Key: api_symbol (e.g., BTCUSDT)
        # A symbol can have only one position in Binance Futures (either LONG or SHORT, not both simultaneously unless in Hedge Mode)
        # For non-Hedge mode, a new trade in opposite direction first closes/reduces existing one.
        # Our bot logic (at least for V1) will likely manage one position per symbol at a time.
        self._positions: Dict[str, Position] = {}
        self._lock = RLock()

    async def add_or_update_position(self, position_data: Position):
        """Adds a new position or updates an existing one based on symbol."""
        with self._lock:
            api_symbol = position_data.symbol
            if api_symbol in self._positions:
                # This would typically mean an update, e.g., SL/TP moved, or PnL update
                # For MVP, a new signal might replace an old position after it is closed.
                # For now, let simple replacement happen if a new position is explicitly added.
                logger.info(f"Updating existing position for {api_symbol}.")
            else:
                logger.info(f"Adding new position for {api_symbol}.")
            self._positions[api_symbol] = position_data
            logger.debug(f"Position for {api_symbol} added/updated: {position_data}")

    async def remove_position(self, api_symbol: str):
        """Removes a position, typically after it has been closed."""
        with self._lock:
            if api_symbol in self._positions:
                logger.info(f"Removing position for {api_symbol}.")
                del self._positions[api_symbol]
            else:
                logger.warning(f"Attempted to remove non-existent position for {api_symbol}.")

    def get_position(self, api_symbol: str) -> Optional[Position]:
        """Retrieves the current position for a given symbol."""
        with self._lock:
            return self._positions.get(api_symbol)

    def get_all_positions(self) -> List[Position]:
        """Retrieves all currently tracked positions."""
        with self._lock:
            return list(self._positions.values())
    
    def has_open_position(self, api_symbol: str) -> bool:
        """Checks if there is an active position for the symbol."""
        with self._lock:
            return api_symbol in self._positions

    async def update_position_on_order_update(self, order: Order):
        """
        Updates position based on an order update (e.g., fill, cancellation).
        This is a crucial part that would typically be driven by user data stream.
        For MVP, this might be called by OrderManager after an order is confirmed filled/cancelled.
        """
        with self._lock:
            api_symbol = order.symbol
            position = self.get_position(api_symbol)

            if not position:
                # If an order update comes for a symbol with no tracked position, it might be an old order
                # or an order not initiated by this bot session. For now, we ignore.
                # logger.debug(f"Received order update for {api_symbol} but no active position tracked. Order ID: {order.order_id}")
                return

            # Was this order part of our tracked position?
            is_entry_order = order.order_id == position.entry_order_id
            is_sl_order = order.order_id == position.sl_order_id
            is_tp_order = order.order_id == position.tp_order_id

            if not (is_entry_order or is_sl_order or is_tp_order):
                # logger.debug(f"Order {order.order_id} for {api_symbol} not related to current tracked position.")
                return

            logger.info(f"Processing order update for position {api_symbol}. Order ID: {order.order_id}, Status: {order.status}")

            if order.status == OrderStatus.FILLED:
                if is_entry_order:
                    # Entry order filled. Position is now fully active.
                    # Update entry price if it was a market order and avgFillPrice is available.
                    if order.avg_fill_price and order.avg_fill_price > 0:
                        logger.info(f"Updating entry price for {api_symbol} from {position.entry_price} to {order.avg_fill_price} based on fill.")
                        position.entry_price = order.avg_fill_price
                    # Potentially update quantity if partial fill handling is added
                    position.quantity = order.filled_quantity or position.quantity
                    logger.info(f"Entry order {order.order_id} for {api_symbol} filled. Position active.")
                
                elif is_sl_order:
                    logger.info(f"Stop-loss order {order.order_id} for {api_symbol} FILLED. Position closed.")
                    # Position closed by SL. Remove it.
                    # Before removing, one might want to log PnL, etc.
                    await self.remove_position(api_symbol)
                    # TODO: Cancel the corresponding TP order if it is still open
                    if position.tp_order_id:
                        logger.info(f"Attempting to cancel TP order {position.tp_order_id} for closed position {api_symbol}")
                        # await self.rest_client.cancel_order(position.tp_order_id, api_symbol) # Requires rest_client here

                elif is_tp_order:
                    logger.info(f"Take-profit order {order.order_id} for {api_symbol} FILLED. Position closed.")
                    # Position closed by TP. Remove it.
                    await self.remove_position(api_symbol)
                    # TODO: Cancel the corresponding SL order if it is still open
                    if position.sl_order_id:
                        logger.info(f"Attempting to cancel SL order {position.sl_order_id} for closed position {api_symbol}")
                        # await self.rest_client.cancel_order(position.sl_order_id, api_symbol)
            
            elif order.status == OrderStatus.CANCELED:
                # If an SL or TP order is canceled, we need to handle it.
                # Maybe the bot decided to close the position manually, or an error occurred.
                if is_sl_order:
                    logger.warning(f"SL order {order.order_id} for {api_symbol} was CANCELED. Position might be unprotected.")
                    position.sl_order_id = None # Mark SL as no longer active
                elif is_tp_order:
                    logger.warning(f"TP order {order.order_id} for {api_symbol} was CANCELED.")
                    position.tp_order_id = None
                # If entry order is canceled, the position was never truly opened.
                elif is_entry_order:
                    logger.warning(f"Entry order {order.order_id} for {api_symbol} was CANCELED. Removing tentative position.")
                    await self.remove_position(api_symbol)
            
            elif order.status == OrderStatus.REJECTED:
                logger.error(f"Order {order.order_id} for {api_symbol} was REJECTED.")
                if is_entry_order:
                    await self.remove_position(api_symbol)
                elif is_sl_order:
                    position.sl_order_id = None
                elif is_tp_order:
                    position.tp_order_id = None
                # This state requires careful handling - why was it rejected?

            # Persist the updated position state (if not removed)
            if self.get_position(api_symbol):
                 self._positions[api_symbol] = position # Ensure the modified position object is stored back

    # Placeholder for fetching positions from exchange on startup (reconciliation)
    async def reconcile_positions_with_exchange(self, rest_client: any):
        """
        Fetches open positions from the exchange and updates internal state.
        Useful on startup to sync with any positions already open on Binance.
        `rest_client` is passed here as PositionManager might not always have it directly.
        """
        with self._lock:
            logger.info("Reconciling positions with exchange...")
            try:
                exchange_positions = await rest_client.fetch_positions() # Fetches all open positions
                if exchange_positions is None: # Error occurred during fetch
                    logger.error("Failed to fetch positions from exchange for reconciliation.")
                    return

                current_bot_symbols = set(self._positions.keys())
                exchange_symbols_with_pos = set()

                for pos_data in exchange_positions:
                    api_symbol = pos_data.get("symbol")
                    if not api_symbol: continue
                    exchange_symbols_with_pos.add(api_symbol)

                    # Basic conversion from ccxt position structure to our Position model
                    # This needs to be robust and handle various fields from ccxt
                    qty = float(pos_data.get("contracts", pos_data.get("amount", 0)))
                    if qty == 0: # No actual position, skip
                        if api_symbol in self._positions: # If bot thought it had a position, remove it
                            logger.info(f"Reconciliation: Exchange shows no position for {api_symbol}, removing from bot state.")
                            del self._positions[api_symbol]
                        continue
                    
                    side = TradeDirection.LONG if qty > 0 else TradeDirection.SHORT
                    entry_price = float(pos_data.get("entryPrice", 0))
                    
                    # Create or update bot's internal position state
                    # This is a simplified reconciliation. A full one would also check SL/TP orders.
                    reconciled_pos = Position(
                        symbol=api_symbol,
                        contract_type="USDT_M" if "USDT" in api_symbol else "COIN_M", # Basic assumption
                        side=side,
                        entry_price=entry_price,
                        quantity=abs(qty),
                        margin_type=pos_data.get("marginType"),
                        leverage=int(pos_data.get("leverage",0)),
                        unrealized_pnl=float(pos_data.get("unrealizedPnl",0)),
                        liquidation_price=float(pos_data.get("liquidationPrice",0)),
                        mark_price=float(pos_data.get("markPrice",0)),
                        # SL/TP order IDs are not directly available from fetch_positions, would need separate logic
                    )
                    self._positions[api_symbol] = reconciled_pos
                    logger.info(f"Reconciliation: Synced position for {api_symbol} from exchange: {side.value} {abs(qty)} @ {entry_price}")

                # Remove positions tracked by bot but not on exchange
                for sym_in_bot in current_bot_symbols:
                    if sym_in_bot not in exchange_symbols_with_pos:
                        logger.info(f"Reconciliation: Bot had position for {sym_in_bot}, but not found on exchange. Removing.")
                        del self._positions[sym_in_bot]
                
                logger.info("Position reconciliation completed.")
            except Exception as e:
                logger.error(f"Error during position reconciliation: {e}", exc_info=True)


# Example Usage (for testing purposes)
async def main_position_manager_test():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    class MockConfigManager:
        def get_config(self): return {}
        def register_callback(self, cb): pass

    cfg_manager = MockConfigManager()
    position_manager = PositionManager(config_manager=cfg_manager)

    # Test adding a position
    pos1_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG, entry_price=50000,
        quantity=0.001, entry_order_id="entry1", sl_order_id="sl1", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos1_data)
    print(f"Position for BTCUSDT: {position_manager.get_position('BTCUSDT')}")
    print(f"All positions: {position_manager.get_all_positions()}")

    # Test updating a position (e.g. SL moved, not fully implemented here but shows add_or_update)
    pos1_updated_data = Position(
        symbol="BTCUSDT", contract_type="USDT_M", side=TradeDirection.LONG, entry_price=50000,
        quantity=0.001, entry_order_id="entry1", sl_order_id="sl1_moved", tp_order_id="tp1"
    )
    await position_manager.add_or_update_position(pos1_updated_data)
    print(f"Updated Position for BTCUSDT: {position_manager.get_position('BTCUSDT')}")

    # Test order update - SL filled
    sl_fill_order = Order(
        order_id="sl1_moved", client_order_id="", symbol="BTCUSDT", type="STOP_MARKET", side="SELL",
        quantity=0.001, status=OrderStatus.FILLED, timestamp=1234567890, avg_fill_price=49000, filled_quantity=0.001
    )
    await position_manager.update_position_on_order_update(sl_fill_order)
    print(f"Position for BTCUSDT after SL fill: {position_manager.get_position('BTCUSDT')}") # Should be None
    assert position_manager.get_position('BTCUSDT') is None

    # Test adding another position
    pos2_data = Position(
        symbol="ETHUSDT", contract_type="USDT_M", side=TradeDirection.SHORT, entry_price=3000,
        quantity=0.05, entry_order_id="entry2", sl_order_id="sl2", tp_order_id="tp2"
    )
    await position_manager.add_or_update_position(pos2_data)
    print(f"All positions: {position_manager.get_all_positions()}")

    # Test removing a position directly
    await position_manager.remove_position("ETHUSDT")
    print(f"All positions after ETHUSDT removal: {position_manager.get_all_positions()}")
    assert position_manager.get_position('ETHUSDT') is None

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_position_manager_test())
