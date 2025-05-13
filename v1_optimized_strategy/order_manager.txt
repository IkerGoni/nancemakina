import logging
import asyncio
import time
from typing import Optional, Dict, Any, Tuple
import math
from decimal import Decimal, ROUND_DOWN, getcontext

from src.config_loader import ConfigManager
from src.connectors import BinanceRESTClient, convert_symbol_to_api_format
from src.models import TradeSignal, Order, OrderStatus, OrderType, OrderSide, Position, TradeDirection
# from src.position_manager import PositionManager # Will be used later

logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, config_manager: ConfigManager, rest_client: BinanceRESTClient):
        self.config_manager = config_manager
        self.rest_client = rest_client
        # self.position_manager = position_manager # To notify about new positions/orders
        self.exchange_info_cache: Dict[str, Any] = {}

    async def _get_exchange_info_for_symbol(self, api_symbol: str) -> Optional[Dict[str, Any]]:
        if api_symbol in self.exchange_info_cache:
            return self.exchange_info_cache[api_symbol]
        
        markets = await self.rest_client.fetch_exchange_info()
        if markets:
            for market_symbol, market_data in markets.items():
                # ccxt market symbols are usually already in API format (e.g., BTCUSDT)
                # We store all of them for potential future use
                self.exchange_info_cache[market_symbol] = market_data 
            
            if api_symbol in self.exchange_info_cache:
                return self.exchange_info_cache[api_symbol]
            else:
                # Try variations if direct match fails (e.g. if input was BTC/USDT vs BTCUSDT)
                # This part might need refinement based on how symbols are consistently handled.
                normalized_api_symbol = api_symbol.replace("/","")
                if normalized_api_symbol in self.exchange_info_cache:
                    return self.exchange_info_cache[normalized_api_symbol]
                logger.warning(f"Exchange info not found for symbol {api_symbol} after fetching markets.")
                return None
        logger.error(f"Failed to fetch exchange info.")
        return None

    def _adjust_quantity_to_precision(self, quantity: float, step_size: float) -> float:
        """Adjusts quantity to the nearest multiple of step_size (floors the value)."""
        if step_size == 0: return quantity # Avoid division by zero if step_size is not set
        
        # Special case for very small step sizes like 0.00001
        if step_size < 0.0001:
            # Convert to string representation for exactness
            step_str = str(step_size)
            precision = len(step_str.split(".")[1]) if "." in step_str else 0
            
            # For very precise step sizes, use decimal module 
            from decimal import Decimal, ROUND_DOWN
            dec_quantity = Decimal(str(quantity))
            dec_step = Decimal(str(step_size))
            
            # Calculate floor division
            steps = dec_quantity / dec_step
            floored_steps = steps.to_integral_exact(rounding=ROUND_DOWN)
            result = floored_steps * dec_step
            
            return float(result)
        
        # For normal step sizes, use regular float math
        return math.floor(quantity / step_size) * step_size

    def _adjust_price_to_precision(self, price: float, tick_size: float) -> float:
        if tick_size == 0: return price
        precision = 0
        if "." in str(tick_size):
            precision = len(str(tick_size).split(".")[1].rstrip("0"))
        else:
            precision = 0
        # For price, usually round to nearest tick_size. Behavior might differ (round, ceil, floor)
        # Let's round for now.
        adjusted_price = round(round(price / tick_size) * tick_size, precision) if precision > 0 else int(round(price / tick_size) * tick_size)
        return adjusted_price

    async def _get_symbol_filters(self, api_symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """ Returns (lot_step_size, price_tick_size, min_notional) """
        market_info = await self._get_exchange_info_for_symbol(api_symbol)
        if not market_info:
            logger.warning(f"Cannot get filters for {api_symbol}, market info not found.")
            return None, None, None

        lot_step_size = None
        price_tick_size = None
        min_notional = None

        try:
            # CCXT structure for precision and limits:
            # market_info["precision"]["amount"] -> quantity precision (step size)
            # market_info["precision"]["price"] -> price precision (tick size)
            # market_info["limits"]["cost"]["min"] -> min notional value
            # market_info["limits"]["amount"]["min"] -> min quantity

            if market_info.get("precision") and market_info["precision"].get("amount") is not None:
                # Handle both string and float formats
                amount_value = market_info["precision"]["amount"]
                lot_step_size = float(amount_value)
            
            if market_info.get("precision") and market_info["precision"].get("price") is not None:
                # Handle both string and float formats
                price_value = market_info["precision"]["price"]
                price_tick_size = float(price_value)

            if market_info.get("limits", {}).get("cost", {}).get("min") is not None:
                # Handle both string and float formats
                min_value = market_info["limits"]["cost"]["min"]
                min_notional = float(min_value)
            elif market_info.get("info", {}).get("filters"):
                # Fallback for direct Binance API structure if ccxt doesn't normalize it fully
                for f in market_info["info"]["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        lot_step_size = float(f["stepSize"])
                    elif f["filterType"] == "PRICE_FILTER":
                        price_tick_size = float(f["tickSize"])
                    elif f["filterType"] == "MIN_NOTIONAL":
                        min_notional = float(f.get("notional", f.get("minNotional"))) # Binance uses "notional" or "minNotional"
            
            # Log warnings if any filters are missing
            if lot_step_size is None: logger.warning(f"Lot step size not found for {api_symbol}")
            if price_tick_size is None: logger.warning(f"Price tick size not found for {api_symbol}")
            if min_notional is None: logger.warning(f"Min notional not found for {api_symbol}")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error accessing filter info for {api_symbol}: {e}. Market info: {market_info}")
            return None, None, None
        except Exception as e:
            logger.error(f"Error parsing filters for {api_symbol}: {e}. Market info: {market_info}")
            return None, None, None
            
        return lot_step_size, price_tick_size, min_notional

    async def handle_trade_signal(self, signal: TradeSignal):
        logger.info(f"OrderManager received signal: {signal.symbol} {signal.direction.value} at {signal.entry_price}")
        config = self.config_manager.get_config()
        pair_configs = config.get("pairs", {})
        global_v1_config = config.get("global_settings", {}).get("v1_strategy", {})
        api_symbol = signal.symbol # Already in API format from signal

        pair_details = {}
        for cfg_sym, details in pair_configs.items():
            if convert_symbol_to_api_format(cfg_sym) == api_symbol:
                pair_details = details
                break
        
        if not pair_details:
            logger.error(f"No configuration found for symbol {api_symbol} in OrderManager.")
            return

        # 1. Set Margin Type and Leverage
        margin_mode = pair_details.get("margin_mode", global_v1_config.get("margin_mode", "ISOLATED"))
        leverage = pair_details.get("leverage", global_v1_config.get("default_leverage", 10))

        # These calls might not be needed if already set and not changing per trade.
        # However, good practice to ensure they are as expected.
        # await self.rest_client.set_margin_mode(api_symbol, margin_mode)
        # await self.rest_client.set_leverage(api_symbol, leverage)
        # For now, assume they are pre-set or handle this in a setup phase.
        # logger.info(f"Ensured margin mode {margin_mode} and leverage {leverage}x for {api_symbol}")

        # 2. Calculate Position Size (MVP: Fixed Margin Allocation)
        # TODO: Implement dynamic position sizing later
        fixed_margin_usdt = pair_details.get("margin_usdt", global_v1_config.get("default_margin_usdt"))
        fixed_margin_coin = pair_details.get("margin_coin") # For COIN-M
        entry_price = signal.entry_price
        quantity: Optional[float] = None

        lot_step, price_tick, min_notional_val = await self._get_symbol_filters(api_symbol)
        if lot_step is None or price_tick is None:
            logger.error(f"Could not get precision filters for {api_symbol}. Aborting trade.")
            return

        if signal.contract_type == "USDT_M":
            if fixed_margin_usdt and entry_price > 0:
                quantity_raw = (fixed_margin_usdt * leverage) / entry_price
                quantity = self._adjust_quantity_to_precision(quantity_raw, lot_step)
            else:
                logger.error(f"Invalid fixed_margin_usdt or entry_price for {api_symbol} (USDT-M).")
                return
        elif signal.contract_type == "COIN_M":
            # For COIN-M, quantity is in number of contracts. Value of 1 contract is e.g. 10 USD or 100 USD.
            # Margin is in base currency (e.g. BTC for BTCUSD_PERP).
            # Size = (MarginInBase * Leverage * EntryPrice) / ContractValueInQuote (if contract value is fixed in quote)
            # Or more simply: SizeInContracts = (MarginInBase * Leverage) / (MarginRequiredPerContractInBase)
            # Binance API for COIN-M: quantity is number of contracts.
            # Let's assume `fixed_margin_coin` is the amount of base currency to risk.
            # The actual USD value of this margin fluctuates.
            # Position size in contracts: (Margin_in_Coin * Leverage) / (InitialMarginPerContract_in_Coin)
            # InitialMarginPerContract_in_Coin = (ContractValue_in_USD / EntryPrice_USD) / Leverage (this is circular)
            # Simpler: Quantity_in_Contracts = (Margin_In_Coin * Leverage * EntryPrice_QuotePerBase) / Contract_FaceValue_In_Quote
            # This needs contract size info (e.g. 1 contract = 100 USD for BTCUSD_PERP)
            # For now, let's assume a simpler model if `fixed_margin_coin` is defined as number of contracts to trade directly.
            # This part needs careful review of COIN-M contract specs.
            # A common approach: if contract value is $100, margin is 0.01 BTC, leverage 10x, entry $50k
            # Total exposure in BTC = 0.01 * 10 = 0.1 BTC. Total exposure in USD = 0.1 * 50000 = $5000
            # Number of contracts = $5000 / $100_per_contract = 50 contracts.
            # So, quantity = (fixed_margin_coin * leverage * entry_price) / contract_value_quote
            # This requires `contract_value_quote` from exchange info or config.
            # For MVP, let's assume `fixed_margin_coin` IS the number of contracts if provided, else error.
            # This is a simplification and likely incorrect for general COIN-M. 
            # A better MVP for COIN-M might be to require quantity directly in config for COIN-M pairs.
            if fixed_margin_coin is not None: # Assuming fixed_margin_coin is intended as # of contracts for MVP
                quantity_raw = fixed_margin_coin # This is a placeholder assumption
                logger.warning(f"COIN-M quantity calculation using 'margin_coin' as number of contracts. Review for correctness.")
                quantity = self._adjust_quantity_to_precision(quantity_raw, lot_step)
            else:
                logger.error(f"COIN-M trading requires 'margin_coin' (interpreted as num_contracts for MVP) or a more robust sizing logic. {api_symbol}")
                return
        else:
            logger.error(f"Unknown contract type: {signal.contract_type} for {api_symbol}")
            return

        if quantity is None or quantity <= 0:
            logger.error(f"Calculated quantity is zero or invalid for {api_symbol}: {quantity}")
            return
        
        # Check min notional if applicable
        if min_notional_val and entry_price > 0:
            notional_value = quantity * entry_price
            # For COIN-M, notional value is trickier as it depends on contract multiplier
            # if signal.contract_type == "COIN_M": # Need contract multiplier
            #    pass 
            if signal.contract_type == "USDT_M" and notional_value < min_notional_val:
                logger.warning(f"Order for {api_symbol} notional {notional_value} is less than min_notional {min_notional_val}. Skipping.")
                return

        # 3. Place Entry Order (MARKET)
        order_side = OrderSide.BUY if signal.direction == TradeDirection.LONG else OrderSide.SELL
        entry_order_params = {"newOrderRespType": "RESULT"} # To get full order details
        logger.info(f"Placing entry order for {api_symbol}: {order_side.value} {quantity} units at MARKET price.")
        entry_order_response = await self.rest_client.create_order(
            symbol=api_symbol, 
            order_type=OrderType.MARKET, 
            side=order_side, 
            amount=quantity, 
            params=entry_order_params
        )

        if not entry_order_response or "id" not in entry_order_response or entry_order_response.get("status") == OrderStatus.REJECTED:
            logger.error(f"Failed to place entry order for {api_symbol}. Response: {entry_order_response}")
            return
        
        entry_order_id = entry_order_response["id"]
        logger.info(f"Entry order placed for {api_symbol}. ID: {entry_order_id}, Status: {entry_order_response.get('status')}")

        # Wait for entry order to fill (or partial fill if handling that)
        # For MARKET orders, they usually fill quickly. Polling might be needed for LIMIT.
        # For simplicity, assume MARKET order fills almost instantly for MVP.
        # A robust system would poll order status or use user data stream.
        # Let's assume we get avgFillPrice from the response if available, or use signal.entry_price
        actual_entry_price = float(entry_order_response.get("avgPrice", signal.entry_price)) 
        if actual_entry_price == 0 and entry_order_response.get("status") == OrderStatus.FILLED:
             actual_entry_price = float(entry_order_response.get("fills",[{}])[0].get("price", signal.entry_price))
        if actual_entry_price == 0 : actual_entry_price = signal.entry_price # Fallback

        # 4. Place SL and TP Orders (STOP_MARKET and TAKE_PROFIT_MARKET with reduceOnly)
        sl_tp_side = OrderSide.SELL if signal.direction == TradeDirection.LONG else OrderSide.BUY
        
        # Adjust SL/TP prices to precision
        adjusted_sl_price = self._adjust_price_to_precision(signal.stop_loss_price, price_tick)
        adjusted_tp_price = self._adjust_price_to_precision(signal.take_profit_price, price_tick)

        sl_order_params = {"reduceOnly": "true"}
        logger.info(f"Placing SL order for {api_symbol}: {sl_tp_side.value} {quantity} units, stopPrice {adjusted_sl_price:.{self._get_price_precision(price_tick)}f}")
        sl_order_response = await self.rest_client.create_order(
            symbol=api_symbol, 
            order_type=OrderType.STOP_MARKET, 
            side=sl_tp_side, 
            amount=quantity, 
            params={**sl_order_params, "stopPrice": adjusted_sl_price}
        )

        if not sl_order_response or "id" not in sl_order_response:
            logger.error(f"Failed to place SL order for {api_symbol}. Response: {sl_order_response}. Entry order {entry_order_id} might need manual SL.")
            # Potentially try to close the position if SL cannot be placed
        else:
            logger.info(f"SL order placed for {api_symbol}. ID: {sl_order_response['id']}")

        tp_order_params = {"reduceOnly": "true"}
        logger.info(f"Placing TP order for {api_symbol}: {sl_tp_side.value} {quantity} units, stopPrice {adjusted_tp_price:.{self._get_price_precision(price_tick)}f}")
        tp_order_response = await self.rest_client.create_order(
            symbol=api_symbol, 
            order_type=OrderType.TAKE_PROFIT_MARKET, 
            side=sl_tp_side, 
            amount=quantity, 
            params={**tp_order_params, "stopPrice": adjusted_tp_price}
        )

        if not tp_order_response or "id" not in tp_order_response:
            logger.error(f"Failed to place TP order for {api_symbol}. Response: {tp_order_response}. Entry order {entry_order_id} might need manual TP.")
        else:
            logger.info(f"TP order placed for {api_symbol}. ID: {tp_order_response['id']}")

        # TODO: Notify PositionManager about the new position and associated orders
        # position_data = Position(
        #     symbol=api_symbol,
        #     contract_type=signal.contract_type,
        #     side=signal.direction,
        #     entry_price=actual_entry_price,
        #     quantity=quantity,
        #     leverage=leverage,
        #     margin_type=margin_mode,
        #     entry_order_id=entry_order_id,
        #     sl_order_id=sl_order_response.get("id") if sl_order_response else None,
        #     tp_order_id=tp_order_response.get("id") if tp_order_response else None,
        #     current_sl_price=adjusted_sl_price,
        #     current_tp_price=adjusted_tp_price
        # )
        # await self.position_manager.add_or_update_position(position_data)
        logger.info(f"Trade execution process completed for signal on {api_symbol}.")

    def _get_price_precision(self, tick_size: float) -> int:
        if tick_size == 0: return 8 # Default high precision if tick_size is 0
        if "." in str(tick_size):
            return len(str(tick_size).split(".")[1].rstrip("0"))
        return 0

# Example Usage (for integration testing, not standalone run)
async def main_order_manager_test():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    class MockConfigManager:
        def get_config(self):
            return {
                "api": {"binance_api_key": "SIM_KEY", "binance_api_secret": "SIM_SECRET"}, # Simulate real keys for REST client init
                "global_settings": {
                    "v1_strategy": {
                        "default_margin_usdt": 50,
                        "default_leverage": 10,
                        "margin_mode": "ISOLATED"
                    }
                },
                "pairs": {
                    "BTC_USDT": {"enabled": True, "contract_type": "USDT_M", "margin_usdt": 100, "leverage": 20},
                    "ETHUSD_PERP": {"enabled": True, "contract_type": "COIN_M", "margin_coin": 0.1, "leverage": 10} # margin_coin as num_contracts
                }
            }
        def register_callback(self, cb): pass
        def stop_watcher(self): pass

    class MockBinanceRESTClient:
        def __init__(self, cfg):
            logger.info("MockBinanceRESTClient initialized")
            self.exchange_info_data = {
                "BTCUSDT": {"precision": {"amount": "0.00001", "price": "0.01"}, "limits": {"cost": {"min": "10"}}},
                "ETHUSD_PERP": {"precision": {"amount": "1", "price": "0.01"}, "limits": {"cost": {"min": "5"}}} # COIN-M contracts are usually integers
            }
        async def fetch_exchange_info(self): return self.exchange_info_data
        async def create_order(self, symbol, order_type, side, amount, price=None, params={}):
            order_id = f"mock_{order_type.lower()}_{int(time.time()*1000)}"
            logger.info(f"MOCK Create Order: {symbol}, {order_type}, {side}, {amount}, Price: {price}, Params: {params}, ID: {order_id}")
            avg_price = price or (params.get("stopPrice") if order_type != OrderType.MARKET else 50000.0) # Simulate some fill price
            return {"id": order_id, "symbol": symbol, "status": "NEW", "avgPrice": str(avg_price), "type": order_type}
        async def set_leverage(self, symbol, leverage, params={}): logger.info(f"MOCK Set Leverage: {symbol} to {leverage}x"); return True
        async def set_margin_mode(self, symbol, mode, params={}): logger.info(f"MOCK Set Margin Mode: {symbol} to {mode}"); return True
        async def close_exchange(self): pass

    cfg = MockConfigManager()
    rest_client = MockBinanceRESTClient(cfg) # Use mock for testing
    order_manager = OrderManager(cfg, rest_client)

    # Test USDT-M signal
    usdt_signal = TradeSignal(
        symbol="BTCUSDT", config_symbol="BTC_USDT", contract_type="USDT_M",
        direction=TradeDirection.LONG, entry_price=50000.0,
        stop_loss_price=49000.0, take_profit_price=53000.0
    )
    logger.info("--- Testing USDT-M Signal ---")
    await order_manager.handle_trade_signal(usdt_signal)

    await asyncio.sleep(1) # Ensure logs are flushed

    # Test COIN-M signal
    coinm_signal = TradeSignal(
        symbol="ETHUSD_PERP", config_symbol="ETHUSD_PERP", contract_type="COIN_M",
        direction=TradeDirection.SHORT, entry_price=3000.0,
        stop_loss_price=3100.0, take_profit_price=2700.0
    )
    logger.info("--- Testing COIN-M Signal ---")
    await order_manager.handle_trade_signal(coinm_signal)
    
    await rest_client.close_exchange()

if __name__ == "__main__":
    asyncio.run(main_order_manager_test())
