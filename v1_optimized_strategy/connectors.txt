import ccxt.async_support as ccxt
import asyncio
import websockets
import json
import logging
import time
import random
from typing import Callable, List, Dict, Any, Optional, Literal

from src.config_loader import ConfigManager
from src.models import Kline # Assuming Kline model is in src.models

logger = logging.getLogger(__name__)

# Symbol conversion utility (can be moved to utils.py later)
def convert_symbol_to_ws_format(config_symbol: str) -> str:
    """Converts BTC_USDT to btcusdt, BTCUSD_PERP to btcusd_perp"""
    if "_" in config_symbol:
        if config_symbol.endswith("_PERP"):
            return config_symbol.replace("_PERP", "_perp").lower()
        return config_symbol.replace("_", "").lower()
    return config_symbol.lower() # Fallback for already formatted or simple symbols

def convert_symbol_to_api_format(config_symbol: str) -> str:
    """Converts BTC_USDT to BTCUSDT, BTCUSD_PERP to BTCUSD_PERP"""
    if "_" in config_symbol:
        if config_symbol.endswith("_PERP"):
            return config_symbol.upper()
        return config_symbol.replace("_", "").upper()
    return config_symbol.upper()

class BinanceRESTClient:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.exchange = None
        self._initialize_exchange()

    def _initialize_exchange(self):
        config = self.config_manager.get_config()
        api_key = config.get("api", {}).get("binance_api_key")
        api_secret = config.get("api", {}).get("binance_api_secret")
        is_testnet = config.get("api", {}).get("testnet", False)

        if not api_key or not api_secret or "YOUR_" in api_key: # Basic check for placeholder
            logger.warning("API key/secret not found or placeholders used. REST client will operate in simulated mode (no real calls).")
            self.exchange = None 
            return

        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {
                "defaultType": "future",
            },
        })
        if is_testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("Binance REST Client initialized in Testnet mode.")
        else:
            logger.info("Binance REST Client initialized in Mainnet mode.")

    async def close_exchange(self):
        if self.exchange:
            await self.exchange.close()
            logger.info("CCXT exchange instance closed.")

    async def fetch_exchange_info(self, params={}):
        if not self.exchange: return None
        try:
            markets = await self.exchange.load_markets()
            return markets 
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return None

    async def fetch_balance(self, params={}):
        if not self.exchange: return None # Or simulate a balance for testing
        try:
            balance = await self.exchange.fetch_balance(params)
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None, params={}):
        if not self.exchange: 
            logger.info(f"Simulated order: {side} {amount} {symbol} at {price or 'market'}")
            return {"id": f"simulated_order_{int(time.time())}", "symbol": symbol, "status": "NEW", "type": order_type, "side": side, "amount": amount, "price": price}
        try:
            api_symbol = convert_symbol_to_api_format(symbol)
            order = await self.exchange.create_order(api_symbol, order_type, side, amount, price, params)
            logger.info(f"Order created: {order}")
            return order
        except Exception as e:
            logger.error(f"Error creating order for {symbol}: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str, params={}):
        if not self.exchange: 
            logger.info(f"Simulated cancel order: {order_id} for {symbol}")
            return {"id": order_id, "status": "CANCELED"}
        try:
            api_symbol = convert_symbol_to_api_format(symbol)
            response = await self.exchange.cancel_order(order_id, api_symbol, params)
            return response
        except Exception as e:
            logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
            return None

    async def fetch_open_orders(self, symbol: Optional[str] = None, params={}):
        if not self.exchange: return []
        try:
            api_symbol = convert_symbol_to_api_format(symbol) if symbol else None
            open_orders = await self.exchange.fetch_open_orders(api_symbol, params=params)
            return open_orders
        except Exception as e:
            logger.error(f"Error fetching open orders for {symbol}: {e}")
            return []

    async def fetch_position(self, symbol: str, params={}):
        if not self.exchange: return None
        try:
            api_symbol = convert_symbol_to_api_format(symbol)
            positions = await self.exchange.fetch_positions([api_symbol], params=params)
            for position in positions:
                # CCXT position symbol format might vary, ensure robust matching
                if position.get("symbol") == api_symbol or position.get("info", {}).get("symbol") == api_symbol:
                    return position
            return None 
        except Exception as e:
            logger.error(f"Error fetching position for {symbol}: {e}")
            return None

    async def fetch_positions(self, symbols: Optional[List[str]] = None, params={}):
        if not self.exchange: return []
        try:
            api_symbols = [convert_symbol_to_api_format(s) for s in symbols] if symbols else None
            positions = await self.exchange.fetch_positions(api_symbols, params=params)
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    async def set_leverage(self, symbol: str, leverage: int, params={}):
        if not self.exchange: 
            logger.info(f"Simulated set leverage: {leverage}x for {symbol}")
            return True 
        try:
            api_symbol = convert_symbol_to_api_format(symbol)
            response = await self.exchange.set_leverage(leverage, api_symbol, params)
            logger.info(f"Leverage set for {symbol} to {leverage}x: {response}")
            return response
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol} to {leverage}x: {e}")
            return False

    async def set_margin_mode(self, symbol: str, margin_mode: str, params={}):
        if not self.exchange: 
            logger.info(f"Simulated set margin mode: {margin_mode} for {symbol}")
            return True
        try:
            api_symbol = convert_symbol_to_api_format(symbol)
            response = await self.exchange.set_margin_mode(margin_mode.upper(), api_symbol, params)
            logger.info(f"Margin mode set for {symbol} to {margin_mode.upper()}: {response}")
            return response
        except Exception as e:
            logger.error(f"Error setting margin mode for {symbol} to {margin_mode.upper()}: {e}")
            return False

    async def fetch_historical_klines(self, symbol: str, timeframe: str = "1m", since: Optional[int] = None, limit: Optional[int] = None, params={}):
        if not self.exchange: return []
        try:
            api_symbol = convert_symbol_to_api_format(symbol)
            klines_data = await self.exchange.fetch_ohlcv(api_symbol, timeframe, since, limit, params)
            # Convert to Kline model before returning
            klines = [
                Kline(
                    timestamp=k[0],
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    is_closed=True, # fetch_ohlcv usually returns closed candles
                    symbol=api_symbol, # Use the API symbol
                    interval=timeframe
                ) for k in klines_data
            ]
            return klines
        except Exception as e:
            logger.error(f"Error fetching historical klines for {symbol} {timeframe}: {e}")
            return []

class BinanceWebSocketConnector:
    def __init__(self, config_manager: ConfigManager, kline_callback: Callable[[Kline, str], asyncio.Task]):
        self.config_manager = config_manager
        self.kline_callback = kline_callback
        self.ws_url_usdm_mainnet = "wss://fstream.binance.com/stream"
        self.ws_url_usdm_testnet = "wss://stream.binancefuture.com/stream" # Testnet URL for USDM
        self.ws_url_coinm_mainnet = "wss://dstream.binance.com/stream"
        self.ws_url_coinm_testnet = "wss://dstream.binancefuture.com/stream" # Testnet URL for COINM
        self._ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._active_subscriptions: Dict[str, List[str]] = {"USDT_M": [], "COIN_M": []}
        self._is_running = False
        self._reconnect_delay = 5
        self._tasks: List[asyncio.Task] = []
        self._is_testnet = self.config_manager.get_config().get("api", {}).get("testnet", False)

    def _get_ws_url(self, market_type: str) -> str:
        if market_type == "USDT_M":
            return self.ws_url_usdm_testnet if self._is_testnet else self.ws_url_usdm_mainnet
        elif market_type == "COIN_M":
            return self.ws_url_coinm_testnet if self._is_testnet else self.ws_url_coinm_mainnet
        raise ValueError(f"Invalid market type: {market_type}")

    def _get_active_pairs_and_intervals(self) -> Dict[str, List[str]]:
        config = self.config_manager.get_config()
        pairs_config = config.get("pairs", {})
        global_intervals = config.get("global_settings", {}).get("v1_strategy", {}).get("indicator_timeframes", ["1m"])
        
        active_streams_usdm: List[str] = []
        active_streams_coinm: List[str] = []

        for pair_symbol_config, pair_details in pairs_config.items():
            if pair_details.get("enabled", False):
                ws_symbol = convert_symbol_to_ws_format(pair_symbol_config)
                intervals_to_subscribe = pair_details.get("indicator_timeframes", global_intervals)
                contract_type = pair_details.get("contract_type", "USDT_M")
                
                for interval in intervals_to_subscribe:
                    stream_name = f"{ws_symbol}@kline_{interval}"
                    if contract_type == "COIN_M":
                        active_streams_coinm.append(stream_name)
                    else:
                        active_streams_usdm.append(stream_name)
        return {"USDT_M": active_streams_usdm, "COIN_M": active_streams_coinm}

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol, streams: List[str]):
        if not streams: return
        sub_payload = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }
        await ws.send(json.dumps(sub_payload))
        logger.info(f"Subscribed to streams: {streams} on {ws.remote_address}")

    async def _unsubscribe(self, ws: websockets.WebSocketClientProtocol, streams: List[str]):
        if not streams or not ws or not ws.open: return
        unsub_payload = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }
        try:
            await ws.send(json.dumps(unsub_payload))
            logger.info(f"Unsubscribed from streams: {streams} on {ws.remote_address}")
        except Exception as e:
            logger.error(f"Error unsubscribing from {streams}: {e}")

    async def _handle_connection(self, market_type: Literal["USDT_M", "COIN_M"]):
        ws_url = self._get_ws_url(market_type)
        while self._is_running:
            try:
                current_streams_for_market = self._active_subscriptions[market_type]
                if not current_streams_for_market:
                    logger.debug(f"No active streams for {market_type}, connection loop sleeping.")
                    await asyncio.sleep(self._reconnect_delay) 
                    continue

                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self._ws_connections[market_type] = ws
                    logger.info(f"Connected to Binance Futures {market_type} WebSocket ({'Testnet' if self._is_testnet else 'Mainnet'}): {ws_url}")
                    await self._subscribe(ws, current_streams_for_market)
                    self._reconnect_delay = 5 
                    async for message in ws:
                        await self._process_message(message, market_type)
            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
                logger.warning(f"WebSocket connection closed for {market_type}: {e}. Reconnecting...")
            except ConnectionRefusedError:
                logger.error(f"Connection refused for {market_type} WebSocket. Check network or Binance status.")
            except Exception as e:
                logger.error(f"Error in {market_type} WebSocket handler: {e}. Reconnecting...")
            finally:
                if market_type in self._ws_connections:
                    del self._ws_connections[market_type]
                if self._is_running:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(self._reconnect_delay * 1.5, 60) 
                    self._reconnect_delay += random.uniform(0,1)

    async def _process_message(self, message_str: str, market_type: str):
        try:
            data = json.loads(message_str)
            if "stream" in data and "@kline_" in data["stream"]:
                kline_data = data["data"]["k"]
                kline_obj = Kline(
                    timestamp=kline_data["t"],
                    open=float(kline_data["o"]),
                    high=float(kline_data["h"]),
                    low=float(kline_data["l"]),
                    close=float(kline_data["c"]),
                    volume=float(kline_data["v"]),
                    quote_asset_volume=float(kline_data["q"]) if "q" in kline_data else 0.0,
                    number_of_trades=int(kline_data["n"]) if "n" in kline_data else 0,
                    is_closed=kline_data["x"],
                    symbol=kline_data["s"], 
                    interval=kline_data["i"]
                )
                # Ensure kline_callback is awaited if it's a coroutine
                if asyncio.iscoroutinefunction(self.kline_callback):
                    await self.kline_callback(kline_obj, market_type)
                else:
                    self.kline_callback(kline_obj, market_type) # If it's a regular function
            elif "result" in data and data["result"] is None: 
                logger.debug(f"Subscription/Unsubscription confirmed: {data}")
            elif "error" in data:
                logger.error(f"WebSocket error message: {data}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON message: {message_str}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e} - Message: {message_str}")

    async def start(self):
        if self._is_running:
            logger.warning("WebSocket connector already running.")
            return
        self._is_running = True
        self._is_testnet = self.config_manager.get_config().get("api", {}).get("testnet", False)
        self.config_manager.register_callback(self._handle_config_update) 
        await self._handle_config_update(self.config_manager.get_config()) # Initial subscription setup
        logger.info("Binance WebSocket Connector started.")

    async def _handle_config_update(self, new_config: Dict):
        logger.info("Configuration updated, re-evaluating WebSocket subscriptions...")
        self._is_testnet = new_config.get("api", {}).get("testnet", False) # Update testnet status
        new_target_streams = self._get_active_pairs_and_intervals()

        for market_type in ["USDT_M", "COIN_M"]:
            current_ws = self._ws_connections.get(market_type)
            old_streams = self._active_subscriptions.get(market_type, [])
            new_streams_for_market = new_target_streams.get(market_type, [])

            streams_to_add = list(set(new_streams_for_market) - set(old_streams))
            streams_to_remove = list(set(old_streams) - set(new_streams_for_market))

            if current_ws and current_ws.open:
                if streams_to_remove:
                    await self._unsubscribe(current_ws, streams_to_remove)
                if streams_to_add:
                    await self._subscribe(current_ws, streams_to_add)
            elif streams_to_add: 
                # If no connection, but new streams, start a new connection task if not already planned
                # Check if a task for this market_type is already running or scheduled
                is_task_active = False
                for task in self._tasks:
                    # This check is a bit simplistic; ideally, tasks would be identifiable
                    if market_type in str(task): # Heuristic to check if task is for this market type
                        is_task_active = True
                        break
                if not is_task_active:
                    logger.info(f"No active connection for {market_type}, but new streams detected. Scheduling connection.")
                    task = asyncio.create_task(self._handle_connection(market_type))
                    self._tasks.append(task)
            
            self._active_subscriptions[market_type] = new_streams_for_market
            
            # If all streams for a market type are removed and a connection exists, close it.
            if not new_streams_for_market and current_ws and current_ws.open:
                logger.info(f"All streams for {market_type} removed. Closing WebSocket connection.")
                await current_ws.close()
                if market_type in self._ws_connections:
                    del self._ws_connections[market_type]
            # If no streams and no task, ensure no new tasks are created for this market type
            elif not new_streams_for_market:
                 logger.debug(f"No streams for {market_type}, ensuring no connection task is active or created.")

        # Initial connection tasks if not already started
        for market_type in ["USDT_M", "COIN_M"]:
            if self._active_subscriptions.get(market_type) and market_type not in self._ws_connections:
                is_task_active = False
                for task in self._tasks:
                    if market_type in str(task):
                        is_task_active = True
                        break
                if not is_task_active:
                    logger.info(f"Initial streams for {market_type} detected. Scheduling connection.")
                    task = asyncio.create_task(self._handle_connection(market_type))
                    self._tasks.append(task)

    async def stop(self):
        logger.info("Stopping Binance WebSocket Connector...")
        self._is_running = False
        self.config_manager.unregister_callback(self._handle_config_update)
        for market_type, ws in list(self._ws_connections.items()): # Iterate over a copy
            if ws and ws.open:
                logger.info(f"Closing WebSocket connection for {market_type}...")
                await ws.close()
        self._ws_connections.clear()
        
        # Cancel and await pending tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, asyncio.CancelledError):
                    logger.debug(f"Task {self._tasks[i]} was cancelled.")
                elif isinstance(result, Exception):
                    logger.error(f"Task {self._tasks[i]} raised an exception during stop: {result}")
        self._tasks.clear()
        logger.info("Binance WebSocket Connector stopped.")

