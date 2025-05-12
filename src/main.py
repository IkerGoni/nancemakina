import asyncio
import logging
import signal
import os

from src.config_loader import ConfigManager
from src.connectors import BinanceRESTClient, BinanceWebSocketConnector, convert_symbol_to_api_format
from src.data_processor import DataProcessor
from src.signal_engine import SignalEngineV1
from src.order_manager import OrderManager
from src.position_manager import PositionManager
from src.models import Kline, Order # For type hinting if needed
# from src.monitoring import PrometheusMonitor # If implementing Prometheus

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("app.log") # Add file handler if needed from config
    ]
)
logger = logging.getLogger("main_app")

class TradingBot:
    def __init__(self):
        self.config_manager = ConfigManager()
        self._configure_logging()
        
        self.rest_client = BinanceRESTClient(config_manager=self.config_manager)
        self.position_manager = PositionManager(config_manager=self.config_manager)
        self.order_manager = OrderManager(config_manager=self.config_manager, rest_client=self.rest_client)
        # Pass position_manager to order_manager if it needs to directly update it, or handle updates in main loop
        # self.order_manager.position_manager = self.position_manager # Example of direct linking

        self.data_processor = DataProcessor(config_manager=self.config_manager)
        self.ws_connector = BinanceWebSocketConnector(
            config_manager=self.config_manager, 
            kline_callback=self._handle_kline_data # Pass the method reference
        )
        self.signal_engine_v1 = SignalEngineV1(config_manager=self.config_manager, data_processor=self.data_processor)
        
        self.running = False
        self.main_loop_task = None
        self.active_trading_pairs = [] # List of config_symbols (e.g. BTC_USDT)

        self.config_manager.register_callback(self._handle_app_config_update)

    def _configure_logging(self):
        log_level_str = self.config_manager.get_specific_config("logging.level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(log_level) # Set root logger level
        for handler in logging.getLogger().handlers:
            handler.setLevel(log_level)
        logger.info(f"Logging level set to {log_level_str}")
        # TODO: Add file logging based on config if specified

    async def _handle_kline_data(self, kline: Kline, market_type: str):
        # This callback is called by BinanceWebSocketConnector
        # logger.debug(f"Main: Received kline for {kline.symbol} {kline.interval} from {market_type}")
        await self.data_processor.process_kline(kline, market_type)
        # Signal checking can be done here if kline is closed, or in a separate loop
        # For V1, signal engine uses 1m timeframe. Check if this kline is relevant.
        if kline.interval == "1m" and kline.is_closed: # Only check signals on closed 1m candles
            # Find the config_symbol corresponding to kline.symbol (API format)
            config = self.config_manager.get_config()
            target_config_symbol = None
            for cfg_sym, details in config.get("pairs", {}).items():
                if convert_symbol_to_api_format(cfg_sym) == kline.symbol and details.get("enabled"):
                    target_config_symbol = cfg_sym
                    break
            
            if target_config_symbol:
                # Check if a position is already open for this symbol to avoid concurrent trades on same signal type
                if not self.position_manager.has_open_position(kline.symbol):
                    signal = await self.signal_engine_v1.check_signal(kline.symbol, target_config_symbol)
                    if signal:
                        logger.info(f"Signal generated for {signal.symbol}: {signal.direction.value}")
                        # Before handling, ensure no conflicting position exists (double check)
                        if not self.position_manager.has_open_position(signal.symbol):
                            await self.order_manager.handle_trade_signal(signal)
                            # After order_manager handles, it should ideally create a position via position_manager
                            # For now, let's assume OrderManager calls PositionManager or we do it here based on order success
                            # This part needs robust handling of order confirmation and position creation.
                            # For MVP, we assume OrderManager places orders and we rely on reconciliation or user data stream for position updates.
                            # A simple way: after handle_trade_signal, if successful, create a placeholder position.
                            # This is simplified. Real system needs order fill confirmation before creating position.
                            # Let's assume OrderManager will eventually call PositionManager.add_or_update_position
                        else:
                            logger.info(f"Signal for {signal.symbol} ignored, position already open.")
                else:
                    logger.debug(f"Skipping signal check for {kline.symbol}, position already open.")
            # else: logger.debug(f"No enabled config symbol found for API symbol {kline.symbol}")

    async def _handle_order_update_data(self, order_data: Order):
        # This callback would be called by a user data stream connector (not implemented in MVP)
        logger.info(f"Main: Received order update: {order_data.symbol} ID {order_data.order_id} Status {order_data.status}")
        await self.position_manager.update_position_on_order_update(order_data)

    def _handle_app_config_update(self, new_config: dict):
        logger.info("MainApp: Detected configuration change. Applying updates...")
        self._configure_logging() # Update log level if changed
        # Other modules (DataProcessor, WSConnector) handle their own config updates via callbacks.
        # Update active trading pairs for the main loop if it depends on it.
        self.active_trading_pairs = [
            cfg_sym for cfg_sym, details in new_config.get("pairs", {}).items() if details.get("enabled")
        ]
        logger.info(f"Updated active trading pairs: {self.active_trading_pairs}")

    async def start(self):
        if self.running:
            logger.warning("Bot is already running.")
            return
        self.running = True
        logger.info("Starting Binance Futures Trading Bot...")

        # Initial config load for active pairs
        self._handle_app_config_update(self.config_manager.get_config())

        # Reconcile positions with exchange on startup
        # await self.position_manager.reconcile_positions_with_exchange(self.rest_client)
        # Note: reconcile_positions_with_exchange in PositionManager needs the rest_client.
        # Consider passing it during PositionManager init or having a dedicated method in main.
        logger.info("Position reconciliation on startup (manual call if needed)...")
        # await self.position_manager.reconcile_positions_with_exchange(self.rest_client)

        # Start WebSocket connector
        await self.ws_connector.start()

        # TODO: Start Prometheus monitor if implemented
        # if hasattr(self, "prometheus_monitor") and self.prometheus_monitor:
        #     self.prometheus_monitor.start()

        logger.info("Bot is now running. Press Ctrl+C to stop.")
        # Main loop could be here if we need periodic tasks not driven by websockets
        # For now, most logic is event-driven by kline_callback
        # self.main_loop_task = asyncio.create_task(self._periodic_tasks())
        try:
            while self.running: # Keep main alive, tasks run in background
                await asyncio.sleep(1) 
        except asyncio.CancelledError:
            logger.info("Main bot loop cancelled.")

    async def _periodic_tasks(self):
        """For tasks that need to run periodically, e.g., checking all pairs for signals if not event-driven."""
        while self.running:
            # Example: Periodically check all active pairs for signals if not purely event-driven
            # for config_symbol in self.active_trading_pairs:
            #     api_symbol = convert_symbol_to_api_format(config_symbol)
            #     if not self.position_manager.has_open_position(api_symbol):
            #         signal = await self.signal_engine_v1.check_signal(api_symbol, config_symbol)
            #         if signal:
            #             await self.order_manager.handle_trade_signal(signal)
            #     await asyncio.sleep(0.1) # Small delay between checks
            await asyncio.sleep(60) # Run periodic checks every 60 seconds

    async def stop(self):
        if not self.running:
            return
        self.running = False
        logger.info("Stopping Binance Futures Trading Bot...")

        if self.main_loop_task and not self.main_loop_task.done():
            self.main_loop_task.cancel()
            try:
                await self.main_loop_task
            except asyncio.CancelledError:
                logger.info("Main loop task successfully cancelled.")

        # Stop WebSocket connector
        if self.ws_connector:
            await self.ws_connector.stop()
        
        # Close REST client connection
        if self.rest_client:
            await self.rest_client.close_exchange()

        # Stop config watcher
        if self.config_manager:
            self.config_manager.stop_watcher()
        
        # TODO: Stop Prometheus monitor
        # if hasattr(self, "prometheus_monitor") and self.prometheus_monitor:
        #     self.prometheus_monitor.stop()

        logger.info("Bot has stopped.")

def handle_sigterm(sig, frame):
    logger.info("SIGTERM received, initiating graceful shutdown...")
    # This function is synchronous, so we need to schedule the async stop
    # Get the running loop or create a new one to run the stop coroutine
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Assuming `bot` is accessible globally or passed appropriately
    if "bot_instance" in globals() and bot_instance:
        # loop.create_task(bot_instance.stop()) # This might not wait for completion
        # To ensure it runs and completes before exit:
        loop.run_until_complete(bot_instance.stop())
    else:
        logger.warning("Bot instance not found for SIGTERM handler.")
    # The application will exit after this handler returns
    # For a cleaner exit, especially if run_until_complete is used, 
    # it might be better to raise an exception that the main try/except block can catch.
    # Or, ensure the main loop itself checks `self.running` and exits.
    # For now, this will trigger stop and then Python will exit.
    # A more robust way is to set a flag and let the main loop handle exit.
    if "bot_instance" in globals() and bot_instance: # Set flag for main loop
        bot_instance.running = False 

async def main():
    global bot_instance # Make bot instance accessible to signal handler
    bot_instance = TradingBot()

    # Register signal handlers for graceful shutdown
    # For asyncio, it's often better to handle KeyboardInterrupt in the main try/except
    # signal.signal(signal.SIGINT, handle_sigterm) # Ctrl+C
    signal.signal(signal.SIGTERM, handle_sigterm) # Termination signal

    try:
        await bot_instance.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping bot...")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        if bot_instance.running: # Ensure stop is called if not already
            await bot_instance.stop()

if __name__ == "__main__":
    # Ensure a config file exists for the bot to run, copy from example if not
    # ConfigManager handles this internally now.
    asyncio.run(main())

