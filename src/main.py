import asyncio
import logging
import signal
import os
from typing import Optional

# Let's use absolute imports to be consistent and avoid potential issues
from src.config_loader import ConfigManager
from src.connectors import BinanceRESTClient, BinanceWebSocketConnector, convert_symbol_to_api_format
from src.data_processor import DataProcessor
from src.signal_engine import SignalEngineV1
from src.order_manager import OrderManager
from src.position_manager import PositionManager
from src.models import Kline, Order, TradeSignal  # For type hinting
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
    def __init__(self) -> None:
        self.config_manager = ConfigManager()
        self._configure_logging()
        
        self.rest_client = BinanceRESTClient(config_manager=self.config_manager)
        self.position_manager = PositionManager(config_manager=self.config_manager)
        self.order_manager = OrderManager(config_manager=self.config_manager, rest_client=self.rest_client)
        # Note: PositionManager updates rely on user data stream (to be implemented) or periodic reconciliation
        # Direct linking could be done if needed: self.order_manager.position_manager = self.position_manager

        self.data_processor = DataProcessor(config_manager=self.config_manager)
        self.ws_connector = BinanceWebSocketConnector(
            config_manager=self.config_manager, 
            kline_callback=self._handle_kline_data
        )
        self.signal_engine_v1 = SignalEngineV1(config_manager=self.config_manager, data_processor=self.data_processor)
        
        self.running = False
        self.active_trading_pairs = []  # List of config_symbols (e.g. BTC_USDT)

        self.config_manager.register_callback(self._handle_app_config_update)

    def _configure_logging(self) -> None:
        """Configure logging based on settings from config."""
        log_level_str = self.config_manager.get_specific_config("logging.level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(log_level)  # Set root logger level
        for handler in logging.getLogger().handlers:
            handler.setLevel(log_level)
        logger.info(f"Logging level set to {log_level_str}")
        # TODO: Add file logging based on config if specified

    async def _handle_kline_data(self, kline: Kline, market_type: str) -> None:
        """
        Process incoming kline data from WebSocket and trigger signal checks when appropriate.
        Includes error handling to prevent WebSocket callback crashes.
        """
        try:
            # Process the kline data through DataProcessor
            await self.data_processor.process_kline(kline, market_type)
        except Exception as e:
            logger.error(f"Error processing kline for {kline.symbol}: {e}", exc_info=True)
            return  # Skip signal check if data processing fails

        # For V1, only check signals on closed 1m candles
        if kline.interval == "1m" and kline.is_closed:
            try:
                # Find the config_symbol corresponding to kline.symbol (API format)
                config = self.config_manager.get_config()
                target_config_symbol = None
                for cfg_sym, details in config.get("pairs", {}).items():
                    if convert_symbol_to_api_format(cfg_sym) == kline.symbol and details.get("enabled"):
                        target_config_symbol = cfg_sym
                        break
                
                if not target_config_symbol:
                    return  # No enabled config symbol found for this API symbol
                
                # Check if a position is already open for this symbol
                if not self.position_manager.has_open_position(kline.symbol):
                    signal = await self.signal_engine_v1.check_signal(kline.symbol, target_config_symbol)
                    if signal:
                        logger.info(f"Signal generated for {signal.symbol}: {signal.direction.value}")
                        # Double check no position exists
                        if not self.position_manager.has_open_position(signal.symbol):
                            try:
                                await self.order_manager.handle_trade_signal(signal)
                                logger.info(f"Signal processed by OrderManager. Position tracking update will be handled via reconciliation or user data stream.")
                                # Note: In the current architecture, position updates rely on:
                                # 1. User data stream (to be implemented in future)
                                # 2. Periodic reconciliation
                            except Exception as e:
                                logger.error(f"Error handling trade signal for {signal.symbol}: {e}", exc_info=True)
                        else:
                            logger.info(f"Signal for {signal.symbol} ignored, position already open.")
                else:
                    logger.debug(f"Skipping signal check for {kline.symbol}, position already open.")
            except Exception as e:
                logger.error(f"Error during signal processing for {kline.symbol}: {e}", exc_info=True)

    async def _handle_order_update_data(self, order_data: Order) -> None:
        """Handle order updates from user data stream (to be implemented)."""
        logger.info(f"Main: Received order update: {order_data.symbol} ID {order_data.order_id} Status {order_data.status}")
        try:
            await self.position_manager.update_position_on_order_update(order_data)
        except Exception as e:
            logger.error(f"Error updating position based on order update: {e}", exc_info=True)

    def _handle_app_config_update(self, new_config: dict) -> None:
        """Handle configuration updates affecting the main application."""
        logger.info("MainApp: Detected configuration change. Applying updates...")
        self._configure_logging()  # Update log level if changed
        
        # Update active trading pairs
        self.active_trading_pairs = [
            cfg_sym for cfg_sym, details in new_config.get("pairs", {}).items() if details.get("enabled")
        ]
        logger.info(f"Updated active trading pairs: {self.active_trading_pairs}")

    async def start(self) -> None:
        """Start the trading bot and all its components."""
        if self.running:
            logger.warning("Bot is already running.")
            return
        
        logger.info("Starting Binance Futures Trading Bot...")
        self.running = True

        # Initial config load for active pairs
        self._handle_app_config_update(self.config_manager.get_config())

        # Reconcile positions with exchange on startup
        logger.info("Reconciling positions with exchange on startup...")
        try:
            await self.position_manager.reconcile_positions_with_exchange(self.rest_client)
            logger.info("Position reconciliation completed.")
        except Exception as e:
            logger.error(f"Error during position reconciliation: {e}", exc_info=True)

        # Start WebSocket connector
        try:
            await self.ws_connector.start()
            logger.info("WebSocket connector started successfully.")
        except Exception as e:
            logger.error(f"Failed to start WebSocket connector: {e}", exc_info=True)
            self.running = False
            return

        logger.info("Bot is now running. Press Ctrl+C to stop.")
        
        # Main event loop - keep the application alive
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Main bot loop cancelled.")

    async def stop(self) -> None:
        """Stop the trading bot and all its components gracefully."""
        if not self.running:
            logger.info("Bot is not running.")
            return
        
        logger.info("Stopping Binance Futures Trading Bot...")
        self.running = False

        # Stop WebSocket connector
        if self.ws_connector:
            try:
                await self.ws_connector.stop()
                logger.info("WebSocket connector stopped successfully.")
            except Exception as e:
                logger.error(f"Error stopping WebSocket connector: {e}", exc_info=True)
        
        # Close REST client connection
        if self.rest_client:
            try:
                await self.rest_client.close_exchange()
                logger.info("REST client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing REST client: {e}", exc_info=True)

        # Stop config watcher
        if self.config_manager:
            try:
                self.config_manager.stop_watcher()
                logger.info("Config manager watcher stopped successfully.")
            except Exception as e:
                logger.error(f"Error stopping config watcher: {e}", exc_info=True)
        
        logger.info("Bot has stopped.")


def handle_sigterm(sig, frame) -> None:
    """
    Handle SIGTERM signal for graceful shutdown.
    Only sets the running flag to False, letting the main loop handle the actual shutdown.
    """
    logger.info("SIGTERM received, initiating graceful shutdown...")
    # Simply set the running flag to False
    if "bot_instance" in globals() and bot_instance:
        bot_instance.running = False
    else:
        logger.warning("Bot instance not found for SIGTERM handler.")

async def main() -> None:
    """Main entry point for the application."""
    global bot_instance  # Make bot instance accessible to signal handler
    bot_instance = None
    
    try:
        bot_instance = TradingBot()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, handle_sigterm)
        
        await bot_instance.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping bot...")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        if bot_instance and bot_instance.running:
            await bot_instance.stop()

if __name__ == "__main__":
    # Ensure a config file exists for the bot to run, copy from example if not
    # ConfigManager handles this internally now.
    asyncio.run(main())

