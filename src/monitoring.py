from prometheus_client import start_http_server, Counter, Gauge, Enum, Info
import time
import logging
import asyncio

logger = logging.getLogger(__name__)

class PrometheusMonitor:
    def __init__(self, port: int = 8000):
        self.port = port
        self.start_time = time.time()

        # --- General Metrics ---
        self.bot_info = Info(
            "trading_bot_info", 
            "Information about the trading bot"
        )
        self.bot_uptime_seconds = Gauge(
            "trading_bot_uptime_seconds", 
            "Time in seconds since the bot started"
        )
        self.bot_errors_total = Counter(
            "trading_bot_errors_total", 
            "Total number of critical errors encountered by the bot",
            ["module", "error_type"]
        )

        # --- WebSocket Metrics ---
        self.websocket_connection_status = Enum(
            "trading_bot_websocket_connection_status",
            "Status of the WebSocket connection",
            ["market_type"], # e.g., USDT_M, COIN_M
            states=["connected", "disconnected", "connecting", "error"]
        )
        self.websocket_messages_received_total = Counter(
            "trading_bot_websocket_messages_received_total",
            "Total number of messages received via WebSocket",
            ["market_type", "message_type"] # e.g., kline, depthUpdate, userData
        )

        # --- Data Processing Metrics ---
        self.kline_processed_total = Counter(
            "trading_bot_klines_processed_total",
            "Total number of klines processed",
            ["symbol", "interval"]
        )
        self.indicator_calculation_duration_seconds = Gauge(
            "trading_bot_indicator_calculation_duration_seconds",
            "Duration of the last indicator calculation cycle",
            ["symbol", "interval"]
        )

        # --- Signal Engine Metrics ---
        self.signals_generated_total = Counter(
            "trading_bot_signals_generated_total",
            "Total number of trade signals generated",
            ["symbol", "strategy", "direction"] # e.g., BTCUSDT, V1_SMA, LONG
        )

        # --- Order Management Metrics ---
        self.orders_placed_total = Counter(
            "trading_bot_orders_placed_total",
            "Total number of orders placed",
            ["symbol", "order_type", "side"]
        )
        self.orders_failed_total = Counter(
            "trading_bot_orders_failed_total",
            "Total number of orders that failed to place or were rejected",
            ["symbol", "reason"]
        )
        self.orders_filled_total = Counter(
            "trading_bot_orders_filled_total",
            "Total number of orders filled",
            ["symbol", "order_type", "side"]
        )

        # --- Position Management Metrics ---
        self.active_positions_count = Gauge(
            "trading_bot_active_positions_count",
            "Current number of active positions",
            ["contract_type"] # USDT_M, COIN_M
        )
        self.position_pnl_unrealized = Gauge(
            "trading_bot_position_pnl_unrealized",
            "Unrealized PnL for an active position",
            ["symbol"]
        )

        self._server_started = False

    def start(self):
        if self._server_started:
            logger.warning("Prometheus metrics server already started.")
            return
        try:
            start_http_server(self.port)
            self._server_started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
            # Set initial info
            self.bot_info.info({"version": "0.1.0-mvp", "name": "BinanceFuturesBot"})
            # Start periodic update of uptime
            asyncio.create_task(self._update_uptime_periodically())
        except OSError as e:
            logger.error(f"Could not start Prometheus server on port {self.port}: {e}. Metrics will not be available.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while starting Prometheus server: {e}")

    async def _update_uptime_periodically(self):
        while self._server_started: # Or a separate running flag for the task
            self.bot_uptime_seconds.set(time.time() - self.start_time)
            await asyncio.sleep(5) # Update every 5 seconds

    def stop(self):
        # The prometheus_client library doesn\t provide a direct way to stop the HTTP server cleanly.
        # It runs in a separate daemon thread. For a real production app, this might need a custom server.
        # For this project, we assume the process termination will handle it.
        logger.info("Prometheus metrics server stopping (typically handled by process exit).")
        self._server_started = False # To stop uptime updates

    # --- Helper methods to update metrics ---
    def inc_bot_error(self, module: str, error_type: str = "general"):
        self.bot_errors_total.labels(module=module, error_type=error_type).inc()

    def set_websocket_status(self, market_type: str, status: str):
        # Ensure status is one of the Enum states
        valid_states = self.websocket_connection_status._states
        if status not in valid_states:
            logger.warning(f"Invalid WebSocket status 	{status}	 for market 	{market_type}	. Valid states: {valid_states}")
            self.websocket_connection_status.labels(market_type=market_type).state("error")
            return
        self.websocket_connection_status.labels(market_type=market_type).state(status)

    def inc_websocket_message(self, market_type: str, message_type: str):
        self.websocket_messages_received_total.labels(market_type=market_type, message_type=message_type).inc()

    def inc_kline_processed(self, symbol: str, interval: str):
        self.kline_processed_total.labels(symbol=symbol, interval=interval).inc()

    def set_indicator_calculation_duration(self, symbol: str, interval: str, duration_seconds: float):
        self.indicator_calculation_duration_seconds.labels(symbol=symbol, interval=interval).set(duration_seconds)

    def inc_signal_generated(self, symbol: str, strategy: str, direction: str):
        self.signals_generated_total.labels(symbol=symbol, strategy=strategy, direction=direction).inc()

    def inc_order_placed(self, symbol: str, order_type: str, side: str):
        self.orders_placed_total.labels(symbol=symbol, order_type=order_type, side=side).inc()

    def inc_order_failed(self, symbol: str, reason: str = "unknown"):
        self.orders_failed_total.labels(symbol=symbol, reason=reason).inc()

    def inc_order_filled(self, symbol: str, order_type: str, side: str):
        self.orders_filled_total.labels(symbol=symbol, order_type=order_type, side=side).inc()

    def set_active_positions_count(self, contract_type: str, count: int):
        self.active_positions_count.labels(contract_type=contract_type).set(count)
    
    def update_active_positions_gauge(self, positions_usdt_m: int, positions_coin_m: int):
        self.set_active_positions_count(contract_type="USDT_M", count=positions_usdt_m)
        self.set_active_positions_count(contract_type="COIN_M", count=positions_coin_m)

    def set_position_pnl(self, symbol: str, pnl: float):
        self.position_pnl_unrealized.labels(symbol=symbol).set(pnl)

# Example Usage (typically integrated into main_app.py)
async def main_monitor_test():
    logging.basicConfig(level=logging.INFO)
    monitor = PrometheusMonitor(port=8088) # Use a different port for testing
    monitor.start()

    # Simulate some bot activity
    monitor.set_websocket_status(market_type="USDT_M", status="connecting")
    await asyncio.sleep(1)
    monitor.set_websocket_status(market_type="USDT_M", status="connected")
    monitor.inc_websocket_message(market_type="USDT_M", message_type="kline")
    monitor.inc_kline_processed(symbol="BTCUSDT", interval="1m")
    monitor.set_indicator_calculation_duration(symbol="BTCUSDT", interval="1m", duration_seconds=0.05)
    monitor.inc_signal_generated(symbol="BTCUSDT", strategy="V1_SMA", direction="LONG")
    monitor.inc_order_placed(symbol="BTCUSDT", order_type="MARKET", side="BUY")
    monitor.inc_order_filled(symbol="BTCUSDT", order_type="MARKET", side="BUY")
    monitor.set_active_positions_count(contract_type="USDT_M", count=1)
    monitor.set_position_pnl(symbol="BTCUSDT", pnl=10.5)
    monitor.inc_bot_error(module="SignalEngine", error_type="DataMissing")

    try:
        while True:
            # Uptime is updated automatically by the monitor itself
            await asyncio.sleep(10)
            logger.info("Monitor test running... check http://localhost:8088/metrics")
    except KeyboardInterrupt:
        logger.info("Stopping monitor test...")
    finally:
        monitor.stop()

if __name__ == "__main__":
    asyncio.run(main_monitor_test())
