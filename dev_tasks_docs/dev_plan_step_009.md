# Development Plan: Step 009 - Refine Main Application Logic (`main.py`)

**Task:** Step 009 - Review and refine the main application logic and orchestration in `src/main.py`. While the basic structure exists, ensure robustness, proper asynchronous handling, and clear data flow management.

**Objective:** Solidify the main application entry point (`TradingBot` class), ensuring all components are initialized correctly, data flows smoothly between them (especially WebSocket -> DataProcessor -> SignalEngine -> OrderManager), configuration updates are handled appropriately by the main application if needed, and graceful startup/shutdown procedures are robust.

**Target File(s):**
*   `src/main.py` (Primary file to review/modify)

**Context:**
*   `src/main.py` defines the `TradingBot` class orchestrating other modules.
*   It initializes `ConfigManager`, `BinanceRESTClient`, `PositionManager`, `OrderManager`, `DataProcessor`, `BinanceWebSocketConnector`, and `SignalEngineV1`.
*   It defines `_handle_kline_data` as the callback for the WebSocket connector, which processes klines and triggers signal checks.
*   It includes basic startup (`start`) and shutdown (`stop`) logic using `asyncio`.
*   It registers a callback (`_handle_app_config_update`) for config changes.
*   A placeholder `_periodic_tasks` method exists but isn't used in the primary event-driven flow.

**Detailed Plan:**

1.  **Review Initialization (`__init__`):**
    *   **Action:** Verify the order of initialization makes sense (e.g., `ConfigManager` first).
    *   **Action:** Confirm all necessary dependencies are passed correctly between components (e.g., `rest_client` to `OrderManager`).
    *   **Action:** Check if `PositionManager` needs to be passed to `OrderManager` or if position updates should be handled via events/callbacks routed through `main.py`. *Decision: For MVP, keep it simple. Assume `OrderManager` might eventually update `PositionManager`, or rely on `update_position_on_order_update` driven by user data stream later. No immediate changes needed unless a direct link is required now.*
    *   **Action:** Ensure logging is configured early based on the loaded config.

2.  **Review Kline Handling (`_handle_kline_data`):**
    *   **Action:** Verify robust error handling within the callback. What happens if `data_processor.process_kline` or `signal_engine.check_signal` raises an exception? Ensure exceptions are caught and logged without crashing the main loop or the WebSocket connection handler.
        *   *Refinement:* Wrap calls to `process_kline` and `check_signal` / `handle_trade_signal` in `try...except` blocks, logging errors appropriately.
    *   **Action:** Confirm the logic for triggering `signal_engine.check_signal` is correct (e.g., only on closed 1-minute candles for V1).
    *   **Action:** Double-check the mapping from API symbol (e.g., "BTCUSDT") back to the `config_symbol` (e.g., "BTC_USDT") needed for `signal_engine.check_signal`.
    *   **Action:** Re-evaluate the check `if not self.position_manager.has_open_position(kline.symbol):` before calling `check_signal`. Is this sufficient? Should it check the *intended direction* vs. existing position? *Decision: For V1 (no hedge mode), checking if *any* position exists for the symbol is a reasonable simplification to prevent conflicting signals on the same pair.* Keep as is for MVP.
    *   **Action:** Review the flow after a signal is generated and `order_manager.handle_trade_signal` is called. How is the `PositionManager` updated? *Decision: Currently, there's no explicit update call in `main`. This relies on a future User Data Stream integration or manual reconciliation. Add a log message indicating a signal was passed to OrderManager, but position tracking update relies on other mechanisms.*

3.  **Review Startup Logic (`start`):**
    *   **Action:** Ensure `_handle_app_config_update` is called initially to set up active pairs.
    *   **Action:** Review the position reconciliation call. Is it necessary for MVP startup? *Decision: It's good practice. Keep the placeholder call `await self.position_manager.reconcile_positions_with_exchange(self.rest_client)` but ensure `rest_client` is correctly passed. Add logging around it.*
    *   **Action:** Verify WebSocket connector is started (`await self.ws_connector.start()`).
    *   **Action:** Review the main `while self.running: await asyncio.sleep(1)` loop. Is it necessary if all logic is event-driven via `_handle_kline_data`? *Decision: Yes, it keeps the main application alive to receive signals (like SIGTERM) and allows background tasks (like WebSocket handlers) to run. Keep it.*
    *   **Action:** Ensure comprehensive logging during startup phases.

4.  **Review Shutdown Logic (`stop`):**
    *   **Action:** Verify the order of stopping components (e.g., WebSocket connector before REST client, config watcher last).
    *   **Action:** Ensure all awaited stop methods (`ws_connector.stop`, `rest_client.close_exchange`, `config_manager.stop_watcher`) are actually called.
    *   **Action:** Add error handling around the `stop` calls in case a component fails to stop gracefully.
    *   **Action:** Ensure `self.running` flag is set to `False` early in the `stop` sequence.

5.  **Review Asynchronous Loop (`_periodic_tasks`):**
    *   **Action:** This task seems commented out or not actively used in the event-driven flow.
    *   *Decision:* Remove the unused `self.main_loop_task = asyncio.create_task(self._periodic_tasks())` line from `start` and the `_periodic_tasks` method itself if it's not required for the V1 event-driven strategy. This simplifies the main loop. If periodic checks (like reconciliation) are needed *during* runtime, this structure could be revived.

6.  **Review Configuration Handling (`_handle_app_config_update`):**
    *   **Action:** Confirm that changes relevant to `main.py` itself (like logging level, active pairs list) are handled correctly.
    *   **Action:** Ensure changes handled by other modules via their own callbacks (e.g., WebSocket subscriptions, DataProcessor buffers) don't need redundant handling here.

7.  **Review Signal Handling (`handle_sigterm`):**
    *   **Action:** Verify the logic correctly finds the running event loop and schedules the `bot_instance.stop()` coroutine.
    *   **Action:** Test the `loop.run_until_complete(bot_instance.stop())` approach. Does it block the signal handler? Is it safe? *Alternative:* A common pattern is to set `bot_instance.running = False` in the signal handler and let the main `while self.running:` loop break naturally, triggering the `finally` block where `stop()` is called. This is often cleaner.
        *   *Refinement:* Modify `handle_sigterm` to just set `bot_instance.running = False`. Ensure the main `try...finally` block in `main()` calls `await bot_instance.stop()` if `bot_instance` exists. Remove `run_until_complete` from the handler.

**Acceptance Criteria:**
*   `src/main.py` code is reviewed and potentially refactored for clarity and robustness.
*   Error handling in `_handle_kline_data` is implemented.
*   Startup and shutdown sequences are verified and logged clearly.
*   Signal handling (`handle_sigterm`) uses a non-blocking approach by setting the `running` flag.
*   Unnecessary code (like the unused `_periodic_tasks` loop) is removed.
*   Data flow between components within `main.py` is clear and logical.
*   Interaction with `PositionManager` post-signal handling is clarified (acknowledged as dependent on future steps for full automation).