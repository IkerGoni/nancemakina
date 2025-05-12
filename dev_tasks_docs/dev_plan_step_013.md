# Development Plan: Step 013 - Validate End-to-End Functionality on Testnet

**Task:** Step 013 - Configure and run the bot on the Binance Futures Testnet to validate end-to-end functionality of the V1 Strategy MVP.

**Objective:** Verify that the integrated system works correctly in a simulated live environment, including WebSocket data reception, indicator calculation, signal generation, order placement (Entry, SL, TP), position tracking (manual observation), logging, and basic monitoring. Identify and document any bugs or discrepancies.

**Target File(s):**
*   `config/config.yaml` (To be configured for Testnet)
*   Bot logs (output during Testnet run)
*   (Potentially) `docs/TROUBLESHOOTING.md` (To update with findings)

**Context:**
*   All core modules have initial implementations and unit/integration tests.
*   The bot is designed to run via `python -m src.main`.
*   Configuration is handled via `config/config.yaml`.
*   The bot supports connecting to Testnet via the `api.testnet: true` flag.
*   Requires Binance Futures Testnet API Keys.

**Prerequisites:**
*   User must obtain API Key and Secret from the [Binance Futures Testnet website](https://testnet.binancefuture.com/).
*   Testnet account needs sufficient test funds (usually provided automatically or via a faucet on the Testnet site).

**Detailed Plan:**

1.  **Configure for Testnet:**
    *   **Action:** Copy `config/config.yaml.example` to `config/config.yaml`.
    *   **Action:** Edit `config/config.yaml`:
        *   Enter the obtained **Testnet** API Key and Secret under the `api` section.
        *   Add or set `testnet: true` under the `api` section.
        *   Enable at least one or two **USDT-M perpetual** pairs commonly available on Testnet (e.g., `BTC_USDT`, `ETH_USDT`) under the `pairs` section. Set `enabled: true`.
        *   Configure `leverage` and `margin_usdt` for the chosen pairs to reasonable test values (e.g., 5x leverage, 50 USDT margin). Ensure the margin is small relative to available Testnet funds.
        *   Set `global_settings.v1_strategy.min_signal_interval_minutes` to a low value (e.g., 1 or 5) for faster testing, but be aware this might increase noise.
        *   Set `logging.level` to `INFO` or `DEBUG` for detailed observation.
        *   Ensure `monitoring.prometheus_port` is set if monitoring is desired.
    *   **Action:** Verify Python environment has all dependencies installed from `requirements.txt`.

2.  **Run the Bot:**
    *   **Action:** Open a terminal in the project's root directory.
    *   **Action:** Execute the bot: `python -m src.main`
    *   **Action:** Monitor the console output / log file (`logs/bot.log` if configured) for startup messages, connection status, and errors.

3.  **Monitor Bot Behavior & Verify Functionality:**
    *   **Verification:** **WebSocket Connection & Data:**
        *   Check logs for successful connection messages to the *Testnet* WebSocket endpoints (e.g., `stream.binancefuture.com`).
        *   Check logs for confirmation of subscriptions to the kline streams for the enabled pairs/timeframes (e.g., `btcusdt@kline_1m`).
        *   Observe if kline data appears to be processing (debug logs might show this, or inferred if signals occur).
    *   **Verification:** **Indicator Calculation:**
        *   (Difficult to verify directly without DEBUG logs showing calculated SMAs). Infer correctness if sensible signals are generated based on visible chart patterns on the Testnet trading interface.
    *   **Verification:** **Signal Generation (V1 SMA Crossover):**
        *   Observe the Testnet chart for the enabled pair(s) on the 1-minute timeframe.
        *   Watch for potential SMA 21 / SMA 200 crossovers.
        *   Check bot logs for messages indicating a "Signal generated" (LONG or SHORT) when a crossover appears to occur on the chart. Note the signal details logged (entry, SL, TP).
    *   **Verification:** **Order Placement:**
        *   When a signal is logged, check the logs for messages indicating "Placing entry order", "Placing SL order", "Placing TP order".
        *   Log in to the Binance Futures Testnet web interface.
        *   Check the "Open Orders" tab for the corresponding pair. Verify that a MARKET order executed (check Trade History), and that corresponding STOP_MARKET (SL) and TAKE_PROFIT_MARKET (TP) orders were created with `reduceOnly` flag set and prices matching (or very close due to precision) the logged SL/TP values.
        *   Verify the quantities match the expected calculated size based on config (margin * leverage / entry).
    *   **Verification:** **Position Tracking (Manual):**
        *   After the entry order fills, check the "Positions" tab on the Testnet interface.
        *   Verify a position exists for the pair with the correct side (Long/Short), entry price, and quantity.
        *   Observe if the position closes when price hits the SL or TP level (check trade history and open orders again). *Note: Automated position tracking within the bot via `PositionManager` relies heavily on User Data Stream (not in MVP) or reconciliation, so internal state might lag the exchange.*
    *   **Verification:** **Logging:**
        *   Review the log output for clarity, appropriate levels (INFO/DEBUG), and useful information regarding bot status, signals, orders, and potential errors.
    *   **Verification:** **Monitoring (Optional):**
        *   If monitoring is enabled, access the Prometheus endpoint (`http://localhost:PORT/metrics`).
        *   Check if metrics like uptime, WebSocket status, klines processed, signals generated, orders placed are updating.

4.  **Identify and Document Issues:**
    *   **Action:** Keep detailed notes of any unexpected behavior, errors in logs, discrepancies between bot actions and Testnet interface state, or crashes.
    *   **Action:** Note specific conditions or sequences of events that trigger issues.
    *   **Action:** If issues are found, attempt to reproduce them.
    *   **Action:** Update `docs/TROUBLESHOOTING.md` with common problems encountered and their solutions/workarounds found during Testnet.

5.  **Refine and Debug (Iterative):**
    *   **Action:** Based on findings, stop the bot (`Ctrl+C`).
    *   **Action:** Modify the code (`src/...`) to fix identified bugs.
    *   **Action:** Restart the bot and re-verify the specific functionality that failed previously.
    *   **Action:** Repeat the monitoring and verification process until the core V1 strategy flow operates reliably on Testnet.

**Acceptance Criteria:**
*   The bot successfully connects to the Binance Futures Testnet WebSocket and REST APIs using provided testnet credentials.
*   Kline data is received and processed for configured pairs.
*   The bot generates V1 SMA crossover signals (both LONG and SHORT) based on observed chart patterns on Testnet.
*   Upon signal generation, the bot successfully places MARKET entry orders and corresponding STOP_MARKET (SL) / TAKE_PROFIT_MARKET (TP) orders with `reduceOnly=true` on the Testnet exchange.
*   Order details (prices, quantities) on Testnet match the bot's calculations and logs (considering precision adjustments).
*   The bot runs stably for a reasonable test period (e.g., several hours) without crashing.
*   Logs provide clear information about the bot's state and actions.
*   Any critical bugs identified during Testnet validation are fixed.