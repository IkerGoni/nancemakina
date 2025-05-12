# Development Plan: Step 016 - Finalize V1 MVP Documentation and Revalidate on Testnet

**Task:** Thoroughly update all project documentation, especially the Configuration Guide, to reflect the complete V1 strategy with all configurable conditions. Perform a comprehensive end-to-end validation of the V1 MVP on the Binance Futures Testnet.

**Objective:** Ensure the trading bot's V1 MVP is well-documented, its configuration is clearly explained, and its functionality is validated in a simulated live trading environment before considering it "feature-complete" for V1.

**Target File(s)/Areas:**
*   `docs/CONFIGURATION_GUIDE.md` (Major update)
*   `README.md` (Review and update setup, config, running sections)
*   `docs/ARCHITECTURE.md` (Minor review for accuracy)
*   `docs/TROUBLESHOOTING.md` (Add any new common issues found)
*   `config/config.yaml.example` (Ensure it's pristine and well-commented)
*   Binance Futures Testnet environment for live-like testing.
*   Bot logs during Testnet operation.

**Context:**
*   The V1 strategy (SMA crossover) is now implemented with a configurable pipeline including:
    *   Buffer time filter.
    *   Volume filter.
    *   ATR-based Volatility filter.
    *   Configurable SL types (pivot, percentage, ATR) with fallback.
    *   Configurable TP types (R:R ratio, fixed percentage, ATR multiple).
*   `DataProcessor` calculates SMAs, Average Volume, and ATR.
*   `scripts/backtest.py` has been used for iterative testing of these components.
*   Core unit tests for individual components are in place.

**Detailed Plan:**

1.  **Update `docs/CONFIGURATION_GUIDE.md`:**
    *   **Action:** This is the highest priority. Systematically go through `config/config.yaml.example`. For *every* parameter, especially under `global_settings.v1_strategy` (including `filters`, `sl_options`, `tp_options`) and pair-specific overrides:
        *   Explain what the parameter does.
        *   Specify its data type (e.g., boolean, integer, float, string enum).
        *   List valid options or expected value ranges (e.g., for `sl_options.primary_type`: "pivot", "percentage", "atr").
        *   Provide a clear example of its usage.
        *   Explain how global and pair-specific settings interact for this parameter.
        *   Clarify default values if not explicitly set in the user's `config.yaml`.
    *   **Action:** Create a clear section for each filter type, explaining its purpose and parameters.
    *   **Action:** Create clear sections for Stop-Loss and Take-Profit options, explaining each type and its associated settings.
    *   **Action:** Ensure the guide is easy to navigate and understand for a new user.

2.  **Review and Update `README.md`:**
    *   **Action:** Verify installation instructions are still accurate.
    *   **Action:** Update the "Configuration" section to briefly mention the key configurable aspects (filters, SL/TP types) and point prominently to the `CONFIGURATION_GUIDE.md` for full details.
    *   **Action:** Ensure the "Running the Bot" section is correct.
    *   **Action:** Review the "Features" list and "Architecture Overview" summary for accuracy.

3.  **Review `config/config.yaml.example`:**
    *   **Action:** Ensure it is a clean, well-commented template that matches the `CONFIGURATION_GUIDE.md`.
    *   **Action:** Ensure all new configurable options have example values and explanatory comments.
    *   **Action:** Remove any temporary debug settings (like SMA 1/2) and revert to sensible defaults (e.g., SMA 21/200).

4.  **Minor Review of Other Documentation:**
    *   **Action:** Briefly review `docs/ARCHITECTURE.md` to ensure the component descriptions are still generally accurate after the recent refactoring of `SignalEngineV1`.
    *   **Action:** Add any common issues or troubleshooting tips discovered during the implementation of advanced filters to `docs/TROUBLESHOOTING.md`.

5.  **Comprehensive Testnet Validation (Revisit of Step 013):**
    *   **Prerequisites:**
        *   Obtain fresh Binance Futures Testnet API keys if necessary.
        *   Ensure sufficient Testnet funds.
    *   **Configuration for Testnet:**
        *   Create a `config/config.yaml` for Testnet.
        *   Set `api.testnet: true`.
        *   Choose a common, liquid pair (e.g., BTCUSDT).
        *   Configure `global_settings.v1_strategy` with a *sensible, non-debug* set of parameters:
            *   Standard SMA periods (e.g., 21/50 or 21/200, depending on what you want to test).
            *   `min_signal_interval_minutes` to a reasonable value (e.g., 15-60 minutes).
            *   Enable one or two filters at a time to observe their behavior (e.g., start with only the buffer filter, then add volume, then ATR).
            *   Choose a primary SL type (e.g., "pivot") and a fallback (e.g., "percentage").
            *   Choose a primary TP type (e.g., "rr_ratio").
            *   Set `logging.level: "INFO"` (or `DEBUG` if issues arise).
    *   **Execution and Monitoring:**
        *   Run the bot: `python -m src.main`.
        *   Monitor bot logs closely for:
            *   Successful WebSocket connections and data streams.
            *   Correct indicator calculations (inferred from signal generation).
            *   Signal generation messages, including which filters passed/failed.
            *   SL/TP calculation details (which method was used: primary or fallback).
            *   Order placement attempts and confirmations from the (mocked or real testnet) `BinanceRESTClient`.
        *   On the Binance Futures Testnet interface:
            *   Verify if orders (Market entry, Stop-Market SL, Take-Profit-Market TP) are placed correctly when signals occur.
            *   Check order parameters (symbol, side, quantity, stop prices, reduceOnly flags).
            *   Observe if positions are opened and if SL/TP orders trigger position closure when price levels are hit.
    *   **Duration:** Let the bot run for a significant period (e.g., several hours, or through various market conditions if possible) to catch intermittent issues.
    *   **Iterate:** If issues are found, stop the bot, debug, fix, and restart Testnet validation.

**Acceptance Criteria:**
*   `docs/CONFIGURATION_GUIDE.md` is comprehensive, accurate, and clearly explains all V1 strategy configuration options, including all filters, SL types, and TP types.
*   `README.md` and `config/config.yaml.example` are up-to-date and user-friendly.
*   The bot, configured with a complete V1 strategy, runs stably on the Binance Futures Testnet.
*   Signals are generated according to the configured strategy rules and filters.
*   Orders (entry, SL, TP) are correctly placed on the Testnet exchange with appropriate parameters.
*   Positions are managed (opened/closed) as expected based on signals and SL/TP triggers.
*   Logs provide clear insight into the bot's decision-making process during Testnet operation.