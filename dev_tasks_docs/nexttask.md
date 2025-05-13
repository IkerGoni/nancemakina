# Development Plan: Step 020 - Integrate "V1_Optimized" Strategy as a Configurable Option

**Task:** Formalize the "Test 12B" (best performing) strategy as a distinct, selectable configuration. This involves adding a "Trend Filter" to `SignalEngineV1`'s capabilities and creating a dedicated YAML configuration file for this optimized V1 strategy.

**Objective:** Allow users to easily select and run the "V1_Optimized" strategy (derived from Test 12B) by choosing a specific configuration file, making the bot's best-found strategy readily accessible.

**Target File(s):**
*   `config/v1_optimized_strategy.yaml` (New configuration file)
*   `config/config.yaml.example` (To include parameters for the new Trend Filter)
*   `src/signal_engine.py` (Add Trend Filter logic to `SignalEngineV1` and its configuration loading)
*   `src/data_processor.py` (No changes expected if 200SMA is already calculated for V1 strategy)
*   `docs/CONFIGURATION_GUIDE.md` (Document the new Trend Filter and the `v1_optimized_strategy.yaml` preset)
*   `tests/unit/test_signal_engine.py` (Add tests for the Trend Filter)

**Context:**
*   Extensive backtesting (Test 12B) has identified an optimized set of parameters for the V1 SMA Crossover strategy, including a crucial "Trend Filter" (price relative to 200SMA).
*   The bot supports loading different YAML configuration files and dynamically loading signal engines.
*   `SignalEngineV1` has a configurable pipeline for filters and SL/TP logic.

**Detailed Plan:**

1.  **Add "Trend Filter" to `SignalEngineV1`'s Configurable Filters:**
    *   **Action (Config):** In `config/config.yaml.example`, under `global_settings.v1_strategy.filters`, add a new filter section for the trend filter:
        ```yaml
        # In config.yaml.example, under global_settings.v1_strategy.filters:
        trend:
          enabled: false # Disabled by default in the generic example
          sma_period: 200 # The SMA period to use for trend determination (matches the long SMA)
          # condition: "price_above_sma_for_long_below_for_short" (This is implicit in the logic)
        ```
    *   **Action (SignalEngine):** In `src/signal_engine.py`:
        *   Update `_get_pair_specific_config` to load settings for `filters.trend` (enabled, sma_period).
        *   Create a new helper method: `_passes_trend_filter(self, latest_kline_data: pd.Series, trend_filter_config: Dict[str, Any], signal_direction: TradeDirection) -> bool:`.
            *   This method will retrieve `latest_kline_data['close']` and `latest_kline_data['sma_long']` (assuming the `sma_period` for the trend filter is the same as the `sma_long_period` used for crossover, or it fetches the specifically configured trend SMA).
            *   If `signal_direction` is LONG, it returns `True` if `close_price > trend_sma_value`.
            *   If `signal_direction` is SHORT, it returns `True` if `close_price < trend_sma_value`.
            *   Log the decision.
        *   Integrate a call to `_passes_trend_filter` into the `check_signal_with_df` pipeline (e.g., after core crossover and other confirmation filters, but before SL/TP calculation). If it returns `False`, log and `return None`.

2.  **Create `config/v1_optimized_strategy.yaml`:**
    *   **Action:** Create this new file.
    *   **Action:** Populate it with all the parameters from your "Test 12B" findings:
        ```yaml
        # config/v1_optimized_strategy.yaml
        engine_config:
          module: "src.signal_engine"
          class: "SignalEngineV1"

        api:
          binance_api_key: "YOUR_API_KEY_PLACEHOLDER"
          binance_api_secret: "YOUR_API_SECRET_PLACEHOLDER"
          testnet: true # Or false, depending on target

        global_settings:
          v1_strategy:
            sma_short_period: 21
            sma_long_period: 200 # Also used by trend filter
            min_signal_interval_minutes: 10 # From Test 12B
            
            indicator_timeframes: ["1m"] # Assuming 10-min interval from Test 12B means signals are on 1m, but decisions/cooldown might relate to a 10m effective view. Clarify if indicators need to be on 10m. For now, assume 1m signals.

            filters:
              buffer_time:
                enabled: false # Specify if Test 12B used it
                # candles: X
              volume:
                enabled: false # Specify if Test 12B used it
                # average_period: Y
                # threshold_ratio: Z
              volatility:
                enabled: false # Specify if Test 12B used it
                # atr_period: A
                # condition_type: "..."
                # value: B
              trend:
                enabled: true # CRUCIAL for Test 12B
                sma_period: 200 # Matches the long SMA for the trend filter

            sl_options:
              primary_type: "pivot"
              pivot_settings:
                lookback_candles: 40 # From Test 12B
              enable_fallback: true
              fallback_type: "atr" # From Test 12B
              atr_settings:        # For ATR-based SL fallback
                atr_period: 14     # From Test 12B
                multiplier: 2.1    # From Test 12B
              percentage_settings: # Still good to have defaults for this type
                value: 0.02 
            
            tp_options:
              primary_type: "rr_ratio"
              rr_ratio_settings:
                value: 2.25 # From Test 12B
            
            # default_leverage, margin_mode, default_margin_usdt as per your Test 12B simulation setup
            default_leverage: 10
            margin_mode: "ISOLATED"
            default_margin_usdt: 50.0 

        pairs:
          BTC_USDT: # Example, adjust to the pair used in Test 12B
            enabled: true
            contract_type: "USDT_M"
            # No overrides needed if global settings match Test 12B for this pair

        logging:
          level: "INFO"

        monitoring:
          prometheus_port: 8000

        backtesting:
          initial_balance_usdt: 1000.0 # Match Test 12B
          fee_percent: 0.02            # Match Test 12B (note: config takes %, script might convert)
          slippage_percent: 0.0 # Assuming Test 12B didn't specify, or add if it did
        ```
    *   **Clarification on "Interval: 10 min" from Test 12B:**
        *   If "Interval: 10 min" means the SMAs and signals were derived from 10-minute candles, then `indicator_timeframes` should be `["10m"]` and `SignalEngineV1` would need to use "10m" data.
        *   If it means `min_signal_interval_minutes: 10`, that's already set.
        *   The current `SignalEngineV1.check_signal` is hardcoded to use "1m" interval from `DataProcessor`. This might need to become configurable if different strategies use different primary signal timeframes. For now, we assume signals are on 1m, and `min_signal_interval_minutes` handles the cooldown.

3.  **Update `DataProcessor` (If Trend Filter SMA is Different):**
    *   **Action:** If the `trend_filter.sma_period` (e.g., 200) is already being calculated as `sma_long` by `DataProcessor`, no changes are needed in `DataProcessor`.
    *   If the trend filter could use an SMA period *different* from `sma_long` (e.g., a very long term SMA like 500), then `DataProcessor` would need to be updated to calculate this additional SMA if configured. (For Test 12B, it seems to be the same as `sma_long`).

4.  **Update Documentation:**
    *   **Action (`docs/CONFIGURATION_GUIDE.md`):** Document the new `filters.trend` section and its parameters.
    *   **Action (`README.md` / New Guide):** Briefly mention the availability of pre-configured strategies like `config/v1_optimized_strategy.yaml` and how to use them with the `--config` flag.

5.  **Testing:**
    *   **Unit Tests:** Add unit tests for `_passes_trend_filter` in `test_signal_engine.py`. Test scenarios where price is above/below the trend SMA for both LONG and SHORT signals.
    *   **Backtesting:**
        *   Run `scripts/backtest.py --config config/v1_optimized_strategy.yaml --data path/to/merged_futures_data_half_copy.csv` (using the same dataset as Test 12B).
        *   Compare the backtest results (number of trades, P/L, win rate) with your manual Test 12B results. They should be very close if all parameters and logic are replicated correctly. Pay attention to the number of trades to see if the trend filter is having the expected effect.
        *   Examine `DEBUG` logs to confirm the trend filter is being applied and making correct pass/fail decisions.

**Acceptance Criteria:**
*   A new `filters.trend` option is available in `config.yaml.example` and can be configured.
*   `SignalEngineV1` correctly implements the trend filter logic.
*   A new `config/v1_optimized_strategy.yaml` file exists, containing all parameters from "Test 12B".
*   Running the backtester with `config/v1_optimized_strategy.yaml` and the corresponding dataset reproduces (or very closely matches) the results of "Test 12B".
*   Documentation is updated to reflect the new trend filter and the availability of the optimized strategy configuration.