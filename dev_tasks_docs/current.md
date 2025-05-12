# Development Plan: Step 014 - Implement Configurable Algorithm Conditions

**Task:** Refactor `SignalEngineV1` and configuration to support categorized and individually-configurable algorithm conditions (filters and logic modules for SL/TP, and other entry/exit confirmations).

**Objective:** Allow users to enable/disable specific parts of the trading strategy (e.g., volume filters, volatility filters, different SL/TP logic types) via `config.yaml` to customize strategy behavior and complexity.

**Target File(s):**
*   `config/config.yaml.example` (to define the new configuration structure)
*   `src/config_loader.py` (if `ConfigManager` needs adjustment for deeper nested structures, though likely fine)
*   `src/signal_engine.py` (major refactoring of `check_signal` and potentially `_get_pair_specific_config`)
*   `src/data_processor.py` (if new indicators like ATR or average volume are needed for filters)
*   `src/models.py` (if new Pydantic models are needed for filter configurations)

**Context:**
*   Basic SMA crossover detection is confirmed to be working in the `scripts/backtest.py` environment.
*   A preliminary fallback mechanism for Stop Loss (e.g., percentage-based if pivot calculation fails) has likely been introduced into `SignalEngineV1.check_signal()` based on prior debugging efforts.
*   The `config/config.yaml` is configured for sensitive SMA settings and `DEBUG` logging.
*   The current task is to formalize and expand the configurability of various strategy components, making them optional and parameterizable.

**Detailed Plan:**

1.  **Define Configuration Structure for Conditions/Filters:**
    *   **Action:** In `config/config.yaml.example`, define a new, clear, and hierarchical structure for algorithm conditions. This should ideally be under `global_settings.v1_strategy` (e.g., `filters`, `entry_conditions`, `exit_conditions`, `sl_options`, `tp_options`), with corresponding override capabilities at the pair level.
    *   **Categories to Consider:**
        *   **Core Signal Trigger:** (e.g., SMA Crossover - this is the V1 base, assumed always on if V1 strategy is active).
        *   **Entry Confirmation Filters:**
            *   `buffer_time_candles`:
                *   `enabled: true/false`
                *   `candles: 3` (integer, number of candles the crossover must persist).
        *   **Volume Filters:**
            *   `enabled: true/false`
            *   `average_period: 50` (integer, for calculating average volume).
            *   `threshold_ratio: 1.5` (float, e.g., current volume must be X times the average).
        *   **Volatility Filters (Example: ATR-based, can be conceptual for now):**
            *   `enabled: true/false`
            *   `atr_period: 14` (integer).
            *   `condition_type: "min_atr_percentage_of_price"` or `"min_atr_absolute_value"`
            *   `value: 0.5` (float, corresponding to condition_type).
        *   **Stop-Loss Options (`sl_options`):**
            *   `primary_type`: "pivot" | "percentage" | "atr" (string enum).
            *   `pivot_settings`: (object, if `primary_type` is "pivot")
                *   `lookback_candles: 30`
            *   `percentage_settings`: (object, if `primary_type` or `fallback_type` is "percentage")
                *   `value: 0.02` (float, e.g., 2% from entry).
            *   `atr_settings`: (object, if `primary_type` or `fallback_type` is "atr")
                *   `atr_period: 14`
                *   `multiplier: 2.0`
            *   `enable_fallback: true/false`
            *   `fallback_type`: "percentage" | "atr" (string enum, used if `primary_type` fails and `enable_fallback` is true).
        *   **Take-Profit Options (`tp_options`):**
            *   `primary_type`: "rr_ratio" | "atr_multiple" | "fixed_percentage" (string enum).
            *   `rr_ratio_settings`: (object, if `primary_type` is "rr_ratio")
                *   `value: 2.0` (float, Risk/Reward ratio based on calculated SL).
            *   `atr_multiple_settings`: (object, if `primary_type` is "atr_multiple")
                *   `atr_period: 14`
                *   `multiplier_from_entry: 3.0`
            *   `fixed_percentage_settings`: (object, if `primary_type` is "fixed_percentage")
                *   `value: 0.04` (float, e.g., 4% from entry).

2.  **Update Data Models (Pydantic - `src/models.py` - Optional but Recommended):**
    *   **Action:** For more complex nested configurations (like `sl_options` or `filters`), consider defining corresponding Pydantic models. This improves validation, provides type safety, and makes accessing these configs in code cleaner.
        *   Example: `SLOptionsConfig(BaseModel)`, `VolumeFilterConfig(BaseModel)`.

3.  **Enhance `DataProcessor` for New Indicators (If Required):**
    *   **Action:** If new filters/conditions require indicators not currently calculated (e.g., ATR for ATR-based SL/TP or volatility filters, average volume for volume filters), modify `src/data_processor.py`'s `_update_indicators` method to compute and store them.
    *   **Action:** Ensure these new indicators are available in the DataFrame returned by `get_indicator_dataframe`.

4.  **Refactor `SignalEngineV1._get_pair_specific_config`:**
    *   **Action:** Modify this method to robustly load the new, potentially nested, configuration structures for filters, SL options, and TP options. It needs to merge global settings with pair-specific overrides and provide sensible defaults if a sub-section or specific parameter is missing.
        *   Example: `pair_sl_options = {**global_sl_options, **pair_specific_sl_options}`.

5.  **Refactor `SignalEngineV1.check_signal()` into a Pipeline:**
    *   **Action:** Retrieve all relevant configurations (filters, SL/TP options) at the beginning of `check_signal` using the updated `_get_pair_specific_config`.
    *   **Action:** Transform `check_signal` into a sequential pipeline of checks. If any mandatory check fails, the method should log the reason and `return None`.
        1.  **Core Signal Trigger** (e.g., SMA Crossover): If no crossover, return `None`.
        2.  **Entry Confirmation Filters:**
            *   If `buffer_time_candles.enabled`: Check condition. If fail, log and `return None`.
        3.  **Volume Filters:**
            *   If `volume_filter.enabled`: Check condition. If fail, log and `return None`.
        4.  **Volatility Filters:**
            *   If `volatility_filter.enabled`: Check condition. If fail, log and `return None`.
        5.  **Stop-Loss Calculation:**
            *   Call a new helper method, e.g., `_calculate_stop_loss(entry_price, direction, df, sl_options_config)`.
            *   This helper will use `sl_options_config.primary_type` first.
            *   If the primary SL calculation returns `None` (e.g., pivot not found) AND `sl_options_config.enable_fallback` is true, it will then attempt calculation using `sl_options_config.fallback_type`.
            *   If SL is still `None` after attempting primary and fallback (if enabled), log and `return None` from `check_signal`.
        6.  **Take-Profit Calculation:**
            *   Call a new helper method, e.g., `_calculate_take_profit(entry_price, stop_loss_price, direction, tp_options_config, df)`.
            *   This helper will use `tp_options_config.primary_type`.
            *   If TP is `None`, log and `return None` from `check_signal`.
        7.  **Final Validation:** Perform checks like `risk > 0` and ensure TP is logically placed relative to entry and SL. If fail, log and `return None`.
        8.  If all checks pass, construct and return the `TradeSignal`.
    *   **Action:** Log which filters/conditions are active for the current pair and whether each check passed or failed.

6.  **Implement Helper Methods for Filters and SL/TP Logic:**
    *   **Action:** Create new private helper methods in `SignalEngineV1`:
        *   `_passes_buffer_time_filter(df, config)`
        *   `_passes_volume_filter(latest_kline_data, average_volume, config)`
        *   `_passes_volatility_filter(latest_kline_data, atr_value, config)` (if implementing ATR filter)
        *   `_calculate_stop_loss(entry_price, direction, df_full_history, sl_options)`: This will internally call `_find_recent_pivot`, `_calculate_percentage_sl`, or `_calculate_atr_sl` based on `sl_options`.
        *   `_calculate_percentage_sl(entry_price, direction, percentage_value)`
        *   `_calculate_atr_sl(entry_price, direction, atr_value, atr_multiplier)` (if implementing ATR SL)
        *   `_calculate_take_profit(entry_price, stop_loss_price, direction, tp_options, df_full_history, atr_value_for_tp)`: This will handle different TP types.

7.  **Update Unit Tests (`tests/unit/test_signal_engine.py`):**
    *   **Action:** Extensively update existing unit tests for `SignalEngineV1` to reflect the new configurable pipeline.
    *   **Action:** Add many new test cases to verify:
        *   Each filter works correctly (passes/fails appropriately) when enabled and is correctly bypassed when disabled.
        *   Different `primary_sl_type` options are correctly selected and calculated.
        *   The `enable_fallback_sl` and `fallback_sl_type` logic works as expected.
        *   Different `tp_type` options lead to correct TP calculations.
        *   The merging of global and pair-specific filter/SL/TP configurations in `_get_pair_specific_config` is correct.
        *   The `check_signal` pipeline correctly exits if any enabled filter fails.

8.  **Update Documentation (`docs/CONFIGURATION_GUIDE.md`):**
    *   **Action:** Thoroughly document all new configuration sections and parameters for `filters`, `sl_options`, and `tp_options`, including valid values and examples.

**Initial Focus for Implementation (to manage complexity):**

1.  Define the full configuration structure in `config/config.yaml.example`.
2.  Update `_get_pair_specific_config` to load these new structures.
3.  Refactor `check_signal` into the pipeline structure.
4.  Implement the `buffer_time_candles` filter.
5.  Implement the new `_calculate_stop_loss` helper, supporting "pivot" and "percentage" as `primary_type`, and "percentage" as `fallback_type`.
6.  Implement the new `_calculate_take_profit` helper, supporting "rr_ratio" and "fixed_percentage" as `primary_type`.
7.  Volume and ATR-based filters/logic can be added as a subsequent refinement after these foundational changes are stable.

**Acceptance Criteria:**
*   The `config.yaml.example` defines a comprehensive and clear structure for algorithm conditions.
*   `SignalEngineV1` correctly reads, merges, and applies these configurable conditions from `config.yaml`.
*   The `check_signal` method is refactored into a clear pipeline of configurable checks.
*   The bot can operate with various combinations of enabled/disabled filters and different SL/TP calculation strategies based on the configuration.
*   Logs clearly detail which conditions are active and the outcome of each check during signal evaluation.
*   Unit tests for `SignalEngineV1` are updated to cover the new configurable logic and its various paths.
*   The `CONFIGURATION_GUIDE.md` is updated to reflect all new configuration possibilities.