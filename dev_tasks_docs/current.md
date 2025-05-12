# Development Plan: Step 015 - Implement Advanced Filters (Volume & Volatility/ATR)

**Task:** Implement the configurable Volume filter and a conceptual ATR-based Volatility filter within `SignalEngineV1`, and update `DataProcessor` to calculate the necessary indicators (Average Volume, ATR).

**Objective:** Enhance the V1 trading strategy with optional volume and volatility confirmation filters, controllable via `config.yaml`, to improve signal quality.

**Target File(s):**
*   `config/config.yaml.example` (to add parameters for new filters)
*   `src/config_loader.py` (no changes expected, but verify)
*   `src/signal_engine.py` (implement filter logic in `check_signal` and new helper methods)
*   `src/data_processor.py` (add Average Volume and ATR calculation to `_update_indicators`)
*   `src/models.py` (if deciding to use Pydantic models for these filter configs)
*   `tests/unit/test_signal_engine.py` (add tests for new filters)
*   `tests/unit/test_data_processor.py` (add tests for new indicator calculations)

**Context:**
*   The `SignalEngineV1` has been refactored into a pipeline of configurable checks.
*   Basic filters (buffer time) and configurable SL/TP (pivot, percentage, rr_ratio, fixed_percentage) are assumed to be implemented and working.
*   The configuration structure for `filters.volume` and `filters.volatility` was designed in the previous step but not implemented.
*   `DataProcessor` does not yet calculate Average Volume or ATR.

**Detailed Plan:**

1.  **Update `config/config.yaml.example` (Verify/Finalize):**
    *   **Action:** Ensure the sections for `filters.volume` and `filters.volatility` (ATR-based) are correctly defined as per the previous design phase, including `enabled` flags and necessary parameters:
        *   **Volume Filter:**
            *   `enabled: false`
            *   `average_period: 50`
            *   `threshold_ratio: 1.5`
        *   **Volatility Filter (ATR-based):**
            *   `enabled: false`
            *   `atr_period: 14`
            *   `condition_type: "min_atr_percentage_of_price"` (or `"min_atr_absolute_value"`, `"max_atr_percentage_of_price"`)
            *   `value: 0.5` (e.g., 0.5% of price for min_atr_percentage)

2.  **Enhance `DataProcessor` to Calculate Average Volume and ATR:**
    *   **Action:** In `src/data_processor.py`, modify the `_update_indicators` method.
    *   **Action:** Add logic to calculate the rolling average of volume using the `average_period` from the volume filter configuration. Store this as a new column (e.g., `avg_volume`) in the `indicator_data` DataFrame.
    *   **Action:** Add logic to calculate ATR (Average True Range) using the `atr_period` from the volatility filter configuration (or SL/TP ATR settings if those are also being implemented). Store this as a new column (e.g., `atr`) in the `indicator_data` DataFrame.
        *   True Range (TR) = max(High - Low, abs(High - Previous_Close), abs(Low - Previous_Close))
        *   ATR = Simple Moving Average of TR for the ATR period.
    *   **Action:** Ensure these calculations handle edge cases (e.g., insufficient data for the rolling periods, NaNs).
    *   **Action:** Add unit tests in `tests/unit/test_data_processor.py` to verify the correctness of Average Volume and ATR calculations.

3.  **Update `SignalEngineV1._get_pair_specific_config`:**
    *   **Action:** Modify this method in `src/signal_engine.py` to correctly load and merge the configurations for `filters.volume` and `filters.volatility` from global and pair-specific settings.

4.  **Implement Filter Logic in `SignalEngineV1`:**
    *   **Action:** Create new private helper methods:
        *   `_passes_volume_filter(self, latest_kline_data: pd.Series, avg_volume_value: float, volume_filter_config: Dict[str, Any]) -> bool:`
            *   Compares `latest_kline_data['volume']` with `avg_volume_value * volume_filter_config['threshold_ratio']`.
            *   Returns `True` if the volume condition is met, `False` otherwise.
            *   Handles cases where `avg_volume_value` might be NaN or zero.
        *   `_passes_volatility_filter(self, latest_kline_data: pd.Series, atr_value: float, volatility_filter_config: Dict[str, Any]) -> bool:`
            *   Implements logic based on `volatility_filter_config['condition_type']` and `volatility_filter_config['value']`.
            *   Example for `min_atr_percentage_of_price`: `(atr_value / latest_kline_data['close']) * 100 >= volatility_filter_config['value']`.
            *   Returns `True` if the volatility condition is met, `False` otherwise.
            *   Handles cases where `atr_value` might be NaN or zero.
    *   **Action:** In `SignalEngineV1.check_signal()`, integrate calls to these new helper methods into the pipeline, after the core signal and buffer filter checks.
        *   Retrieve `avg_volume` and `atr` from the `latest` series (which comes from the DataFrame provided by `DataProcessor`).
        *   If a filter is enabled in the configuration:
            *   Call the respective helper method.
            *   If it returns `False`, log the failure reason and `return None`.
        *   Log whether each active filter passed or failed.

5.  **Update `TradeSignal` Model (Optional):**
    *   **Action:** Consider adding relevant filter data (e.g., current volume vs. average, current ATR) to the `details` dictionary of the `TradeSignal` object for better traceability of why a signal was generated.

6.  **Update Unit Tests (`tests/unit/test_signal_engine.py`):**
    *   **Action:** Add new test cases for `_passes_volume_filter` and `_passes_volatility_filter` with various inputs to ensure they work correctly.
    *   **Action:** Modify existing tests for `check_signal` to include scenarios where these filters are enabled and either pass or fail, verifying the pipeline behavior. Test with different configuration overrides for these filters.

7.  **Update Documentation (`docs/CONFIGURATION_GUIDE.md`):**
    *   **Action:** Document the new configuration options for `filters.volume` and `filters.volatility`, explaining their parameters and how they work.

**Priority:**
1.  Implement Average Volume calculation in `DataProcessor` and the Volume Filter in `SignalEngineV1`.
2.  Implement ATR calculation in `DataProcessor` and the ATR-based Volatility Filter in `SignalEngineV1`.

**Acceptance Criteria:**
*   `DataProcessor` correctly calculates and provides Average Volume and ATR values.
*   `SignalEngineV1` correctly applies Volume and Volatility filters based on `config.yaml` settings.
*   Signals are appropriately filtered out if they do not meet the criteria of enabled volume/volatility filters.
*   Logs clearly indicate the status and outcome of each applied filter.
*   Unit tests cover the new indicator calculations and filter logic.
*   Configuration guide is updated with the new filter options.