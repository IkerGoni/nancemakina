# Development Plan: Step 015a - Finalize Implementation of Advanced Filters (Volume & Volatility/ATR)

**Task:** Complete the implementation of the configurable Volume filter and ATR-based Volatility filter within `SignalEngineV1`, and ensure `DataProcessor` correctly calculates and provides the necessary indicators (Average Volume, ATR).

**Objective:** Make the V1 trading strategy fully operational with optional, configurable volume and volatility confirmation filters to improve signal quality.

**Target File(s):**
*   `src/data_processor.py` (finalize Average Volume and ATR calculation in `_update_indicators`)
*   `src/signal_engine.py` (fully implement logic in `_passes_volume_filter`, `_passes_volatility_filter`, and ensure correct integration in `check_signal` pipeline)
*   `tests/unit/test_data_processor.py` (add/complete tests for Average Volume and ATR calculations)
*   `tests/unit/test_signal_engine.py` (add/complete tests for the new filter logic)
*   `scripts/backtest.py` (for end-to-end testing with these filters enabled)

**Context:**
*   The configuration structure for Volume and Volatility filters is defined in `config.yaml.example`.
*   `SignalEngineV1._get_pair_specific_config` is expected to load these configurations.
*   `SignalEngineV1.check_signal` pipeline has placeholders/stubs for calling these filter methods.
*   `DataProcessor._update_indicators` has conceptual logic for Average Volume and ATR but needs to be fully implemented and tested.

**Detailed Plan:**

1.  **Finalize `DataProcessor._update_indicators` for Average Volume and ATR:**
    *   **Action:** In `src/data_processor.py`, within `_update_indicators`:
        *   Retrieve `filters.volume.average_period` and `filters.volatility.atr_period` from the configuration (via `self.config_manager.get_config()`). It's important that `DataProcessor` uses the *global* settings for these indicator periods, as the indicators are calculated for all data, and `SignalEngine` will decide if/how to use them based on pair-specific filter settings. Alternatively, if indicators should vary per pair, `DataProcessor` would need to be aware of pair-specific indicator parameters, which adds complexity. For now, assume global periods for these base indicators.
        *   Implement the rolling average volume calculation: `df['avg_volume'] = df['volume'].rolling(window=avg_vol_period).mean()`.
        *   Implement the ATR calculation:
            *   `high_low = df['high'] - df['low']`
            *   `high_close_prev = abs(df['high'] - df['close'].shift(1))`
            *   `low_close_prev = abs(df['low'] - df['close'].shift(1))`
            *   `tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)`
            *   `df['atr'] = tr.rolling(window=atr_period).mean()`
        *   Ensure NaNs resulting from `shift(1)` or initial rolling periods are handled (Pandas will naturally produce NaNs, which filter methods in `SignalEngine` must then check for).
    *   **Action:** Add robust unit tests in `tests/unit/test_data_processor.py` to verify that `avg_volume` and `atr` columns are correctly calculated and match manual examples. Test edge cases like insufficient data.

2.  **Complete Filter Helper Methods in `SignalEngineV1`:**
    *   **Action:** In `src/signal_engine.py`, fully implement the logic for:
        *   `_passes_volume_filter(self, latest_kline_data: pd.Series, volume_filter_config: Dict[str, Any]) -> bool:`
            *   Retrieve `current_volume = latest_kline_data.get('volume')` and `avg_volume = latest_kline_data.get('avg_volume')`.
            *   Handle `pd.isna(avg_volume)` or `avg_volume == 0` (perhaps return `True` or `False` based on desired strictness, log a warning).
            *   Compare `current_volume` with `avg_volume * volume_filter_config['threshold_ratio']`.
            *   Log the comparison values and the outcome.
        *   `_passes_volatility_filter(self, latest_kline_data: pd.Series, volatility_filter_config: Dict[str, Any]) -> bool:`
            *   Retrieve `atr = latest_kline_data.get('atr')` and `close_price = latest_kline_data.get('close')`.
            *   Handle `pd.isna(atr)` or `pd.isna(close_price)` or `close_price == 0`.
            *   Implement logic based on `volatility_filter_config['condition_type']` and `value`.
            *   Log comparison values and outcome.

3.  **Ensure Correct Integration in `SignalEngineV1.check_signal()`:**
    *   **Action:** Verify that the calls to `_passes_volume_filter` and `_passes_volatility_filter` in the `check_signal` pipeline are correctly placed.
    *   **Action:** Ensure that the `latest_kline_data` Series passed to these filter methods actually contains the `avg_volume` and `atr` values (i.e., `DataProcessor` has successfully added them). This implies checking if `latest_kline_data.get('avg_volume')` is not `None` before using it.

4.  **Update Unit Tests for `SignalEngineV1`:**
    *   **Action:** In `tests/unit/test_signal_engine.py`:
        *   Write specific unit tests for `_passes_volume_filter` and `_passes_volatility_filter`. Mock the `latest_kline_data` Series with various `volume`, `avg_volume`, `atr`, and `close` values, and test with different filter configurations (`enabled`, `threshold_ratio`, `condition_type`, `value`).
        *   Expand tests for `check_signal` to include scenarios where Volume and Volatility filters are enabled.
            *   Test cases where signals pass these filters.
            *   Test cases where signals are rejected by these filters.
            *   Test with missing `avg_volume` or `atr` data in the input DataFrame to ensure graceful handling.

5.  **End-to-End Testing with Backtester:**
    *   **Action:** Use `scripts/backtest.py` with various settings in `config.yaml` to enable/disable the Volume and Volatility filters and adjust their parameters.
    *   **Action:** Observe the log output to confirm:
        *   `DataProcessor` logs show `avg_volume` and `atr` being calculated.
        *   `SignalEngineV1` logs show the filter checks being performed when enabled.
        *   Signals are correctly passed or rejected based on these filter conditions.
    *   **Action:** Analyze the impact of these filters on the backtest performance metrics (number of trades, win rate, P&L).

**Acceptance Criteria:**
*   `DataProcessor` accurately calculates and makes `avg_volume` and `atr` available in its output DataFrames.
*   The Volume filter logic in `SignalEngineV1` correctly compares current volume to average volume against the configured threshold.
*   The ATR-based Volatility filter logic in `SignalEngineV1` correctly evaluates conditions based on ATR, price, and configured type/value.
*   Both filters can be independently enabled/disabled via `config.yaml` and their parameters adjusted.
*   Signals are correctly filtered by these new mechanisms when they are active.
*   Unit tests provide good coverage for the new indicator calculations in `DataProcessor` and the filter logic in `SignalEngineV1`.
*   Backtesting demonstrates the functional application of these filters.