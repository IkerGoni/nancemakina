# Development Plan: Step 006 - Add Unit Tests for `signal_engine.py`

**Task:** Step 006 - Add comprehensive unit tests for the `SignalEngineV1` module (`src/signal_engine.py`), focusing on V1 signal logic and SL/TP calculation.

**Objective:** Ensure the `SignalEngineV1` class correctly identifies SMA crossover conditions, applies filters (like minimum signal interval), calculates Stop Loss (SL) based on pivot points, calculates Take Profit (TP) based on the configured Risk/Reward ratio, and generates accurate `TradeSignal` objects or `None`.

**Target File(s):**
*   `tests/unit/test_signal_engine.py` (Primary file to create/modify)

**Context:**
*   `src/signal_engine.py` exists with the `SignalEngineV1` class implementation.
*   It depends on `ConfigManager` for strategy parameters (SMA periods, R:R ratio, interval filter) and `DataProcessor` for accessing indicator DataFrames.
*   The V1 strategy involves checking the last two data points for SMA crossover.
*   SL calculation involves finding recent pivot lows/highs from the indicator DataFrame.
*   TP is calculated based on SL distance and R:R ratio.
*   `src/models.py` defines the `TradeSignal` object returned.
*   A basic `tests/unit/test_signal_engine.py` file exists but needs comprehensive test cases.

**Detailed Plan:**

1.  **Setup Test Fixtures:**
    *   Create a `pytest` fixture (`mock_config_manager_for_se`) providing mock configurations, including different SMA periods, R:R ratios, and `min_signal_interval_minutes`.
    *   Create a `pytest` fixture (`mock_data_processor_for_se`) using `unittest.mock.MagicMock`. Mock its `get_indicator_dataframe` method to return controlled `pd.DataFrame` objects for specific test scenarios.
    *   Create a `pytest` fixture (`signal_engine_v1`) that initializes `SignalEngineV1` with the mock config manager and data processor.
    *   Use `pytest.mark.asyncio` for all test functions as `check_signal` is async.
    *   Create helper functions (e.g., `create_mock_df`) within the test file to easily generate DataFrames representing specific market conditions (crossovers, no crossovers, specific pivot points).

2.  **Test Signal Generation Logic (`check_signal`):**
    *   **Test Case:** No signal when data is insufficient.
        *   **Action:** Configure `mock_data_processor_for_se.get_indicator_dataframe` to return `None`, an empty DataFrame, or a DataFrame with only one row. Call `check_signal`.
        *   **Assert:** `check_signal` should return `None`.
    *   **Test Case:** No signal when SMA values are NA.
        *   **Action:** Return a DataFrame where the latest or previous SMA values are `pd.NA`. Call `check_signal`.
        *   **Assert:** `check_signal` should return `None`.
    *   **Test Case:** No signal when no crossover occurs (e.g., SMAs diverging, moving parallel).
        *   **Action:** Return a DataFrame where `sma_short > sma_long` in both the previous and current row. Call `check_signal`.
        *   **Action:** Return a DataFrame where `sma_short < sma_long` in both the previous and current row. Call `check_signal`.
        *   **Assert:** `check_signal` should return `None` in both cases.
    *   **Test Case:** Generate a LONG signal on valid bullish crossover.
        *   **Action:** Return a DataFrame representing `prev: short <= long`, `curr: short > long`. Call `check_signal`.
        *   **Assert:** `check_signal` should return a `TradeSignal` object. Verify `signal.direction == TradeDirection.LONG`. Verify `signal.symbol`, `config_symbol`, `contract_type` are correctly populated. Verify `signal.entry_price` matches the `close` of the last row in the DataFrame.
    *   **Test Case:** Generate a SHORT signal on valid bearish crossover.
        *   **Action:** Return a DataFrame representing `prev: short >= long`, `curr: short < long`. Call `check_signal`.
        *   **Assert:** `check_signal` should return a `TradeSignal` object. Verify `signal.direction == TradeDirection.SHORT`. Verify other fields as above.

3.  **Test SL/TP Calculation:**
    *   **Test Case:** Correct SL calculation for LONG signal (pivot low).
        *   **Action:** Use the LONG signal scenario. Ensure the mock DataFrame provided has identifiable recent low points *before* the signal candle.
        *   **Assert:** Verify `signal.stop_loss_price` matches the expected minimum low from the lookback period defined in `_find_recent_pivot` (e.g., 30 candles before the signal candle by default).
    *   **Test Case:** Correct SL calculation for SHORT signal (pivot high).
        *   **Action:** Use the SHORT signal scenario. Ensure the mock DataFrame has identifiable recent high points *before* the signal candle.
        *   **Assert:** Verify `signal.stop_loss_price` matches the expected maximum high from the lookback period.
    *   **Test Case:** No signal if pivot point cannot be determined (e.g., insufficient historical data in DataFrame).
        *   **Action:** Provide a DataFrame that triggers a crossover but is too short for the pivot lookback.
        *   **Assert:** `check_signal` should return `None` (and log a warning).
    *   **Test Case:** Correct TP calculation based on SL and R:R ratio (LONG).
        *   **Action:** Use the LONG signal scenario. Manually calculate `expected_tp = entry + (entry - sl) * rr_ratio`.
        *   **Assert:** `signal.take_profit_price` should be close to `expected_tp` (allow for float precision).
    *   **Test Case:** Correct TP calculation based on SL and R:R ratio (SHORT).
        *   **Action:** Use the SHORT signal scenario. Manually calculate `expected_tp = entry - (sl - entry) * rr_ratio`.
        *   **Assert:** `signal.take_profit_price` should be close to `expected_tp`.
    *   **Test Case:** No signal if calculated SL results in zero or negative risk (e.g., SL >= entry for LONG, SL <= entry for SHORT).
        *   **Action:** Craft DataFrame data where the calculated pivot leads to invalid risk.
        *   **Assert:** `check_signal` should return `None`.
    *   **Test Case:** Verify `signal.signal_kline` content matches the last row of the input DataFrame.
        *   **Action:** Check the `signal.signal_kline` attribute when a valid signal is generated.
        *   **Assert:** The `Kline` object's fields (`timestamp`, `open`, `high`, `low`, `close`, etc.) match the data from the last row of the DataFrame used for signal generation.

4.  **Test Filters:**
    *   **Test Case:** Minimum signal interval filter prevents rapid signals.
        *   **Action:** Configure mock config with a non-zero `min_signal_interval_minutes` (e.g., 15). Call `check_signal` for a symbol to generate a first signal. Immediately call `check_signal` again for the *same symbol* with data that would otherwise generate a signal.
        *   **Assert:** The first call returns a `TradeSignal`. The second call returns `None`.
    *   **Test Case:** Minimum signal interval filter allows signal after interval passes.
        *   **Action:** Generate the first signal. Manually manipulate `signal_engine_v1.last_signal_time[symbol]` using `time.time()` to simulate the interval having passed. Call `check_signal` again.
        *   **Assert:** The second call should now return a `TradeSignal`.
    *   **Test Case:** Minimum signal interval is pair-specific.
        *   **Action:** Use a config where Pair A has interval > 0, Pair B has interval = 0. Generate signal for Pair A. Immediately generate signal for Pair B.
        *   **Assert:** Signal for Pair A prevents immediate repeat. Signal for Pair B is generated successfully right after.

5.  **Test Helper Methods (Optional but Recommended):**
    *   **Test Case:** Direct test for `_get_pair_specific_config`.
        *   **Action:** Call `_get_pair_specific_config` with different `config_symbol` values.
        *   **Assert:** Verify it correctly merges global and pair-specific settings from the mock config.
    *   **Test Case:** Direct test for `_find_recent_pivot`.
        *   **Action:** Create various DataFrames and call `_find_recent_pivot` directly for LONG and SHORT directions with different lookback values.
        *   **Assert:** Verify it correctly identifies the min low / max high within the specified lookback window (excluding the last row). Test edge cases like flat lines or insufficient data.

**Acceptance Criteria:**
*   All test cases described above pass.
*   Test coverage for `src/signal_engine.py` is significantly increased, covering crossover logic, SL/TP calculation, and filters.
*   Tests use mocked dependencies (`ConfigManager`, `DataProcessor`).
*   Tests accurately simulate different market data scenarios using controlled DataFrames.
*   Generated `TradeSignal` objects contain correct data based on inputs.
*   Edge cases for SL/TP calculation and filters are handled correctly.