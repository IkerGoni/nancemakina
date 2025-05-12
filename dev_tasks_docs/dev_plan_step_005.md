# Development Plan: Step 005 - Add Unit Tests for `data_processor.py`

**Task:** Step 005 - Add comprehensive unit tests for the `DataProcessor` module (`src/data_processor.py`).

**Objective:** Ensure the `DataProcessor` class correctly initializes based on configuration, processes incoming Kline data, manages internal buffers (deques and DataFrames), calculates SMA indicators accurately, handles edge cases, and responds to configuration updates.

**Target File(s):**
*   `tests/unit/test_data_processor.py` (Primary file to create/modify)

**Context:**
*   `src/data_processor.py` exists with the `DataProcessor` class implementation.
*   It uses `pandas` for DataFrames and indicator calculations (SMA).
*   It uses `collections.deque` for Kline buffering.
*   It registers a callback with `ConfigManager` for hot-reloading.
*   A basic `tests/unit/test_data_processor.py` file exists but needs comprehensive test cases.

**Detailed Plan:**

1.  **Setup Test Fixtures:**
    *   Create a `pytest` fixture (`mock_config_manager_for_dp`) using `unittest.mock.MagicMock` or a helper class to simulate `ConfigManager`. This fixture should provide different mock configurations for various test scenarios (e.g., different enabled pairs, timeframes, SMA periods).
    *   Create a `pytest` fixture (`data_processor`) that initializes `DataProcessor` with the mock config manager. Ensure each test gets a fresh instance.
    *   Use `pytest.mark.asyncio` for all test functions as `process_kline` is async.

2.  **Test Initialization (`__init__` & `_initialize_buffers_from_config`):**
    *   **Test Case:** Verify buffer creation based on `enabled` pairs and `indicator_timeframes` in config.
        *   **Action:** Initialize `DataProcessor` with a mock config having a mix of enabled/disabled pairs and global/specific timeframes.
        *   **Assert:** Check `data_processor._active_pairs_timeframes` dict matches the expected enabled pairs and their associated timeframes. Assert that `data_processor.kline_buffers[symbol][timeframe]` exists and is a `deque` for all active combinations. Assert that buffers for *disabled* pairs are *not* present or are empty. Assert deque `maxlen` is `MAX_KLINE_BUFFER_LENGTH`.
    *   **Test Case:** Verify initialization with empty `pairs` configuration.
        *   **Action:** Initialize `DataProcessor` with a mock config where `pairs` is empty or missing.
        *   **Assert:** `_active_pairs_timeframes`, `kline_buffers`, and `indicator_data` should be empty.

3.  **Test Kline Processing (`process_kline`):**
    *   **Test Case:** Process a single *closed* Kline for an active pair/timeframe.
        *   **Action:** Call `await data_processor.process_kline(kline_closed)` with a sample `Kline` object.
        *   **Assert:** Check the kline is appended to the correct `kline_buffers[symbol][timeframe]` deque. Verify `_update_indicators` was implicitly called (e.g., by checking if `indicator_data` DataFrame is updated).
    *   **Test Case:** Process a single *unclosed* Kline for an active pair/timeframe.
        *   **Action:** Call `await data_processor.process_kline(kline_unclosed)`.
        *   **Assert:** Check the kline is appended. Verify `_update_indicators` might *not* have been called yet (depending on implementation, but typically only runs on closed candles). Check `indicator_data` state.
    *   **Test Case:** Update the last *unclosed* Kline.
        *   **Action:** Process `kline_unclosed_t1`. Process `kline_unclosed_t1_update` (same timestamp, different data).
        *   **Assert:** The deque should still contain only one element, and it should be `kline_unclosed_t1_update`.
    *   **Test Case:** Replace the last *unclosed* Kline with its *closed* version.
        *   **Action:** Process `kline_unclosed_t1`. Process `kline_closed_t1` (same timestamp).
        *   **Assert:** The deque should contain one element (`kline_closed_t1`). Verify `_update_indicators` was called.
    *   **Test Case:** Process Kline for an *inactive* pair/timeframe.
        *   **Action:** Create a Kline for a pair/timeframe *not* enabled in the mock config. Call `process_kline`.
        *   **Assert:** Verify the kline was *not* added to any buffer and `indicator_data` remains unchanged for that symbol/tf.
    *   **Test Case:** Buffer limit enforcement.
        *   **Action:** Process `MAX_KLINE_BUFFER_LENGTH + 5` klines.
        *   **Assert:** The length of the deque should be exactly `MAX_KLINE_BUFFER_LENGTH`. Verify the oldest klines were discarded.

4.  **Test Indicator Calculation (`_update_indicators`):**
    *   **Test Case:** Insufficient data for SMAs.
        *   **Action:** Process fewer klines than `sma_short_period`. Call `_update_indicators` directly (or process a closed kline).
        *   **Assert:** The resulting DataFrame in `indicator_data` should have `sma_short` and `sma_long` columns filled with `pd.NA` or appropriate null values.
    *   **Test Case:** Correct SMA calculation (short period).
        *   **Action:** Process exactly `sma_short_period` klines. Process one more *closed* kline.
        *   **Assert:** Fetch the DataFrame using `get_indicator_dataframe`. Verify the `sma_short` value for the last row matches the manually calculated SMA. `sma_long` should still be NA.
    *   **Test Case:** Correct SMA calculation (long period).
        *   **Action:** Process exactly `sma_long_period` klines. Process one more *closed* kline.
        *   **Assert:** Fetch the DataFrame. Verify both `sma_short` and `sma_long` values for the last row match manually calculated SMAs.
    *   **Test Case:** SMA calculation with duplicate timestamps (ensure last update is used).
        *   **Action:** Process klines including updates with the same timestamp.
        *   **Assert:** Verify the DataFrame used for SMA calculation does not contain duplicate indices and uses the final update for that timestamp. Check SMA correctness.

5.  **Test Getter Methods (`get_latest_kline`, `get_indicator_dataframe`, `get_latest_indicators`):**
    *   **Test Case:** Getters return correct data after processing.
        *   **Action:** Process some klines.
        *   **Assert:** `get_latest_kline` returns the last processed Kline object. `get_indicator_dataframe` returns a DataFrame copy. `get_latest_indicators` returns the last row (Series) of the indicator DataFrame.
    *   **Test Case:** Getters return `None` or empty DataFrame for non-existent symbol/timeframe.
        *   **Action:** Call getters with invalid symbols/timeframes.
        *   **Assert:** Verify `None` or empty `pd.DataFrame` is returned as appropriate.

6.  **Test Configuration Hot-Reload (`_handle_config_update`):**
    *   **Test Case:** Adding a new active pair via config update.
        *   **Action:** Initialize `DataProcessor`. Simulate a config update callback (`_handle_config_update`) with a new config dictionary where a previously disabled pair is now enabled.
        *   **Assert:** Check `_active_pairs_timeframes` now includes the new pair. Verify that corresponding buffers have been created in `kline_buffers` and `indicator_data`.
    *   **Test Case:** Disabling an existing active pair via config update.
        *   **Action:** Initialize `DataProcessor`. Simulate a config update callback disabling an active pair.
        *   **Assert:** Check `_active_pairs_timeframes` no longer includes the disabled pair. (Note: The current implementation might not *remove* old buffers, just stop processing for them. Assert behavior based on implementation).
    *   **Test Case:** Changing `indicator_timeframes` for a pair.
        *   **Action:** Initialize `DataProcessor`. Simulate a config update changing the timeframes list for an active pair.
        *   **Assert:** Check `_active_pairs_timeframes` reflects the new timeframes. Verify necessary buffers exist/are managed.

**Acceptance Criteria:**
*   All test cases described above pass.
*   Test coverage for `src/data_processor.py` is significantly increased.
*   Tests use mocked dependencies (`ConfigManager`) and do not rely on external systems.
*   Tests adhere to PEP 8, use type hints, and follow async patterns correctly.
*   Edge cases like empty data, NA values, and buffer limits are handled.