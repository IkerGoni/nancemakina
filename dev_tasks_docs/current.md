# Development Plan: Step 017 - Refactor SignalEngineV1 for DataFrame Input in Backtesting

**Task:** Modify `SignalEngineV1` to accept a pandas DataFrame directly for signal checking, allowing the backtester to pass point-in-time historical data slices.

**Objective:** Enable accurate backtesting by ensuring `SignalEngineV1` makes decisions based on the complete historical data available up to each specific candle being evaluated, rather than being limited by `DataProcessor`'s live-trading buffer (`MAX_KLINE_BUFFER_LENGTH`).

**Target File(s):**
*   `src/signal_engine.py` (Modify existing `check_signal` or add new `check_signal_with_df` method)
*   `scripts/backtest.py` (Update `generate_signals_iteratively` or equivalent to use the new/modified `SignalEngineV1` method)
*   `tests/unit/test_signal_engine.py` (Update tests to reflect changes)

**Context:**
*   The current `SignalEngineV1.check_signal` relies on its internal `self.data_processor` instance to get indicator DataFrames.
*   `DataProcessor`'s internal buffers are capped by `MAX_KLINE_BUFFER_LENGTH` (e.g., 300 candles), which is suitable for live trading memory management but limits the historical context seen by `SignalEngineV1` during iterative backtesting over long datasets.
*   This limitation causes the backtester to effectively only evaluate signals based on a rolling window of the most recent 300 candles of the *entire dataset processed so far by DataProcessor*, not the full history up to each specific point in time being backtested for signal generation.
*   The result is potentially very few signals generated, or signals based on an incomplete historical view.

**Detailed Plan:**

1.  **Refactor/Add Method in `SignalEngineV1` for DataFrame Input:**
    *   **Action:** In `src/signal_engine.py`, implement the following strategy:
        *   Create a new public method, e.g., `async def check_signal_with_df(self, api_symbol: str, config_symbol: str, point_in_time_df: pd.DataFrame) -> Optional[TradeSignal]:`.
        *   Move the *entire core logic* from the current `async def check_signal(...)` method into this new `check_signal_with_df(...)` method.
        *   This new method will perform all its operations (getting `latest`, `previous`, calling `_find_recent_pivot`, etc.) directly on the `point_in_time_df` passed as an argument, instead of calling `self.data_processor.get_indicator_dataframe(...)`.
        *   Modify the original `async def check_signal(self, api_symbol: str, config_symbol: str) -> Optional[TradeSignal]:` to become a wrapper. It will now:
            1.  Call `self.data_processor.get_indicator_dataframe(api_symbol, "1m")` to get the current DataFrame (as it does for live trading).
            2.  Call `await self.check_signal_with_df(api_symbol, config_symbol, df_from_processor)`.
    *   **Rationale:** This approach keeps the original `check_signal` interface for live trading (which relies on the live `DataProcessor` instance) while providing a dedicated, testable interface for the backtester to inject complete point-in-time historical data.

2.  **Update `DataProcessor` for Full History DataFrame Generation (for Backtester Use):**
    *   **Action:** In `scripts/backtest.py` (or a helper utility if preferred), ensure there's a mechanism to create a `DataProcessor` instance *specifically for generating the full historical DataFrame with all indicators*.
    *   This mechanism should:
        1.  Instantiate `DataProcessor`.
        2.  Temporarily set `MAX_KLINE_BUFFER_LENGTH` to a very large number or make the kline deques unbounded for the duration of processing the full historical dataset *for this specific DataFrame generation task only*. (e.g., `temp_dp.kline_buffers[symbol][interval] = deque()` before processing all klines, then restore original maxlen if that DP instance were to be reused, though for this task it's usually a one-off).
        3.  Iterate through all loaded `Kline` objects from the dataset, calling `await temp_dp.process_kline(kline_obj)` for each.
        4.  After processing all klines, retrieve the complete DataFrame using `temp_dp.get_indicator_dataframe(symbol, "1m")`. This DataFrame will contain all historical klines and their calculated indicators.

3.  **Modify `scripts/backtest.py` to Use `check_signal_with_df`:**
    *   **Action:** In the `generate_signals_iteratively` (or a similarly named method like `generate_signals_from_full_df`) method:
        1.  It should first obtain the `full_historical_indicator_df` as described in step 2 above.
        2.  Then, it will iterate from `warmup_period` to `len(full_historical_indicator_df)`.
        3.  In each iteration `i`, it will create a slice: `point_in_time_df_slice = full_historical_indicator_df.iloc[:i+1]`. This slice represents all data available up to and including the current candle `i`.
        4.  It will then call `await self.signal_engine.check_signal_with_df(api_symbol, config_symbol, point_in_time_df_slice)`.
    *   **Action:** The `simulate_trading` method will continue to use the `full_historical_indicator_df` (the one containing all data) to check for SL/TP hits against market highs/lows.

4.  **Update Unit Tests for `SignalEngineV1`:**
    *   **Action:** Modify tests in `tests/unit/test_signal_engine.py`.
    *   Existing tests for `check_signal` might now indirectly test `check_signal_with_df` if they mock `data_processor.get_indicator_dataframe`.
    *   Add new tests that directly call `check_signal_with_df` with various handcrafted `point_in_time_df` DataFrames to test specific scenarios (e.g., just enough data for SMAs, data that should trigger pivots, data with active filters).

**Acceptance Criteria:**
*   `SignalEngineV1` has a method (e.g., `check_signal_with_df`) that accepts a pandas DataFrame as input for its signal logic.
*   The original `SignalEngineV1.check_signal` method now uses this new method, fetching its DataFrame from `self.data_processor`.
*   `scripts/backtest.py` generates a complete historical DataFrame with all indicators for the entire dataset.
*   The backtester's signal generation loop iterates through this complete DataFrame, passing incrementally larger slices (point-in-time history) to `SignalEngineV1`'s new DataFrame-accepting method.
*   Backtests run over larger datasets now generate signals based on the true historical context at each candle, rather than a limited rolling window.
*   The number of signals generated by the backtester is more consistent with expectations for the strategy over the full dataset.