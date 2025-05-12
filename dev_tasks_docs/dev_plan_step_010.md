# Development Plan: Step 010 - Add Unit Tests for `monitoring.py`

**Task:** Step 010 - Add unit tests for the basic Prometheus metrics exposure in `src/monitoring.py`.

**Objective:** Ensure the `PrometheusMonitor` class correctly initializes Prometheus metrics (Counters, Gauges, Enums, Info), updates their values via helper methods, and handles the start/stop logic conceptually (actual server start/stop is hard to unit test cleanly).

**Target File(s):**
*   `tests/unit/test_monitoring.py` (Primary file to create/modify)

**Context:**
*   `src/monitoring.py` defines the `PrometheusMonitor` class using the `prometheus_client` library.
*   It defines various metrics types (Counter, Gauge, Enum, Info).
*   It provides helper methods (e.g., `inc_bot_error`, `set_websocket_status`) to update metric values with labels.
*   The `start` method uses `prometheus_client.start_http_server`.
*   An async task `_update_uptime_periodically` is used for the uptime gauge.
*   Stopping the server relies on process exit as `prometheus_client` doesn't offer a clean stop function.

**Detailed Plan:**

1.  **Setup Test Fixtures:**
    *   Create a `pytest` fixture (`monitor`) that initializes `PrometheusMonitor` with a test port.
    *   Use `pytest.mark.asyncio` as `_update_uptime_periodically` is async.
    *   **Mocking `start_http_server`:** Since we don't want to start a real HTTP server during unit tests, patch `prometheus_client.start_http_server`.
        *   `@patch('src.monitoring.start_http_server')`

2.  **Test Initialization:**
    *   **Test Case:** Verify metric objects are created correctly.
        *   **Action:** Initialize `PrometheusMonitor`.
        *   **Assert:** Check that attributes like `monitor.bot_uptime_seconds`, `monitor.bot_errors_total`, `monitor.websocket_connection_status`, etc., exist and are instances of the correct `prometheus_client` metric types (Gauge, Counter, Enum).

3.  **Test Metric Update Methods:**
    *   **Test Case:** Update Counter metrics (`inc_` methods).
        *   **Action:** Call methods like `monitor.inc_bot_error("moduleA", "type1")`, `monitor.inc_kline_processed("BTCUSDT", "1m")`, etc. Call them multiple times with the same and different labels.
        *   **Assert:** Use `prometheus_client.REGISTRY.get_sample_value` to retrieve the current value of the counter for specific label combinations and verify they increment as expected.
            *   Example: `assert REGISTRY.get_sample_value('trading_bot_errors_total', {'module': 'moduleA', 'error_type': 'type1'}) == 1`
    *   **Test Case:** Update Gauge metrics (`set_` methods).
        *   **Action:** Call methods like `monitor.set_indicator_calculation_duration("ETHUSDT", "5m", 0.123)`, `monitor.set_active_positions_count("USDT_M", 5)`. Call again with different values.
        *   **Assert:** Use `REGISTRY.get_sample_value` to verify the gauge reflects the *last* set value for the given labels.
    *   **Test Case:** Update Enum metrics (`set_websocket_status`).
        *   **Action:** Call `monitor.set_websocket_status("USDT_M", "connected")`, then `monitor.set_websocket_status("USDT_M", "disconnected")`. Call with an invalid state.
        *   **Assert:** Use `REGISTRY.get_sample_value` for the Enum metric. Enums expose multiple gauges (`<metric_name>{<label>=<value>} == 1`). Verify the gauge corresponding to the *last valid state* is 1 and others are 0. Verify that setting an invalid state logs a warning and potentially sets an 'error' state if implemented.
            *   Example: `assert REGISTRY.get_sample_value('trading_bot_websocket_connection_status', {'market_type': 'USDT_M', 'trading_bot_websocket_connection_status': 'disconnected'}) == 1`
            *   `assert REGISTRY.get_sample_value('trading_bot_websocket_connection_status', {'market_type': 'USDT_M', 'trading_bot_websocket_connection_status': 'connected'}) == 0`
    *   **Test Case:** Update Info metric (`bot_info`).
        *   **Action:** Call `monitor.start()` (with `start_http_server` mocked). The `info` metric is set within `start`.
        *   **Assert:** Use `REGISTRY.get_sample_value('trading_bot_info_info')` to check the labels set (version, name).

4.  **Test Start/Stop Logic (Conceptual):**
    *   **Test Case:** `start` calls `start_http_server` and sets info.
        *   **Action:** Call `monitor.start()` with `start_http_server` patched.
        *   **Assert:** Verify the mocked `start_http_server` was called once with the correct port. Verify the `bot_info` metric was populated. Verify `monitor._server_started` is `True`.
    *   **Test Case:** `start` handles `OSError` if server start fails.
        *   **Action:** Configure the patched `start_http_server` to raise `OSError`. Call `monitor.start()`.
        *   **Assert:** Verify an error is logged and `monitor._server_started` remains `False`.
    *   **Test Case:** `_update_uptime_periodically` updates the uptime gauge.
        *   **Action:** Call `monitor.start()`. Wait briefly using `asyncio.sleep`.
        *   **Assert:** Get the value of `trading_bot_uptime_seconds` using `REGISTRY.get_sample_value`. Verify it's a small positive number. Wait again, verify the value increases. (This requires careful handling of the async task lifecycle in tests, potentially mocking `asyncio.create_task` or running the loop briefly). *Alternative:* Test the update logic directly by calling the method once manually.
    *   **Test Case:** `stop` method (conceptual check).
        *   **Action:** Call `monitor.start()`, then `monitor.stop()`.
        *   **Assert:** Verify `monitor._server_started` is set to `False`. Verify a log message indicates shutdown. (Actual server stop cannot be tested here).

5.  **Cleanup:**
    *   **Action:** Ensure Prometheus registry is cleared between tests if necessary, although `get_sample_value` usually works on the current state. Using separate processes or carefully managing the global registry might be needed if tests interfere. For basic unit tests, checking values immediately after setting them should be sufficient. *Consider using `prometheus_client.REGISTRY.clear()` in test setup/teardown if interference occurs.*

**Acceptance Criteria:**
*   All test cases described above pass.
*   Test coverage for `src/monitoring.py` is significantly increased.
*   Tests verify the correct initialization and update of different Prometheus metric types (Counter, Gauge, Enum, Info) via the helper methods.
*   Labels are correctly applied and verified in assertions.
*   Conceptual testing of `start` and `stop` logic confirms expected behavior (mocked server start, flag setting).
*   Tests use mocked dependencies (`start_http_server`).