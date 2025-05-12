# Development Plan: Step 007 - Add Unit Tests for `order_manager.py`

**Task:** Step 007 - Add comprehensive unit tests for the `OrderManager` module (`src/order_manager.py`).

**Objective:** Ensure the `OrderManager` class correctly receives `TradeSignal` objects, calculates position size based on configuration (fixed margin MVP), fetches and applies exchange filters (precision, min notional), places the correct entry, Stop Loss (SL), and Take Profit (TP) orders via the `BinanceRESTClient`, and handles specifics for USDT-M and COIN-M contracts.

**Target File(s):**
*   `tests/unit/test_order_manager.py` (Primary file to create/modify)

**Context:**
*   `src/order_manager.py` exists with the `OrderManager` class implementation.
*   It depends on `ConfigManager` for parameters (leverage, fixed margin amounts) and `BinanceRESTClient` for executing orders and fetching exchange info.
*   MVP focuses on fixed margin sizing (USDT amount for USDT-M, potentially number of contracts for COIN-M as per current simple implementation).
*   Places MARKET entry, STOP_MARKET SL, and TAKE_PROFIT_MARKET TP orders.
*   Includes logic for adjusting quantity and price to meet exchange precision rules.
*   Includes a basic min_notional value check.
*   A basic `tests/unit/test_order_manager.py` file exists but needs comprehensive test cases.

**Detailed Plan:**

1.  **Setup Test Fixtures:**
    *   Create a `pytest` fixture (`mock_config_manager_for_om`) providing mock configurations, including USDT-M and COIN-M pairs with different leverage and fixed margin settings.
    *   Create a `pytest` fixture (`mock_rest_client_for_om`) using `unittest.mock.AsyncMock`.
        *   Mock `rest_client.fetch_exchange_info` to return realistic market data including `precision` (amount, price) and `limits` (cost/min_notional) for test symbols (e.g., BTCUSDT, ETHUSD_PERP).
        *   Mock `rest_client.create_order` to simulate successful order placement, returning a mock order dictionary with an ID and status. It should also capture the arguments it was called with for verification.
        *   Mock other methods if needed (e.g., `set_leverage`, `set_margin_mode`), although the current `handle_trade_signal` assumes these are pre-set.
    *   Create a `pytest` fixture (`order_manager`) that initializes `OrderManager` with the mock config manager and rest client. Prime the exchange info cache within the fixture using `await order_manager._get_exchange_info_for_symbol(symbol)`.
    *   Use `pytest.mark.asyncio` for all test functions as `handle_trade_signal` and underlying REST calls are async.

2.  **Test Precision and Filter Logic:**
    *   **Test Case:** `_adjust_quantity_to_precision`.
        *   **Action:** Call `order_manager._adjust_quantity_to_precision` directly with various quantities and step sizes (e.g., 0.001, 0.1, 1).
        *   **Assert:** Verify the returned quantity is correctly floored to the specified step size precision.
    *   **Test Case:** `_adjust_price_to_precision`.
        *   **Action:** Call `order_manager._adjust_price_to_precision` directly with various prices and tick sizes (e.g., 0.01, 0.5, 10).
        *   **Assert:** Verify the returned price is correctly rounded to the specified tick size precision.
    *   **Test Case:** `_get_symbol_filters` retrieves correct filters.
        *   **Action:** Call `await order_manager._get_symbol_filters(symbol)`.
        *   **Assert:** Verify it returns the correct tuple `(lot_step_size, price_tick_size, min_notional)` based on the mocked `fetch_exchange_info` data. Test that the cache prevents repeated `fetch_exchange_info` calls.
    *   **Test Case:** `_get_symbol_filters` handles missing filter data gracefully.
        *   **Action:** Mock `fetch_exchange_info` to return data missing certain precision/limits fields. Call `_get_symbol_filters`.
        *   **Assert:** Verify it returns `None` for the missing values and logs appropriate warnings.

3.  **Test Position Sizing (Fixed Margin MVP):**
    *   **Test Case:** Correct quantity calculation for USDT-M pair.
        *   **Action:** Create a `TradeSignal` for a configured USDT-M pair. Call `await order_manager.handle_trade_signal(signal)`.
        *   **Assert:** Manually calculate the expected quantity: `(margin_usdt * leverage) / entry_price`, adjusted to the mocked `lot_step_size`. Verify the `amount` passed to `mock_rest_client.create_order` for the entry order matches this expected quantity.
    *   **Test Case:** Correct quantity calculation for COIN-M pair (MVP assumption: `margin_coin` is number of contracts).
        *   **Action:** Create a `TradeSignal` for a configured COIN-M pair where `margin_coin` is set. Call `await order_manager.handle_trade_signal(signal)`.
        *   **Assert:** Verify the `amount` passed to `create_order` matches the `margin_coin` value from the config, adjusted to the mocked `lot_step_size`. Add a note acknowledging this is a simplification.
    *   **Test Case:** Handling missing margin configuration.
        *   **Action:** Create a signal for a pair where `margin_usdt` (for USDT-M) or `margin_coin` (for COIN-M) is missing in both pair-specific and global config. Call `handle_trade_signal`.
        *   **Assert:** Verify that `create_order` is *not* called and an error is logged.
    *   **Test Case:** Handling zero or invalid entry price.
        *   **Action:** Create a signal with `entry_price=0`. Call `handle_trade_signal`.
        *   **Assert:** Verify `create_order` is not called.

4.  **Test Order Placement Logic:**
    *   **Test Case:** Correct parameters for MARKET entry order (LONG).
        *   **Action:** Handle a LONG `TradeSignal`. Inspect the first call to `mock_rest_client.create_order`.
        *   **Assert:** Verify `symbol`, `order_type=OrderType.MARKET`, `side=OrderSide.BUY`, and calculated `amount` are correct.
    *   **Test Case:** Correct parameters for MARKET entry order (SHORT).
        *   **Action:** Handle a SHORT `TradeSignal`. Inspect the first call.
        *   **Assert:** Verify `symbol`, `order_type=OrderType.MARKET`, `side=OrderSide.SELL`, and calculated `amount` are correct.
    *   **Test Case:** Correct parameters for STOP_MARKET SL order (for LONG entry).
        *   **Action:** Handle a LONG `TradeSignal`. Inspect the second call.
        *   **Assert:** Verify `symbol`, `order_type=OrderType.STOP_MARKET`, `side=OrderSide.SELL`, `amount` (same as entry), `params['stopPrice']` (adjusted SL price), and `params['reduceOnly'] == 'true'`.
    *   **Test Case:** Correct parameters for STOP_MARKET SL order (for SHORT entry).
        *   **Action:** Handle a SHORT `TradeSignal`. Inspect the second call.
        *   **Assert:** Verify `symbol`, `order_type=OrderType.STOP_MARKET`, `side=OrderSide.BUY`, `amount`, `params['stopPrice']` (adjusted SL price), and `params['reduceOnly'] == 'true'`.
    *   **Test Case:** Correct parameters for TAKE_PROFIT_MARKET TP order (for LONG entry).
        *   **Action:** Handle a LONG `TradeSignal`. Inspect the third call.
        *   **Assert:** Verify `symbol`, `order_type=OrderType.TAKE_PROFIT_MARKET`, `side=OrderSide.SELL`, `amount`, `params['stopPrice']` (adjusted TP price), and `params['reduceOnly'] == 'true'`.
    *   **Test Case:** Correct parameters for TAKE_PROFIT_MARKET TP order (for SHORT entry).
        *   **Action:** Handle a SHORT `TradeSignal`. Inspect the third call.
        *   **Assert:** Verify `symbol`, `order_type=OrderType.TAKE_PROFIT_MARKET`, `side=OrderSide.BUY`, `amount`, `params['stopPrice']` (adjusted TP price), and `params['reduceOnly'] == 'true'`.
    *   **Test Case:** Min Notional check prevents order placement.
        *   **Action:** Configure mock exchange info with a `min_notional` value higher than the calculated notional value (`quantity * entry_price` for USDT-M). Handle a relevant `TradeSignal`.
        *   **Assert:** Verify `mock_rest_client.create_order` is *not* called and a warning is logged.
    *   **Test Case:** Calculated quantity is zero or negative.
        *   **Action:** Contrive a scenario (e.g., via precision adjustment or bad input signal) where the final calculated quantity is <= 0. Call `handle_trade_signal`.
        *   **Assert:** Verify `create_order` is not called.

5.  **Test Error Handling:**
    *   **Test Case:** Failure to fetch exchange info.
        *   **Action:** Mock `rest_client.fetch_exchange_info` to return `None` or raise an exception. Clear the cache (`order_manager.exchange_info_cache = {}`) Call `handle_trade_signal`.
        *   **Assert:** Verify `create_order` is not called and an error is logged.
    *   **Test Case:** Entry order placement fails.
        *   **Action:** Mock `rest_client.create_order` for the first call to return `None` or an error response (e.g., `{"status": "REJECTED"}`). Handle a `TradeSignal`.
        *   **Assert:** Verify only the first call to `create_order` is made. Verify subsequent SL/TP orders are *not* placed. Verify an error is logged.
    *   **Test Case:** SL or TP order placement fails.
        *   **Action:** Mock `rest_client.create_order` to succeed for entry but fail for SL (return `None`). Handle a `TradeSignal`.
        *   **Assert:** Verify entry and SL calls are made. Verify TP call is still attempted (or not, depending on desired logic - current code attempts TP even if SL fails). Verify an error is logged regarding the failed SL order.

**Acceptance Criteria:**
*   All test cases described above pass.
*   Test coverage for `src/order_manager.py` is significantly increased, covering sizing, precision adjustment, order parameters, and basic error handling.
*   Tests use mocked dependencies (`ConfigManager`, `BinanceRESTClient`).
*   Order parameters (`symbol`, `type`, `side`, `amount`, `price`, `stopPrice`, `reduceOnly`) are validated for entry, SL, and TP orders in different scenarios.
*   Fixed margin position sizing logic for USDT-M and the simplified COIN-M approach are tested.
*   Precision adjustments and min_notional checks are verified.