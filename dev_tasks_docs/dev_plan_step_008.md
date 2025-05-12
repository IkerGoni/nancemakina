# Development Plan: Step 008 - Add Unit Tests for `position_manager.py`

**Task:** Step 008 - Add comprehensive unit tests for the `PositionManager` module (`src/position_manager.py`).

**Objective:** Ensure the `PositionManager` class correctly tracks open positions (adds, updates, removes), provides accurate status information (`get_position`, `get_all_positions`, `has_open_position`), and updates position state correctly based on simulated order update events. Also, test the basic reconciliation logic.

**Target File(s):**
*   `tests/unit/test_position_manager.py` (Primary file to create/modify)

**Context:**
*   `src/position_manager.py` exists with the `PositionManager` class implementation.
*   It stores `Position` objects (defined in `src/models.py`) in a dictionary keyed by symbol.
*   It uses a `threading.RLock` for basic thread safety.
*   The core logic involves CRUD operations on the internal `_positions` dictionary and updating state based on `Order` objects passed to `update_position_on_order_update`.
*   Includes a `reconcile_positions_with_exchange` method requiring a `BinanceRESTClient` instance.
*   A basic `tests/unit/test_position_manager.py` file exists but needs comprehensive test cases.

**Detailed Plan:**

1.  **Setup Test Fixtures:**
    *   Create a `pytest` fixture (`mock_config_manager_for_pm`) - can be a simple mock as `PositionManager` doesn't heavily rely on config in its current form.
    *   Create a `pytest` fixture (`position_manager`) that initializes `PositionManager` with the mock config manager. Ensure each test gets a fresh instance.
    *   Use `pytest.mark.asyncio` for all test functions as methods like `add_or_update_position` and `update_position_on_order_update` are async.

2.  **Test Basic Position Management:**
    *   **Test Case:** Add a new position.
        *   **Action:** Create a sample `Position` object. Call `await position_manager.add_or_update_position(pos_data)`.
        *   **Assert:** `position_manager.get_position(symbol)` returns the added position object. `position_manager.has_open_position(symbol)` returns `True`. `position_manager.get_all_positions()` returns a list containing the position.
    *   **Test Case:** Update an existing position.
        *   **Action:** Add a position. Create a second `Position` object for the same symbol with modified data (e.g., updated `sl_order_id`, `unrealized_pnl`). Call `add_or_update_position` with the updated data.
        *   **Assert:** `position_manager.get_position(symbol)` returns the *updated* position object. `len(position_manager.get_all_positions())` should remain 1.
    *   **Test Case:** Remove an existing position.
        *   **Action:** Add a position. Call `await position_manager.remove_position(symbol)`.
        *   **Assert:** `position_manager.has_open_position(symbol)` returns `False`. `position_manager.get_position(symbol)` returns `None`. `position_manager.get_all_positions()` is empty.
    *   **Test Case:** Remove a non-existent position.
        *   **Action:** Call `remove_position` for a symbol not in the manager.
        *   **Assert:** No error occurs. State remains unchanged.
    *   **Test Case:** Getters return correct values when empty.
        *   **Action:** Initialize a fresh `PositionManager`.
        *   **Assert:** `get_position(symbol)` returns `None`. `get_all_positions()` returns `[]`. `has_open_position(symbol)` returns `False`.

3.  **Test `update_position_on_order_update` Logic:**
    *   **Test Case:** Entry order FILLED.
        *   **Action:** Add a `Position` with `entry_order_id="E1"`. Create an `Order` object for order "E1" with `status=OrderStatus.FILLED` and a specific `avg_fill_price`. Call `await position_manager.update_position_on_order_update(order)`.
        *   **Assert:** The position should still exist. Verify `position.entry_price` is updated to the `avg_fill_price` from the order. Verify `position.quantity` is updated if `filled_quantity` is present in the order.
    *   **Test Case:** Stop Loss order FILLED.
        *   **Action:** Add a `Position` with `sl_order_id="S1"`. Create an `Order` object for order "S1" with `status=OrderStatus.FILLED`. Call `update_position_on_order_update`.
        *   **Assert:** The position for that symbol should be removed (`get_position` returns `None`).
    *   **Test Case:** Take Profit order FILLED.
        *   **Action:** Add a `Position` with `tp_order_id="T1"`. Create an `Order` object for order "T1" with `status=OrderStatus.FILLED`. Call `update_position_on_order_update`.
        *   **Assert:** The position for that symbol should be removed.
    *   **Test Case:** Entry order CANCELED.
        *   **Action:** Add a `Position` with `entry_order_id="E1"`. Create an `Order` object for order "E1" with `status=OrderStatus.CANCELED`. Call `update_position_on_order_update`.
        *   **Assert:** The position for that symbol should be removed.
    *   **Test Case:** SL order CANCELED.
        *   **Action:** Add a `Position` with `sl_order_id="S1"`. Create an `Order` object for order "S1" with `status=OrderStatus.CANCELED`. Call `update_position_on_order_update`.
        *   **Assert:** The position should still exist, but `position.sl_order_id` should be set to `None`.
    *   **Test Case:** TP order CANCELED.
        *   **Action:** Add a `Position` with `tp_order_id="T1"`. Create an `Order` object for order "T1" with `status=OrderStatus.CANCELED`. Call `update_position_on_order_update`.
        *   **Assert:** The position should still exist, but `position.tp_order_id` should be set to `None`.
    *   **Test Case:** Order REJECTED (e.g., SL order).
        *   **Action:** Add a `Position` with `sl_order_id="S1"`. Create an `Order` object for "S1" with `status=OrderStatus.REJECTED`. Call `update_position_on_order_update`.
        *   **Assert:** Position exists, `position.sl_order_id` is `None`.
    *   **Test Case:** Order update for an unrelated order ID.
        *   **Action:** Add a `Position` with specific order IDs. Create an `Order` object with a different `order_id`. Call `update_position_on_order_update`.
        *   **Assert:** The position state should remain unchanged.
    *   **Test Case:** Order update for a symbol with no tracked position.
        *   **Action:** Ensure no position exists for "XYZUSDT". Create an `Order` for "XYZUSDT". Call `update_position_on_order_update`.
        *   **Assert:** No error occurs, internal state remains unchanged.

4.  **Test `reconcile_positions_with_exchange` Logic:**
    *   **Setup:** Create a mock `rest_client` using `unittest.mock.AsyncMock`. Mock its `fetch_positions` method.
    *   **Test Case:** Sync positions from exchange to empty manager.
        *   **Action:** Mock `fetch_positions` to return a list of position dictionaries (simulating ccxt response). Call `await position_manager.reconcile_positions_with_exchange(mock_rest_client)`.
        *   **Assert:** Verify internal `_positions` dict contains `Position` objects corresponding to the data returned by `fetch_positions`. Check key fields like `symbol`, `side`, `quantity`, `entry_price`.
    *   **Test Case:** Remove stale positions from manager not present on exchange.
        *   **Action:** Add a position manually to the manager (e.g., "STALEUSDT"). Mock `fetch_positions` to return an empty list or a list *without* "STALEUSDT". Call `reconcile_positions_with_exchange`.
        *   **Assert:** Verify the "STALEUSDT" position is removed from the manager.
    *   **Test Case:** Update existing positions in manager based on exchange data.
        *   **Action:** Add a position for "BTCUSDT" to the manager with quantity 0.01. Mock `fetch_positions` to return "BTCUSDT" data with quantity 0.05. Call `reconcile_positions_with_exchange`.
        *   **Assert:** Verify the "BTCUSDT" position in the manager now reflects the data from `fetch_positions` (quantity 0.05).
    *   **Test Case:** Handle exchange position with zero quantity.
        *   **Action:** Add a position for "ETHUSDT". Mock `fetch_positions` to return data for "ETHUSDT" but with quantity/contracts = 0. Call `reconcile_positions_with_exchange`.
        *   **Assert:** Verify the "ETHUSDT" position is removed from the manager.
    *   **Test Case:** Handle errors during `fetch_positions`.
        *   **Action:** Mock `fetch_positions` to return `None` or raise an exception. Call `reconcile_positions_with_exchange`.
        *   **Assert:** Verify internal state is unchanged and an error is logged.

5.  **Test Thread Safety (Conceptual):**
    *   **Test Case:** Verify lock object existence and type.
        *   **Assert:** Check `position_manager._lock` exists and is an instance of `threading.RLock`. (Direct testing of locking behavior is hard in unit tests but verifies setup).

**Acceptance Criteria:**
*   All test cases described above pass.
*   Test coverage for `src/position_manager.py` is significantly increased.
*   Tests verify adding, updating, retrieving, and removing positions.
*   Tests confirm correct state changes based on simulated `Order` updates for various statuses (FILLED, CANCELED, REJECTED).
*   Reconciliation logic correctly syncs internal state with mocked exchange data.
*   Basic thread safety setup is confirmed.