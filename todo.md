# Binance Futures Trading Bot - Development Checklist

## Phase 1: Core Infrastructure & MVP (V1 Strategy)

- [ ] **Step 001: Initialize Project Repository and Directory Structure**
    - [x] Create main project directory `binance_futures_bot`.
    - [x] Create subdirectories: `.github/workflows`, `config`, `data`, `docs`, `src`, `tests/unit`, `tests/integration`, `tests/e2e`.
    - [x] Create initial empty files: `src/__init__.py`, `src/main.py`, `src/config_loader.py`, `src/connectors.py`, `src/data_processor.py`, `src/signal_engine.py`, `src/order_manager.py`, `src/position_manager.py`, `src/models.py`, `src/utils.py`, `src/monitoring.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `tests/e2e/__init__.py`, `Dockerfile`, `README.md`, `.gitignore`, `run.sh`, `config/config.yaml.example`, `docs/ARCHITECTURE.md`, `docs/CONFIGURATION_GUIDE.md`, `.github/workflows/ci.yml`.
    - [x] Create initial `requirements.txt` with basic dependencies.
    - [x] Create initial `.gitignore`.
- [x] **Step 002: Implement Core Configuration Management Module (`src/config_loader.py`)**
    - [x] Define `config.yaml` structure (API keys, global V1 params, pair-specific settings, logging, monitoring).
    - [x] Implement `ConfigManager` class: load, validate, provide access to configuration.
    - [x] Implement hot-reloading of `config.yaml` using `watchdog`.
    - [x] Add unit tests for `ConfigManager`.
- [x] **Step 003: Implement Data Models (`src/models.py`)**
    - [x] Define Pydantic or dataclass models for `Kline`, `TradeSignal`, `Order`, `Position`, `PairConfig`.
    - [x] Add unit tests for data models.
- [x] **Step 004: Develop Binance WebSocket and REST Connectors (`src/connectors.py`)**
    - [x] Implement `BinanceWebSocketConnector`: connect, subscribe (1m, 5m, 15m, 1h, 4h), handle messages, reconnect logic for USDⓈ-M and COIN-M.
    - [x] Implement `BinanceRESTClient` (using `ccxt`): fetch exchange info, balance, place/cancel orders, fetch open orders/positions, set margin/leverage for USDⓈ-M and COIN-M.
    - [x] Add unit tests for connectors (mocking external calls)- [x] **Step 005: Implement Data Processing and Indicator Calculation (`src/data_processor.py`)**
    - [x] Receive k-line data from WebSocket.
    - [x] Store k-lines in buffers (per pair, per timeframe).
    - [x] Calculate SMAs (21, 200) for all configured timeframes (1m, 5m, 15m, 1h, 4h) using `pandas`.
    - [ ] Add unit tests for data processing and SMA calculation.dd unit tests for data processing and SMA calculation.
- [x] **Step 006: Develop Signal Engine V1 (`src/signal_engine.py`)**
    - [x] Implement `SignalEngineV1` class.
    - [x] Detect 1-minute SMA crossovers (21/200).
    - [x] Implement significance filter (configurable buffer time).
    - [x] Calculate SL (recent pivot lows/highs) and TP (fixed R:R ratio).
    - [x] Output `TradeSignal` object.
    - [ ] Add unit tests for V1 signal logic, SL/TP calculation.
- [x] **Step 007: Implement Order Manager (`src/order_manager.py`)**
    - [x] Receive `TradeSignal`.
    - [x] Set margin type and leverage.
    - [x] Implement position sizing (MVP: fixed margin allocation).
    - [x] Place MARKET entry, STOP_MARKET SL, TAKE_PROFIT_MARKET TP orders (with `reduceOnly=true`).
    - [x] Handle USDⓈ-M and COIN-M specifics.
    - [ ] Add unit tests for order manager logic (mocking API calls).
- [x] **Step 008: Implement Basic Position Manager (`src/position_manager.py`)**
    - [x] Track open positions (symbol, side, entry, size, SL/TP order IDs).
    - [x] Update status based on order fills/cancellations.
    - [ ] Add unit tests for position manager.- [x] **Step 009: Build Main Application Logic and Orchestration (`src/main.py`)**
    - [x] Initialize and orchestrate all modules.
    - [x] Manage data flow: WebSocket -> DataProcessor -> SignalEngine -> OrderManager.
    - [x] Handle startup, shutdown, config hot-reloads..
    - [ ] Implement main asynchronous loop.- [x] **Step 010: Add Basic Monitoring (`src/monitoring.py`)**
    - [x] Expose basic Prometheus metrics (uptime, WebSocket status, signals, errors).
    - [ ] Add unit tests for metrics exposure..
- [x] **Step 011: Write Unit and Integration Tests for MVP Scope**
    - [x] Ensure comprehensive unit test coverage for all new modules.
    - [x] Develop integration tests for key data flows (e.g., signal to order) with mocked Binance services- [x] **Step 012: Document Setup, Configuration, and Usage**
    - [x] Update `README.md` with setup, configuration, and running instructions.
    - [x] Create `docs/CONFIGURATION_GUIDE.md` detailing `config.yaml` options.
    - [x] Create `docs/ARCHITECTURE.md` with an overview of the system design.
    - [x] Create `docs/TROUBLESHOOTING.md` for common issues.net**
    - [ ] Configure bot for Binance Futures Testnet.
    - [ ] Run the bot and monitor its behavior for V1 strategy.
    - [ ] Verify signal generation, order placement (entry, SL, TP), and position tracking.
    - [ ] Debug and refine based on Testnet observations.
- [ ] **Step 014: Report and Send Functional App to User**
    - [ ] Package the application (e.g., as a zip archive with instructions or Docker image details).
    - [ ] Provide the functional app and necessary documentation to the user.

## Phase 2: Enhancements, V2 Strategy Foundation & Robustness (Post-MVP)
- [ ] Advanced V1 Filters (Volume, Volatility)
- [ ] Enhanced Risk Management for V1 (Dynamic Position Sizing, Trailing SL)
- [ ] COIN-M Full Support Refinement
- [ ] Signal Engine V2 Foundation (Multi-TF data, EMA cross, StochRSI, ATR SL/TP)
- [ ] Backtesting Framework (Initial)
- [ ] Improved Error Handling & Resilience
- [ ] Expanded Testing

## Phase 3: Full V2 Strategy & Advanced Features (Post-Phase 2)
- [ ] Signal Engine V2 Completion
- [ ] Advanced Filters for V2 (BTC Correlation, Funding Rate)
- [ ] Advanced Risk & Exit Strategies (Adaptive TP, Volatility-Based Leverage, Time-Based/Reversal Exits)
- [ ] Dynamic Pair Selection (Optional)
- [ ] Backtesting Optimization
- [ ] Advanced Monitoring & Alerting

## Phase 4: Deployment & Maintenance (Post-Phase 3)
- [ ] Dockerization (`Dockerfile`, `run.sh`)
- [ ] Deployment Strategy for 24/7 Free Operation
- [ ] CI/CD Pipeline (`.github/workflows/ci.yml`)
- [ ] Comprehensive Documentation Finalization
- [ ] Ongoing Maintenance Plan
