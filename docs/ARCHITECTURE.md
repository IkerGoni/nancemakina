# Architecture Overview

This document provides an overview of the system architecture for the Binance Futures Trading Bot. The bot is designed with a modular approach to ensure separation of concerns, testability, and extensibility.

## Core Principles

- **Modularity**: Each major function (data ingestion, signal generation, order execution, etc.) is handled by a distinct module.
- **Asynchronous Operations**: Leverages `asyncio` for efficient handling of I/O-bound tasks like network requests and WebSocket communication.
- **Configurability**: Most operational parameters are externalized to a YAML configuration file, supporting hot-reloading.
- **Event-Driven**: The bot primarily reacts to real-time market events (k-line updates) and internal signals.
- **Testability**: Modules are designed to be unit-testable, with dependencies injectable or mockable.

## System Components

The following diagram illustrates the high-level interaction between the core components:

```mermaid
graph TD
    subgraph User_Interaction
        ConfigFile[config.yaml]
    end

    subgraph Bot_Core_Application
        MainApp[Main Application (main.py)]
        ConfigMgr[ConfigManager (config_loader.py)]
        Scheduler[Async Event Loop]
    end

    subgraph Data_Pipeline
        BinanceWS[Binance WebSocket API]
        WSConnector[WebSocketConnector (connectors.py)]
        DataProcessor[DataProcessor (data_processor.py)]
    end

    subgraph Trading_Logic
        SignalEngine[SignalEngineV1 (signal_engine.py)]
        OrderManager[OrderManager (order_manager.py)]
        PositionManager[PositionManager (position_manager.py)]
    end

    subgraph External_Services
        BinanceREST[Binance REST API]
        RESTClient[RESTClient (connectors.py)]
    end

    subgraph Monitoring_System
        Prometheus[Prometheus Server]
        Metrics[PrometheusMonitor (monitoring.py)]
    end

    ConfigFile --> ConfigMgr
    ConfigMgr --> MainApp
    ConfigMgr --> WSConnector
    ConfigMgr --> DataProcessor
    ConfigMgr --> SignalEngine
    ConfigMgr --> OrderManager
    ConfigMgr --> PositionManager
    ConfigMgr --> RESTClient
    ConfigMgr --> Metrics

    MainApp -- Manages/Orchestrates --> WSConnector
    MainApp -- Manages/Orchestrates --> DataProcessor
    MainApp -- Manages/Orchestrates --> SignalEngine
    MainApp -- Manages/Orchestrates --> OrderManager
    MainApp -- Manages/Orchestrates --> PositionManager
    MainApp -- Manages/Orchestrates --> RESTClient
    MainApp -- Manages/Orchestrates --> Metrics
    MainApp -- Uses --> Scheduler

    BinanceWS -- Streams Data --> WSConnector
    WSConnector -- Raw Kline Data --> MainApp
    MainApp -- Kline Data --> DataProcessor
    DataProcessor -- Indicator Data --> SignalEngine
    SignalEngine -- Trade Signals --> MainApp
    MainApp -- Trade Signals --> OrderManager
    
    OrderManager -- Executes Orders via --> RESTClient
    RESTClient -- Interacts with --> BinanceREST
    PositionManager -- Tracks Positions --> MainApp # Position updates can come from OrderManager or UserDataStream
    OrderManager -- Updates --> PositionManager # After successful order execution

    Metrics -- Exposes Data to --> Prometheus
```

### 1. Main Application (`src/main.py`)
- **Responsibilities**: 
    - Initializes and orchestrates all other modules.
    - Manages the main asynchronous event loop.
    - Handles startup, graceful shutdown, and signal handling (SIGINT, SIGTERM).
    - Routes data between components (e.g., k-line data from WebSocket connector to Data Processor, signals from Signal Engine to Order Manager).
    - Responds to configuration changes detected by the `ConfigManager`.

### 2. Configuration Manager (`src/config_loader.py`)
- **Responsibilities**:
    - Loads trading parameters, API keys, and other settings from `config/config.yaml`.
    - Provides access to configuration values for all other modules.
    - Implements hot-reloading: monitors the configuration file for changes and notifies registered callbacks in other modules to apply updates dynamically.
    - Handles creation of default `config.yaml` from `config.yaml.example` if it doesn_t exist.

### 3. Connectors (`src/connectors.py`)
   Contains modules for interacting with Binance APIs.
   - **`BinanceWebSocketConnector`**:
     - Establishes and maintains WebSocket connections to Binance Futures for real-time market data (k-lines, potentially order book depth, user data streams in future).
     - Subscribes to streams for configured trading pairs and timeframes.
     - Parses incoming JSON messages into structured `Kline` objects (defined in `src/models.py`).
     - Passes `Kline` data to the Main Application via a callback for further processing.
     - Handles connection management, automatic reconnections, and subscription updates on configuration changes.
   - **`BinanceRESTClient`**:
     - Interacts with the Binance Futures REST API for actions like placing orders, fetching account balance, getting open positions, setting leverage/margin mode, and retrieving exchange information (symbol precision, limits).
     - Uses the `ccxt` library for standardized API interaction.
     - Manages API authentication (API key and secret).
     - Implements retry logic for transient API errors (using `tenacity`).
     - Provides a simulation mode if API keys are not configured or are placeholders, allowing basic testing without live trading.

### 4. Data Models (`src/models.py`)
- **Responsibilities**:
    - Defines Pydantic data models for structured representation of various entities used throughout the application. This ensures data consistency and provides validation.
    - Key models include:
        - `Kline`: Represents a single k-line/candlestick data point.
        - `PairConfig`: Represents configuration for a specific trading pair.
        - `TradeSignal`: Represents a trading signal generated by the strategy engine.
        - `Order`: Represents a trading order (entry, SL, TP).
        - `Position`: Represents an open trading position.
        - Enums for `TradeDirection`, `OrderStatus`, `OrderType`, `OrderSide`, etc.

### 5. Data Processor (`src/data_processor.py`)
- **Responsibilities**:
    - Receives raw `Kline` objects from the Main Application (originating from the `WebSocketConnector`).
    - Maintains rolling buffers (e.g., `collections.deque` of `Kline` objects) for each active trading pair and timeframe.
    - Calculates technical indicators (e.g., SMAs for V1 strategy) based on the k-line data in the buffers.
    - Stores calculated indicators, typically in a Pandas DataFrame associated with each pair/timeframe buffer for easy access and analysis.
    - Provides methods for other modules (like `SignalEngine`) to retrieve the latest k-line data and calculated indicators.
    - Updates its internal state and buffers based on configuration changes (e.g., new pairs enabled, indicator timeframes changed).

### 6. Signal Engine (`src/signal_engine.py`)
- **Responsibilities**:
    - Implements the trading strategy logic (e.g., V1 SMA Crossover).
    - Uses data provided by the `DataProcessor` (k-lines and indicators) to identify potential trading opportunities.
    - When a trading condition is met (e.g., SMA crossover), it generates a `TradeSignal` object.
    - Includes logic for signal filtering, such as minimum interval between signals for the same pair, and buffer time for crossover confirmation.
    - Calculates initial stop-loss (SL) and take-profit (TP) levels for the generated signal based on strategy rules (e.g., recent pivot lows/highs for SL, fixed R:R ratio for TP).
    - Passes the `TradeSignal` to the Main Application for execution.

### 7. Order Manager (`src/order_manager.py`)
- **Responsibilities**:
    - Receives `TradeSignal` objects from the Main Application.
    - Determines position size based on configured margin (fixed USDT amount for USDT-M, fixed coin amount for COIN-M in MVP) and leverage.
    - Fetches symbol-specific trading rules (precision for price/quantity, minimum notional value) from `BinanceRESTClient` to ensure orders are valid.
    - Constructs and places market entry orders, and associated stop-market SL and take-profit-market TP orders (with `reduceOnly=true`) via the `BinanceRESTClient`.
    - Sets margin type (ISOLATED/CROSSED) and leverage for the pair before placing trades if necessary.
    - Handles specifics for USDT-M and COIN-M contracts.
    - Logs order placement details and outcomes.
    - (Future Enhancement) Could interact with `PositionManager` to update position state after orders are confirmed.

### 8. Position Manager (`src/position_manager.py`)
- **Responsibilities**:
    - Tracks currently open trading positions managed by the bot.
    - Stores details for each position: symbol, side (LONG/SHORT), entry price, quantity, entry/SL/TP order IDs, unrealized PnL, etc.
    - Updates position status based on order fill events (e.g., entry filled, SL/TP hit, manual close). This is crucial and would typically be driven by a user data WebSocket stream or polling order statuses.
    - Provides methods to query current open positions.
    - (Future Enhancement) Handles reconciliation of positions with the exchange on startup or periodically.
    - (Future Enhancement) Manages trailing stops or other dynamic exit strategies.

### 9. Monitoring (`src/monitoring.py`)
- **Responsibilities**:
    - Implements a Prometheus metrics exporter.
    - Exposes key operational metrics such as:
        - Bot uptime, error counts.
        - WebSocket connection status, messages received.
        - K-lines processed, indicator calculation times.
        - Signals generated, orders placed/filled/failed.
        - Active position counts, PnL (if available).
    - Starts an HTTP server on a configurable port to serve the `/metrics` endpoint for Prometheus to scrape.

## Data Flow Example (V1 SMA Crossover Strategy)

1. **Configuration**: `ConfigManager` loads `config.yaml`.
2. **Initialization**: `MainApp` initializes all components using the loaded configuration.
3. **WebSocket Connection**: `WebSocketConnector` connects to Binance and subscribes to k-line streams for enabled pairs (e.g., BTCUSDT 1m, 5m, etc.).
4. **Kline Reception**: `WebSocketConnector` receives a k-line update, parses it into a `Kline` object, and passes it to `MainApp`.
5. **Data Processing**: `MainApp` forwards the `Kline` to `DataProcessor`.
   - `DataProcessor` appends the k-line to the relevant buffer (e.g., BTCUSDT, 1m).
   - It then recalculates indicators (e.g., 21 SMA, 200 SMA) for that buffer and updates its internal DataFrame.
6. **Signal Check**: If the k-line is closed (for 1m timeframe as per V1 strategy):
   - `MainApp` (or a callback mechanism) triggers `SignalEngineV1` to check for signals on BTCUSDT.
   - `SignalEngineV1` requests the latest indicator DataFrame for BTCUSDT 1m from `DataProcessor`.
   - It checks for SMA crossover conditions (e.g., 21 SMA crosses above 200 SMA).
   - If a crossover is confirmed (and passes filters like `min_signal_interval_minutes` and `buffer_time_candles`):
     - It calculates SL (e.g., recent low) and TP (e.g., entry + (entry-SL) * R:R).
     - It creates a `TradeSignal` object (e.g., LONG BTCUSDT).
     - It returns the `TradeSignal` to `MainApp`.
7. **Order Execution**: `MainApp` receives the `TradeSignal` and passes it to `OrderManager`.
   - `OrderManager` determines position size (e.g., 100 USDT margin, 10x leverage for BTCUSDT).
   - It fetches BTCUSDT trading rules (precision, min notional) from `RESTClient` (cached if possible).
   - It adjusts quantity and prices to meet precision requirements.
   - It calls `RESTClient` to:
     - Set leverage and margin mode for BTCUSDT (if not already set or changed).
     - Place a MARKET BUY order (entry).
     - Place a STOP_MARKET SELL order (SL) with `reduceOnly=true`.
     - Place a TAKE_PROFIT_MARKET SELL order (TP) with `reduceOnly=true`.
8. **Position Tracking**: 
   - `OrderManager` (or `MainApp` upon order confirmation) informs `PositionManager` about the new potential position, including the order IDs for entry, SL, and TP.
   - `PositionManager` creates a new `Position` entry.
   - (Ideally, a User Data Stream via `WebSocketConnector` would provide real-time order fill updates, which would then be routed to `PositionManager` to confirm the position is active or closed).
9. **Monitoring**: Throughout this process, modules update metrics via `PrometheusMonitor` (e.g., k-lines processed, signals generated, orders placed).

## Future Enhancements Considerations

- **User Data Stream**: Integrate a user data WebSocket stream for real-time updates on order fills, account balance changes, and position updates. This would make position management more robust and reactive.
- **Database Integration**: For persistent storage of trades, historical performance, and potentially k-line data for backtesting.
- **Advanced Strategies**: The modular design allows for new strategy engines to be developed and plugged in.
- **Web UI/Dashboard**: A user interface for monitoring the bot, managing settings, and viewing performance.
- **Backtesting Engine**: To test strategies on historical data before live deployment.

This architecture provides a solid foundation for a reliable and extensible trading bot.
