# Configuration Guide

This document provides a detailed explanation of all options available in the `config/config.yaml` file for the Binance Futures Trading Bot.

## Root Level Configuration

```yaml
api:
  # ... see API section
global_settings:
  # ... see Global Settings section
pairs:
  # ... see Pairs section
logging:
  # ... see Logging section
monitoring:
  # ... see Monitoring section
```

## 1. API Configuration (`api`)

This section contains your Binance API credentials.

```yaml
api:
  binance_api_key: "YOUR_BINANCE_API_KEY"    # Your Binance API Key
  binance_api_secret: "YOUR_BINANCE_API_SECRET" # Your Binance API Secret
  # testnet: false # Optional: Set to true to use Binance Testnet. Default is false (mainnet).
```

- **`binance_api_key`**: (Required) Your public API key from Binance. Ensure API restrictions are set appropriately (e.g., enable Futures trading, disable withdrawals if not needed by other tools).
- **`binance_api_secret`**: (Required) Your secret API key from Binance. Keep this confidential.
- **`testnet`**: (Optional) Boolean. If set to `true`, the bot will connect to the Binance Futures Testnet. If `false` or omitted, it connects to the mainnet. It is highly recommended to test thoroughly on testnet before trading with real funds.

## 2. Global Settings (`global_settings`)

These settings apply to all trading pairs unless overridden in the specific pair's configuration.

```yaml
global_settings:
  v1_strategy: # Settings specific to the V1 SMA Crossover strategy
    sma_short_period: 21
    sma_long_period: 200
    min_signal_interval_minutes: 60
    tp_sl_ratio: 2.0
    default_margin_usdt: 50.0
    default_leverage: 10
    margin_mode: "ISOLATED" # or "CROSSED"
    indicator_timeframes: ["1m", "5m", "15m", "1h", "4h"]
    # buffer_time_candles: 3 # Optional: Number of candles for buffer time after crossover
    # pivot_lookback_candles: 30 # Optional: Lookback period for SL pivot calculation

  risk_management:
    # dynamic_sizing_enabled: false # Optional: Enable dynamic position sizing (e.g., % of balance)
    # max_account_risk_per_trade_pct: 1.0 # Optional: If dynamic sizing, max % of account to risk
    # max_concurrent_trades: 5 # Optional: Limit on total concurrent open positions
```

### 2.1. V1 Strategy (`v1_strategy`)

Parameters for the default SMA (Simple Moving Average) crossover strategy.

- **`sma_short_period`**: Integer. The period for the shorter SMA (e.g., 21).
- **`sma_long_period`**: Integer. The period for the longer SMA (e.g., 200).
- **`min_signal_interval_minutes`**: Integer. Minimum time in minutes to wait between generating new signals for the *same trading pair* to avoid over-trading. Set to `0` to disable this wait.
- **`tp_sl_ratio`**: Float. The Take Profit to Stop Loss ratio (Risk/Reward Ratio). For example, `2.0` means the Take Profit distance will be twice the Stop Loss distance from the entry price.
- **`default_margin_usdt`**: Float. For USDT-M pairs, the default amount of USDT to allocate as margin for a new trade if not specified per pair. This is used for fixed margin position sizing.
- **`default_leverage`**: Integer. The default leverage to use for new trades if not specified per pair (e.g., 10 for 10x leverage).
- **`margin_mode`**: String. The margin mode to use: `"ISOLATED"` or `"CROSSED"`. It is generally recommended to use `"ISOLATED"` for better risk control per position.
- **`indicator_timeframes`**: List of strings. The k-line timeframes the bot should subscribe to and use for calculating indicators for each pair. Examples: `["1m", "5m", "15m", "1h", "4h"]`. The V1 strategy primarily uses the `1m` timeframe for signal generation based on the provided documents, but other timeframes are fetched for potential future strategy enhancements or multi-timeframe analysis.
- **`buffer_time_candles`**: (Optional) Integer. Number of candles the crossover condition must persist before a signal is considered valid. This helps filter out false signals during choppy markets. (As per "Binance Futures Trading Bot Strategy Improvements (2).txt")
- **`pivot_lookback_candles`**: (Optional) Integer. The lookback period (number of candles) used to determine recent pivot highs/lows for stop-loss placement. (As per "Binance Futures Trading Bot Strategy Improvements (2).txt")

### 2.2. Risk Management (`risk_management`)

Global risk parameters (some are placeholders for future enhancements).

- **`dynamic_sizing_enabled`**: (Optional) Boolean. If `true`, enables dynamic position sizing (e.g., risking a fixed percentage of account balance per trade). If `false` (default), uses fixed margin allocation (`default_margin_usdt` or pair-specific `margin_usdt`). *MVP focuses on fixed margin.*
- **`max_account_risk_per_trade_pct`**: (Optional) Float. If `dynamic_sizing_enabled` is `true`, this is the maximum percentage of the total account balance to risk on a single trade (e.g., `1.0` for 1%).
- **`max_concurrent_trades`**: (Optional) Integer. The maximum number of trades the bot can have open simultaneously across all pairs.

## 3. Pairs Configuration (`pairs`)

This section defines the specific trading pairs the bot will monitor and trade. You can list multiple pairs.

```yaml
pairs:
  BTC_USDT: # User-defined name for the pair, used internally
    enabled: true
    contract_type: "USDT_M" # "USDT_M" or "COIN_M"
    leverage: 20 # Optional: Overrides global_settings.v1_strategy.default_leverage
    margin_usdt: 100.0 # Optional: For USDT_M, overrides global_settings.v1_strategy.default_margin_usdt
    # margin_coin: 0.01 # Optional: For COIN_M, amount of base coin for margin (e.g., 0.01 BTC for BTCUSD_PERP)
    # indicator_timeframes: ["1m", "15m"] # Optional: Overrides global_settings.v1_strategy.indicator_timeframes for this pair
    # min_signal_interval_minutes: 30 # Optional: Overrides global setting for this pair
    # tp_sl_ratio: 1.5 # Optional: Overrides global setting for this pair

  ETH_USDT:
    enabled: true
    contract_type: "USDT_M"
    # Uses global settings for leverage, margin_usdt, etc.

  BTCUSD_PERP: # Example for a COIN-M contract
    enabled: false # This pair is currently disabled
    contract_type: "COIN_M"
    leverage: 5
    margin_coin: 0.002 # e.g., 0.002 BTC for this contract
```

For each pair (e.g., `BTC_USDT`, `ETHUSD_PERP`): 
- The key (e.g., `BTC_USDT`) is a user-defined identifier. The bot will convert this to the format Binance API expects (e.g., `BTCUSDT` or `BTCUSD_PERP`).
  - For USDT-margined perpetuals, use the format `BASE_QUOTE` (e.g., `BTC_USDT`).
  - For COIN-margined perpetuals, use the format `BASEUSD_PERP` (e.g., `BTCUSD_PERP`).
- **`enabled`**: Boolean. If `true`, the bot will actively trade this pair. If `false`, it will be ignored.
- **`contract_type`**: String. Specifies the type of futures contract:
    - `"USDT_M"`: For USDT-margined contracts (e.g., BTC/USDT, ETH/USDT).
    - `"COIN_M"`: For COIN-margined contracts (e.g., BTC/USD, ETH/USD perpetuals).
- **`leverage`**: (Optional) Integer. Pair-specific leverage. Overrides `global_settings.v1_strategy.default_leverage`.
- **`margin_usdt`**: (Optional) Float. For `USDT_M` contracts, the amount of USDT to use as margin for trades on this specific pair. Overrides `global_settings.v1_strategy.default_margin_usdt`.
- **`margin_coin`**: (Optional) Float. For `COIN_M` contracts, the amount of the base coin (e.g., BTC for BTCUSD_PERP) to use as margin for trades on this specific pair. *Note: For COIN-M, position sizing is typically in terms of number of contracts. This `margin_coin` value might be interpreted as the number of contracts to trade for MVP if fixed sizing is used, or as the coin margin if the API supports that directly for sizing.*
- **`indicator_timeframes`**: (Optional) List of strings. Pair-specific k-line timeframes. Overrides `global_settings.v1_strategy.indicator_timeframes`.
- **`min_signal_interval_minutes`**: (Optional) Integer. Pair-specific minimum signal interval. Overrides `global_settings.v1_strategy.min_signal_interval_minutes`.
- **`tp_sl_ratio`**: (Optional) Float. Pair-specific Take Profit / Stop Loss ratio. Overrides `global_settings.v1_strategy.tp_sl_ratio`.

## 4. Logging Configuration (`logging`)

Controls how the bot logs information.

```yaml
logging:
  level: "INFO" # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  # file: "logs/bot.log" # Optional: Path to a log file. If omitted, logs to console only.
  # rotation_size_mb: 10 # Optional: Max size in MB before log file rotates.
  # rotation_backup_count: 5 # Optional: Number of backup log files to keep.
```

- **`level`**: String. The minimum logging level to output. Options (from least to most verbose): `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`.
- **`file`**: (Optional) String. Path to the file where logs should be saved. If commented out or omitted, logs will only be printed to the console.
- **`rotation_size_mb`**: (Optional) Integer. If `file` is specified, this is the maximum size in megabytes a log file can reach before it is rotated (e.g., `bot.log` becomes `bot.log.1`).
- **`rotation_backup_count`**: (Optional) Integer. If `file` and `rotation_size_mb` are specified, this is the number of old log files to keep.

## 5. Monitoring Configuration (`monitoring`)

Settings for the Prometheus metrics exporter.

```yaml
monitoring:
  prometheus_port: 8000 # Port for the Prometheus metrics HTTP server
  # enabled: true # Optional: Set to false to disable Prometheus exporter. Defaults to true.
```

- **`prometheus_port`**: Integer. The TCP port on which the bot will expose Prometheus metrics. Default is `8000`.
- **`enabled`**: (Optional) Boolean. Set to `false` to disable the Prometheus metrics server. Defaults to `true` if the section exists.

## Hot Reloading

The bot monitors the `config/config.yaml` file for changes. Most settings (API keys excluded for security during runtime) can be updated live without restarting the bot. This includes:
- Logging levels
- Enabling/disabling pairs
- Leverage, margin settings for pairs
- Strategy parameters like `min_signal_interval_minutes`, `tp_sl_ratio`
- Indicator timeframes

When the file is saved, the bot will detect the changes and apply them. Check the bot's logs for confirmation of reloaded configurations.

---

*Always ensure your configuration is validated and tested on a testnet environment before deploying with real funds. Incorrect configurations can lead to unintended trading behavior and financial loss.*
