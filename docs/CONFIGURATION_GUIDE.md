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
backtesting:
  # ... see Backtesting section
```

## 1. API Configuration (`api`)

This section contains your Binance API credentials.

```yaml
api:
  binance_api_key: "YOUR_BINANCE_API_KEY"    # Your Binance API Key
  binance_api_secret: "YOUR_BINANCE_API_SECRET" # Your Binance API Secret
  testnet: false # Optional: Set to true to use Binance Testnet. Default is false (mainnet).
```

- **`binance_api_key`**: (Required) String. Your public API key from Binance. Ensure API restrictions are set appropriately (e.g., enable Futures trading, disable withdrawals if not needed by other tools).
- **`binance_api_secret`**: (Required) String. Your secret API key from Binance. Keep this confidential.
- **`testnet`**: (Optional) Boolean. If set to `true`, the bot will connect to the Binance Futures Testnet. If `false` or omitted, it connects to the mainnet. It is highly recommended to test thoroughly on testnet before trading with real funds.

## 2. Global Settings (`global_settings`)

These settings apply to all trading pairs unless overridden in the specific pair's configuration.

```yaml
global_settings:
  v1_strategy: # Settings specific to the V1 SMA Crossover strategy
    # ... see V1 Strategy section
  risk_management:
    # ... see Risk Management section
```

### 2.1. V1 Strategy (`v1_strategy`)

Parameters for the default SMA (Simple Moving Average) crossover strategy.

```yaml
v1_strategy:
  sma_short_period: 21
  sma_long_period: 200
  min_signal_interval_minutes: 15
  tp_sl_ratio: 3.0
  default_margin_usdt: 50.0
  default_leverage: 10
  margin_mode: "ISOLATED"
  fallback_sl_percentage: 0.02
  indicator_timeframes: ["1m", "5m", "15m", "1h", "4h"]
  
  filters:
    # ... see Filters section
  
  sl_options:
    # ... see Stop Loss Options section
  
  tp_options:
    # ... see Take Profit Options section
```

#### 2.1.1. Core Strategy Parameters

- **`sma_short_period`**: Integer. The period for the shorter SMA (e.g., 21). This is the faster-moving average.
- **`sma_long_period`**: Integer. The period for the longer SMA (e.g., 200). This is the slower-moving average.
- **`min_signal_interval_minutes`**: Integer. Minimum time in minutes to wait between generating new signals for the *same trading pair* to avoid over-trading. Set to `0` to disable this wait. Default is `15`.
- **`tp_sl_ratio`**: Float. The Take Profit to Stop Loss ratio (Risk/Reward Ratio). For example, `3.0` means the Take Profit distance will be three times the Stop Loss distance from the entry price. *Note: This setting is now superseded by the more detailed `tp_options.rr_ratio_settings.value`, but is maintained for backward compatibility.*
- **`default_margin_usdt`**: Float. For USDT-M pairs, the default amount of USDT to allocate as margin for a new trade if not specified per pair. This is used for fixed margin position sizing.
- **`default_leverage`**: Integer. The default leverage to use for new trades if not specified per pair (e.g., 10 for 10x leverage).
- **`margin_mode`**: String. The margin mode to use: `"ISOLATED"` or `"CROSSED"`. It is generally recommended to use `"ISOLATED"` for better risk control per position.
- **`fallback_sl_percentage`**: Float. *Deprecated*. This setting is maintained for backward compatibility but is superseded by `sl_options.percentage_settings.value`. It defines the percentage-based stop loss when pivot-based calculation fails. Default is `0.02` (2%).
- **`indicator_timeframes`**: List of strings. The k-line timeframes the bot should subscribe to and use for calculating indicators for each pair. Examples: `["1m", "5m", "15m", "1h", "4h"]`. The V1 strategy primarily uses the `1m` timeframe for signal generation, but other timeframes are fetched for potential future strategy enhancements or multi-timeframe analysis.

#### 2.1.2. Filters Configuration (`filters`)

Filters help reduce false signals by adding additional confirmation criteria beyond the basic SMA crossover.

```yaml
filters:
  buffer_time:
    enabled: false
    candles: 2
  
  volume:
    enabled: false
    lookback_periods: 20
    min_threshold: 1.5
  
  volatility:
    enabled: false
    indicator: "atr"
    lookback_periods: 14
    min_threshold: 0.002
    max_threshold: 0.05
```

##### Buffer Time Filter (`buffer_time`)

This filter requires the crossover condition to persist for a specified number of consecutive candles before generating a signal.

- **`enabled`**: Boolean. If `true`, the buffer time filter is active. If `false`, the filter is bypassed. Default is `false`.
- **`candles`**: Integer. The number of consecutive candles the crossover condition must persist before generating a signal. Default is `2`.

This filter helps reduce false signals during choppy market conditions by ensuring the crossover is not just a temporary fluctuation.

##### Volume Filter (`volume`)

This filter verifies that the trading volume is sufficiently high compared to recent average volume before generating a signal.

- **`enabled`**: Boolean. If `true`, the volume filter is active. If `false`, the filter is bypassed. Default is `false`.
- **`lookback_periods`**: Integer. The number of previous candles to consider when calculating the average volume. Default is `20`.
- **`min_threshold`**: Float. The minimum volume required as a multiplier of the average volume. For example, `1.5` means the current volume must be at least 1.5 times the average volume of the lookback period. Default is `1.5`.

High volume during a crossover can indicate stronger conviction in the market direction.

##### Volatility Filter (`volatility`)

This filter ensures that market volatility is within an acceptable range before generating a signal.

- **`enabled`**: Boolean. If `true`, the volatility filter is active. If `false`, the filter is bypassed. Default is `false`.
- **`indicator`**: String. The indicator to use for measuring volatility. Currently, only `"atr"` (Average True Range) is supported. Default is `"atr"`.
- **`lookback_periods`**: Integer. The number of previous candles to consider when calculating the volatility indicator. Default is `14`.
- **`min_threshold`**: Float. The minimum acceptable volatility as a percentage of price. Default is `0.002` (0.2%).
- **`max_threshold`**: Float. The maximum acceptable volatility as a percentage of price. Default is `0.05` (5%).

This filter helps avoid trading during extremely low volatility (when moves may not be significant enough to overcome trading costs) or during extremely high volatility (when markets may be unpredictable and risky).

#### 2.1.3. Stop Loss Options (`sl_options`)

Configuration for determining how stop losses are calculated.

```yaml
sl_options:
  primary_type: "pivot"
  
  pivot_settings:
    lookback_candles: 30
  
  percentage_settings:
    value: 0.02
  
  atr_settings:
    multiplier: 2.0
    lookback_periods: 14
  
  enable_fallback: true
  fallback_type: "percentage"
```

- **`primary_type`**: String. The primary method for calculating stop loss. Options:
  - `"pivot"`: Uses recent swing highs/lows as support/resistance levels for stop loss placement.
  - `"percentage"`: Uses a fixed percentage of the entry price.
  - `"atr"`: Uses a multiple of the Average True Range.
  Default is `"pivot"`.

##### Pivot-based Stop Loss Settings (`pivot_settings`)

- **`lookback_candles`**: Integer. The number of previous candles to analyze when finding pivot points (swing highs/lows). Default is `30`.

##### Percentage-based Stop Loss Settings (`percentage_settings`)

- **`value`**: Float. The stop loss as a percentage of the entry price. For example, `0.02` means a 2% stop loss from the entry price. Default is `0.02` (2%).

##### ATR-based Stop Loss Settings (`atr_settings`)

- **`multiplier`**: Float. The multiplier to apply to the ATR value. Higher values place the stop loss further from the entry price. Default is `2.0`.
- **`lookback_periods`**: Integer. The number of candles to consider when calculating the ATR. Default is `14`.

##### Fallback Configuration

- **`enable_fallback`**: Boolean. If `true`, the bot will use a fallback method when the primary method fails or gives invalid results. Default is `true`.
- **`fallback_type`**: String. The fallback method to use. Currently, only `"percentage"` is supported. Default is `"percentage"`.

#### 2.1.4. Take Profit Options (`tp_options`)

Configuration for determining how take profits are calculated.

```yaml
tp_options:
  primary_type: "rr_ratio"
  
  rr_ratio_settings:
    value: 3.0
  
  fixed_percentage_settings:
    value: 0.06
  
  atr_settings:
    multiplier: 4.0
    lookback_periods: 14
```

- **`primary_type`**: String. The primary method for calculating take profit. Options:
  - `"rr_ratio"`: Uses the risk-to-reward ratio applied to the stop loss distance.
  - `"fixed_percentage"`: Uses a fixed percentage of the entry price.
  - `"atr"`: Uses a multiple of the Average True Range.
  Default is `"rr_ratio"`.

##### Risk-Reward Ratio Settings (`rr_ratio_settings`)

- **`value`**: Float. The risk-to-reward ratio used to calculate take profit based on the stop loss distance. For example, `3.0` means the take profit distance will be 3 times the stop loss distance from the entry price. Default is `3.0`.

*Note: This setting supersedes the older global `tp_sl_ratio` parameter.*

##### Fixed Percentage Settings (`fixed_percentage_settings`)

- **`value`**: Float. The take profit as a percentage of the entry price. For example, `0.06` means a 6% take profit from the entry price. Default is `0.06` (6%).

##### ATR-based Take Profit Settings (`atr_settings`)

- **`multiplier`**: Float. The multiplier to apply to the ATR value. Higher values place the take profit further from the entry price. Default is `4.0`.
- **`lookback_periods`**: Integer. The number of candles to consider when calculating the ATR. Default is `14`.

### 2.2. Risk Management (`risk_management`)

Global risk parameters.

```yaml
risk_management:
  dynamic_sizing_risk_percent: 1.0
  dynamic_sizing_enabled: false
```

- **`dynamic_sizing_risk_percent`**: Float. If `dynamic_sizing_enabled` is `true`, this is the percentage of the available account balance to risk per trade. Default is `1.0` (1%).
- **`dynamic_sizing_enabled`**: Boolean. If `true`, enables dynamic position sizing based on account balance. If `false` (default), uses fixed margin allocation (`default_margin_usdt` or pair-specific `margin_usdt`).

## 3. Pairs Configuration (`pairs`)

This section defines the specific trading pairs the bot will monitor and trade. Each pair can have its own settings that override the global settings.

```yaml
pairs:
  BTC_USDT:
    enabled: true
    # Override global strategy settings if needed
    # sma_short_period: 10
    # leverage: 20
    # margin_usdt: 100
    
    # Pair-specific filter overrides
    filters:
      buffer_time:
        enabled: true
        candles: 3
      volume:
        enabled: true
        min_threshold: 2.0
    
    # Pair-specific SL/TP overrides
    sl_options:
      primary_type: "percentage"
      percentage_settings:
        value: 0.015
    tp_options:
      primary_type: "fixed_percentage"
      fixed_percentage_settings:
        value: 0.045

  ETH_USDT:
    enabled: true
    # leverage: 15
    # margin_usdt: 75

  BTCUSD_PERP:
    enabled: false
    # margin_coin: 0.01
    # leverage: 10
```

For each pair (e.g., `BTC_USDT`, `ETHUSD_PERP`): 
- The key (e.g., `BTC_USDT`) is a user-defined identifier. The bot will convert this to the format Binance API expects (e.g., `BTCUSDT` or `BTCUSD_PERP`).
  - For USDT-margined perpetuals, use the format `BASE_QUOTE` (e.g., `BTC_USDT`).
  - For COIN-margined perpetuals, use the format `BASEUSD_PERP` (e.g., `BTCUSD_PERP`).

### 3.1 Core Pair Settings

- **`enabled`**: Boolean. If `true`, the bot will actively trade this pair. If `false`, it will be ignored.
- **`leverage`**: (Optional) Integer. Pair-specific leverage. Overrides `global_settings.v1_strategy.default_leverage`.
- **`margin_usdt`**: (Optional) Float. For USDT-M contracts, the amount of USDT to use as margin for trades on this specific pair. Overrides `global_settings.v1_strategy.default_margin_usdt`.
- **`margin_coin`**: (Optional) Float. For COIN-M contracts, the amount of the base coin to use as margin.
- **`sma_short_period`**: (Optional) Integer. Pair-specific short SMA period. Overrides `global_settings.v1_strategy.sma_short_period`.
- **`sma_long_period`**: (Optional) Integer. Pair-specific long SMA period. Overrides `global_settings.v1_strategy.sma_long_period`.
- **`min_signal_interval_minutes`**: (Optional) Integer. Pair-specific minimum signal interval. Overrides `global_settings.v1_strategy.min_signal_interval_minutes`.

### 3.2 Pair-Specific Filter Overrides (`filters`)

Each pair can override the global filter settings. The structure is identical to the global filters configuration.

```yaml
filters:
  buffer_time:
    enabled: true
    candles: 3
  volume:
    enabled: true
    min_threshold: 2.0
  volatility:
    enabled: true
    min_threshold: 0.003
    max_threshold: 0.04
```

All filter settings can be overridden at the pair level. If a setting is not specified, the global setting is used.

### 3.3 Pair-Specific Stop Loss Overrides (`sl_options`)

Each pair can override the global stop loss settings.

```yaml
sl_options:
  primary_type: "percentage"
  percentage_settings:
    value: 0.015
  enable_fallback: false
```

All stop loss settings can be overridden at the pair level. If a setting is not specified, the global setting is used.

### 3.4 Pair-Specific Take Profit Overrides (`tp_options`)

Each pair can override the global take profit settings.

```yaml
tp_options:
  primary_type: "fixed_percentage"
  fixed_percentage_settings:
    value: 0.045
```

All take profit settings can be overridden at the pair level. If a setting is not specified, the global setting is used.

## 4. Logging Configuration (`logging`)

Controls how the bot logs information.

```yaml
logging:
  level: "INFO" # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  # log_file: "logs/bot.log" # Optional: Path to a log file.
```

- **`level`**: String. The minimum logging level to output. Options (from least to most verbose): `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`. Default is `INFO`.
- **`log_file`**: (Optional) String. Path to the file where logs should be saved. If commented out or omitted, logs will only be printed to the console.

## 5. Monitoring Configuration (`monitoring`)

Settings for the Prometheus metrics exporter.

```yaml
monitoring:
  prometheus_port: 8000 # Port for the Prometheus metrics HTTP server
```

- **`prometheus_port`**: Integer. The TCP port on which the bot will expose Prometheus metrics. Default is `8000`.

## 6. Backtesting Configuration (`backtesting`)

Settings for the backtesting module.

```yaml
backtesting:
  initial_balance_usdt: 10000
  fee_percent: 0.04
  slippage_percent: 0.01
```

- **`initial_balance_usdt`**: Float. The initial USDT balance for backtesting. Default is `10000`.
- **`fee_percent`**: Float. The trading fee as a percentage (e.g., `0.04` means 0.04% or 0.0004 in decimal). Default is `0.04`.
- **`slippage_percent`**: Float. The estimated slippage as a percentage to simulate real trading conditions. Default is `0.01`.

## Hot Reloading

The bot monitors the `config/config.yaml` file for changes. Most settings (API keys excluded for security during runtime) can be updated live without restarting the bot. This includes:
- Logging levels
- Enabling/disabling pairs
- Leverage, margin settings for pairs
- Strategy parameters including filters, SL/TP options
- Indicator timeframes

When the file is saved, the bot will detect the changes and apply them. Check the bot's logs for confirmation of reloaded configurations.

## Configuration Tips

1. **Start Simple**: Begin with the most basic configuration (just the SMA crossover strategy) and add filters one by one to observe their effect.
2. **Testnet First**: Always test your configuration on the Binance Futures Testnet before using real funds.
3. **Monitor Logs**: Set `logging.level` to `"DEBUG"` initially to see detailed information about signal generation, filter decisions, and order placement.
4. **Pair Selection**: Focus on liquid pairs with stable trading volumes to avoid excessive slippage.
5. **Risk Management**: Start with small position sizes until you are confident in your configuration.

---

*Always ensure your configuration is validated and tested on a testnet environment before deploying with real funds. Incorrect configurations can lead to unintended trading behavior and financial loss.*
