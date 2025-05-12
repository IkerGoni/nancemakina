# Binance Futures Trading Bot

A robust, configurable trading bot for Binance Futures markets (USDT-M and COIN-M) with a focus on reliability, configurability, and extensibility.

## Features

- **Multi-Market Support**: Trade on both USDT-M and COIN-M Binance Futures markets
- **Advanced SMA Crossover Strategy**: Configurable moving averages with multiple validation filters:
  - Buffer Time Filter: Requires consecutive candles to confirm crossover
  - Volume Filter: Validates signals based on higher-than-average volume
  - Volatility Filter: Ensures market volatility is within acceptable ranges
- **Flexible Risk Management**:
  - Multiple Stop Loss types: Pivot-based, Percentage-based, or ATR-based
  - Multiple Take Profit types: Risk-Reward Ratio, Fixed Percentage, or ATR-based
  - Fallback mechanisms when primary calculation methods fail
- **Real-time Data Processing**: WebSocket connections for live market data
- **Position Sizing**: Configurable fixed or dynamic position sizing
- **Monitoring**: Prometheus metrics for monitoring bot performance and health
- **Hot-reload Configuration**: Update trading parameters without restarting the bot
- **Extensible Architecture**: Modular design for easy addition of new strategies and features

## Installation

### Prerequisites

- Python 3.8+
- Binance Futures account with API keys
- (Optional) Docker for containerized deployment

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/binance-futures-bot.git
   cd binance-futures-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create configuration file:
   ```bash
   cp config/config.yaml.example config/config.yaml
   ```

4. Edit the configuration file with your Binance API keys and trading preferences:
   ```bash
   nano config/config.yaml
   ```

## Configuration

The bot is configured through the `config/config.yaml` file. Configuration includes:

- API connections (with Testnet support)
- Trading pairs configuration
- Strategy parameters (SMA periods, signal intervals)
- Advanced filters (Buffer Time, Volume, Volatility)
- Stop Loss and Take Profit options
- Risk management settings
- Logging and monitoring

See the comprehensive [Configuration Guide](docs/CONFIGURATION_GUIDE.md) for detailed options and examples.

For quick testing on Testnet, start with:

```yaml
api:
  binance_api_key: "YOUR_TESTNET_API_KEY"
  binance_api_secret: "YOUR_TESTNET_API_SECRET"
  testnet: true

global_settings:
  v1_strategy:
    # Core strategy settings remain unchanged
    
    # Enable just one filter to start
    filters:
      buffer_time:
        enabled: true
        candles: 2
      volume:
        enabled: false
      volatility:
        enabled: false
    
    # Using pivot-based SL with percentage fallback
    sl_options:
      primary_type: "pivot"
      enable_fallback: true
      fallback_type: "percentage"
    
    # Simple risk-reward ratio for TP
    tp_options:
      primary_type: "rr_ratio"
      rr_ratio_settings:
        value: 2.0

pairs:
  BTC_USDT:
    enabled: true
    leverage: 5
    margin_usdt: 50.0

logging:
  level: "DEBUG"  # Use DEBUG for maximum visibility during testing
```

## Running the Bot

### Standard Mode

Run the bot with:

```bash
python -m src.main
```

### Development Mode

For development with hot-reloading of code changes:

```bash
python -m src.main --dev
```

### Docker

Build and run with Docker:

```bash
docker build -t binance-futures-bot .
docker run -v $(pwd)/config:/app/config -v $(pwd)/logs:/app/logs binance-futures-bot
```

## Monitoring

The bot exposes Prometheus metrics on port 8000 (configurable). You can use Grafana or other Prometheus-compatible tools to visualize these metrics.

Default metrics endpoint: `http://localhost:8000/metrics`

## Architecture

The bot follows a modular architecture with clear separation of concerns. See [Architecture Overview](docs/ARCHITECTURE.md) for details on the system design.

Key components:
- **Config Manager**: Handles configuration loading and hot-reloading
- **WebSocket Connector**: Manages real-time data streams from Binance
- **REST Client**: Handles API calls for account data and order execution
- **Data Processor**: Processes market data and calculates indicators
- **Signal Engine**: Generates trade signals based on strategy rules
- **Order Manager**: Executes trades with proper position sizing
- **Position Manager**: Tracks open positions and their status

## Testing

Run the test suite with:

```bash
pytest
```

For specific test categories:

```bash
pytest tests/unit/  # Unit tests only
pytest tests/integration/  # Integration tests only
```

## Troubleshooting

See [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for common issues and solutions.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred from using this software. Always test thoroughly on testnet before using with real funds.

## License

MIT License
