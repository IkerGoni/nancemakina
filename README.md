# Binance Futures Trading Bot

A robust, configurable trading bot for Binance Futures markets (USDT-M and COIN-M) with a focus on reliability, configurability, and extensibility.

## Features

- **Multi-Market Support**: Trade on both USDT-M and COIN-M Binance Futures markets
- **Configurable Strategy**: SMA crossover strategy (21 SMA crossing 200 SMA) with configurable parameters
- **Real-time Data Processing**: WebSocket connections for live market data
- **Risk Management**: Configurable position sizing, stop-loss, and take-profit
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

The bot is configured through the `config/config.yaml` file. See [Configuration Guide](docs/CONFIGURATION_GUIDE.md) for detailed options.

Basic configuration example:

```yaml
api:
  binance_api_key: "YOUR_BINANCE_API_KEY"
  binance_api_secret: "YOUR_BINANCE_API_SECRET"

global_settings:
  v1_strategy:
    sma_short_period: 21
    sma_long_period: 200
    min_signal_interval_minutes: 60
    tp_sl_ratio: 2.0
    default_margin_usdt: 50.0
    default_leverage: 10
    margin_mode: "ISOLATED"
    indicator_timeframes: ["1m", "5m", "15m", "1h", "4h"]

pairs:
  BTC_USDT:
    enabled: true
    contract_type: "USDT_M"
    leverage: 5
    margin_usdt: 100.0
  ETH_USDT:
    enabled: true
    contract_type: "USDT_M"
  BTCUSD_PERP:
    enabled: false
    contract_type: "COIN_M"
    margin_coin: 0.001

logging:
  level: "INFO"
  file: "logs/bot.log"

monitoring:
  prometheus_port: 8000
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
