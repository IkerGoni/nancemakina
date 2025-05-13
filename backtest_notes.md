# Test 12B Backtest Notes

## Configuration Parameters
- **Strategy**: V1 SMA Crossover
- **Engine**: SignalEngineV1
- **Short SMA Period**: 21
- **Long SMA Period**: 200
- **Min Signal Interval**: 10 minutes
- **Trend Filter**: Enabled (using SMA 200)
- **Other Filters**: Buffer time, Volume, Volatility all disabled
- **Stop Loss**: Primary method is pivot (lookback=40), fallback to ATR (periods=14, multiplier=2.1)
- **Take Profit**: Risk-Reward ratio of 2.25
- **Initial Balance**: 1000 USDT
- **Trading Fee**: 0.02%
- **Slippage**: 0.0% (none)

## Dataset
- File: `merged_futures_data_quarter.csv`
- Symbol: BTCUSDT
- Contract Type: USDT-M

## Results
*The backtest is currently running. Results will be updated once completed.*

## Performance Metrics
To be filled after backtest completion:
- Final Balance: TBD
- Net Profit/Loss: TBD
- Total Trades: TBD
- Win Rate: TBD
- Profit Factor: TBD
- Maximum Drawdown: TBD

## Observations
*To be filled after analyzing the results*

## Next Steps
1. Compare results with previous tests
2. Analyze trades for any pattern or issues
3. Consider adjustments to parameters if needed 