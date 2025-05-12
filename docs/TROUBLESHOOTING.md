# Troubleshooting Guide

This document provides solutions for common issues you might encounter when running the Binance Futures Trading Bot.

## Connection Issues

### WebSocket Connection Failures

**Symptoms:**
- Repeated log messages about WebSocket connection failures
- Bot not receiving market data

**Possible Solutions:**
1. **Check Internet Connection**: Ensure your server has a stable internet connection.
2. **Verify API Endpoints**: Confirm Binance API endpoints are accessible from your location.
3. **Check Firewall Settings**: Make sure outbound WebSocket connections are allowed.
4. **Proxy Configuration**: If using a proxy, verify it's correctly configured.

```bash
# Test connectivity to Binance WebSocket endpoint
ping fstream.binance.com
```

### REST API Connection Issues

**Symptoms:**
- Error messages when placing orders
- Failed account balance retrieval

**Possible Solutions:**
1. **Verify API Keys**: Ensure your API keys are valid and have the correct permissions.
2. **Check IP Restrictions**: If you've set IP restrictions on your API keys, verify your server's IP is allowed.
3. **Rate Limits**: You might be hitting Binance's rate limits. Check logs for 429 errors and consider reducing request frequency.

## Configuration Issues

### Bot Not Trading Expected Pairs

**Symptoms:**
- Some configured pairs are not being traded
- No signals generated for certain pairs

**Possible Solutions:**
1. **Check Pair Configuration**: Ensure the pair is correctly formatted and `enabled: true` is set.
2. **Verify Symbol Existence**: Confirm the trading pair exists on Binance Futures.
3. **Check Logs**: Look for any error messages related to the specific pair.

### Strategy Parameters Not Taking Effect

**Symptoms:**
- Bot behavior doesn't match configured strategy parameters

**Possible Solutions:**
1. **Config Hot-Reload**: Verify the configuration was properly reloaded (check logs).
2. **Parameter Scope**: Some parameters might be pair-specific. Check if you've set them at the correct level.
3. **Restart Bot**: Some core parameters might require a bot restart to take effect.

## Trading Issues

### No Signals Generated

**Symptoms:**
- Bot is running but not generating any trade signals

**Possible Solutions:**
1. **Market Conditions**: The strategy conditions might not be met in current market conditions.
2. **Check Indicator Calculation**: Verify indicators are being calculated correctly.
3. **Signal Interval**: Check if `min_signal_interval_minutes` is set too high.
4. **Buffer Time**: If using buffer time, the crossover might need to persist for several candles.

### Orders Failing

**Symptoms:**
- Signals generated but orders not placed
- Error messages when placing orders

**Possible Solutions:**
1. **Insufficient Funds**: Ensure your account has sufficient balance.
2. **Leverage/Margin Settings**: Check if the configured leverage is allowed for the pair.
3. **Minimum Notional**: The calculated position size might be below the minimum notional value required by Binance.
4. **Symbol Precision**: Order quantities might not meet the symbol's precision requirements.

### Unexpected Position Sizes

**Symptoms:**
- Position sizes differ from what you expected based on configuration

**Possible Solutions:**
1. **Check Margin Settings**: Verify `margin_usdt` or `margin_coin` settings.
2. **Leverage Configuration**: Confirm the leverage setting is applied correctly.
3. **Price Impact**: For market orders, the execution price might differ from the signal price.

## Monitoring Issues

### Prometheus Metrics Not Available

**Symptoms:**
- Cannot access metrics at http://your-server:8000/metrics

**Possible Solutions:**
1. **Port Configuration**: Verify the `prometheus_port` setting.
2. **Firewall Settings**: Ensure the configured port is accessible.
3. **Metrics Enabled**: Check that monitoring is enabled in the configuration.

## Logging Issues

### Missing or Insufficient Logs

**Symptoms:**
- Not enough information in logs to diagnose issues

**Possible Solutions:**
1. **Log Level**: Set `logging.level` to "DEBUG" for more detailed logs.
2. **Log File Configuration**: Ensure the log file path is writable if using file logging.

## Performance Issues

### High CPU Usage

**Symptoms:**
- Bot process consuming excessive CPU resources

**Possible Solutions:**
1. **Too Many Pairs**: Reduce the number of active trading pairs.
2. **Timeframe Overload**: Limit the number of indicator timeframes being processed.
3. **Indicator Calculation**: Some indicators might be computationally expensive.

### Memory Leaks

**Symptoms:**
- Increasing memory usage over time

**Possible Solutions:**
1. **Buffer Size**: Check if k-line buffers are growing unbounded.
2. **Restart Periodically**: Consider implementing a scheduled restart.

## Testnet vs. Mainnet

### Testing on Testnet

Before trading with real funds, test thoroughly on Binance Futures Testnet:

1. Create a Testnet account at https://testnet.binancefuture.com/
2. Generate API keys for the Testnet
3. Configure the bot with `testnet: true` in the API section
4. Verify all functionality works as expected

### Moving to Mainnet

When ready to trade with real funds:

1. Generate new API keys on the main Binance platform
2. Update your configuration with the new keys
3. Set `testnet: false` or remove the testnet parameter
4. Start with small position sizes to verify everything works correctly

## Getting Help

If you encounter issues not covered in this guide:

1. Check the detailed logs for error messages
2. Review the [Binance API documentation](https://binance-docs.github.io/apidocs/futures/en/)
3. Open an issue on the project's GitHub repository with:
   - Detailed description of the issue
   - Relevant log excerpts
   - Configuration (with sensitive information redacted)
   - Steps to reproduce the problem
