import logging
import time
import pandas as pd
from typing import Optional, Dict, Any, Tuple

from src.config_loader import ConfigManager
from src.data_processor import DataProcessor
from src.models import TradeSignal, TradeDirection, Kline
from src.connectors import convert_symbol_to_api_format, convert_symbol_to_ws_format # For consistency

logger = logging.getLogger(__name__)

class SignalEngineV1:
    def __init__(self, config_manager: ConfigManager, data_processor: DataProcessor):
        self.config_manager = config_manager
        self.data_processor = data_processor
        self.last_signal_time: Dict[str, float] = {} # Stores timestamp of the last signal for each pair

    def _get_pair_specific_config(self, config_symbol: str) -> Dict[str, Any]:
        config = self.config_manager.get_config()
        pair_configs = config.get("pairs", {})
        global_v1_config = config.get("global_settings", {}).get("v1_strategy", {})
        
        pair_specific_overrides = {}
        for cfg_sym, details in pair_configs.items():
            if cfg_sym.upper() == config_symbol.upper(): # Match config_symbol (e.g. BTC_USDT)
                pair_specific_overrides = details
                break
        
        return {
            "sma_short_period": pair_specific_overrides.get("sma_short_period", global_v1_config.get("sma_short_period", 21)),
            "sma_long_period": pair_specific_overrides.get("sma_long_period", global_v1_config.get("sma_long_period", 200)),
            "min_signal_interval_minutes": pair_specific_overrides.get("min_signal_interval_minutes", global_v1_config.get("min_signal_interval_minutes", 15)),
            "tp_sl_ratio": pair_specific_overrides.get("tp_sl_ratio", global_v1_config.get("tp_sl_ratio", 3.0)),
            "contract_type": pair_specific_overrides.get("contract_type", "USDT_M") # Default to USDT_M
        }

    def _find_recent_pivot(self, klines_df: pd.DataFrame, lookback: int = 10, direction: TradeDirection = TradeDirection.LONG) -> Optional[float]:
        """Finds the most recent significant pivot low (for LONG) or high (for SHORT)."""
        if klines_df is None or klines_df.empty or len(klines_df) < 3:
            return None

        # Ensure we only look at a recent window of closed candles for pivots
        # The signal candle itself is usually the last one, so we look before it.
        relevant_klines = klines_df[klines_df["is_closed"]].iloc[-lookback-1:-1] # Look at N closed candles before the signal candle
        if len(relevant_klines) < 3:
             relevant_klines = klines_df[klines_df["is_closed"]].iloc[-len(klines_df):-1] # use all available if not enough
             if len(relevant_klines) < 3:
                return None

        if direction == TradeDirection.LONG:
            # Find pivot lows: low[i] < low[i-1] and low[i] < low[i+1]
            # For simplicity, let\"s find the minimum low in the lookback window as a proxy for recent support
            # A more robust pivot detection would use scipy.signal.find_peaks or similar
            min_low = relevant_klines["low"].min()
            return min_low
        else: # TradeDirection.SHORT
            # Find pivot highs: high[i] > high[i-1] and high[i] > high[i+1]
            # For simplicity, let\"s find the maximum high in the lookback window as a proxy for recent resistance
            max_high = relevant_klines["high"].max()
            return max_high

    async def check_signal(self, api_symbol: str, config_symbol: str) -> Optional[TradeSignal]:
        """Checks for a V1 SMA crossover signal for the given API symbol (e.g., BTCUSDT)."""
        pair_config = self._get_pair_specific_config(config_symbol)
        if not pair_config: 
            # logger.debug(f"No configuration found for {config_symbol} in SignalEngineV1")
            return None

        # V1 strategy uses 1-minute timeframe for signals
        signal_interval = "1m"
        df = self.data_processor.get_indicator_dataframe(api_symbol, signal_interval)

        if df is None or df.empty or len(df) < 2:
            # logger.debug(f"Not enough data for {api_symbol} {signal_interval} to generate signal.")
            return None

        # Get the latest two candles for crossover detection
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        sma_short_col = "sma_short"
        sma_long_col = "sma_long"

        if not all(col in latest.index and col in previous.index for col in [sma_short_col, sma_long_col, "close", "is_closed"]):
            # logger.debug(f"SMA data not available for {api_symbol} {signal_interval}.")
            return None
        
        # Ensure latest candle data is present and SMAs are calculated
        if pd.isna(latest[sma_short_col]) or pd.isna(latest[sma_long_col]) or \
           pd.isna(previous[sma_short_col]) or pd.isna(previous[sma_long_col]):
            # logger.debug(f"SMA values are NA for {api_symbol} {signal_interval}. Latest: {latest[sma_short_col]}, {latest[sma_long_col]}. Previous: {previous[sma_short_col]}, {previous[sma_long_col]}")
            return None

        # --- Significance Filter (Time-based) ---
        min_interval_seconds = pair_config["min_signal_interval_minutes"] * 60
        current_time = time.time()
        if api_symbol in self.last_signal_time and (current_time - self.last_signal_time[api_symbol]) < min_interval_seconds:
            # logger.debug(f"Signal for {api_symbol} too soon. Last signal at {self.last_signal_time[api_symbol]}, current: {current_time}")
            return None

        # --- SMA Crossover Detection ---
        # Ensure we are checking on a closed candle or a very recent candle
        # The signal is based on the state at the close of the \"previous\" candle that caused the crossover, 
        # and the \"latest\" candle confirms it.
        # Let\"s assume the signal is valid if the crossover happened on the \"latest\" candle compared to \"previous\".
        
        crossed_up = (previous[sma_short_col] <= previous[sma_long_col] and
                      latest[sma_short_col] > latest[sma_long_col])
        crossed_down = (previous[sma_short_col] >= previous[sma_long_col] and
                        latest[sma_short_col] < latest[sma_long_col])

        signal_direction: Optional[TradeDirection] = None
        if crossed_up:
            signal_direction = TradeDirection.LONG
        elif crossed_down:
            signal_direction = TradeDirection.SHORT
        else:
            return None # No crossover

        # --- SL/TP Calculation ---
        entry_price = latest["close"] # Current close price as entry
        stop_loss_price: Optional[float] = None
        take_profit_price: Optional[float] = None
        tp_sl_ratio = pair_config["tp_sl_ratio"]

        # Use a lookback of N candles for pivot detection, e.g., 20-50 candles for 1m chart
        # The klines for pivot detection should be from the main DataFrame `df`
        pivot_lookback = 30 # Number of 1-minute candles to look back for pivot

        if signal_direction == TradeDirection.LONG:
            pivot_low = self._find_recent_pivot(df, lookback=pivot_lookback, direction=TradeDirection.LONG)
            if pivot_low is not None:
                # Add a small buffer to SL, e.g., a few ticks or a percentage
                # For simplicity, direct use for now. Buffer can be added from config.
                stop_loss_price = pivot_low 
                risk = entry_price - stop_loss_price
                if risk <= 0: # Invalid SL (e.g. pivot_low >= entry_price)
                    logger.warning(f"Invalid SL for LONG {api_symbol}: entry={entry_price}, pivot_low={pivot_low}. Skipping signal.")
                    return None
                take_profit_price = entry_price + (risk * tp_sl_ratio)
            else:
                logger.warning(f"Could not determine pivot low for LONG SL for {api_symbol}. Skipping signal.")
                return None
        else: # TradeDirection.SHORT
            pivot_high = self._find_recent_pivot(df, lookback=pivot_lookback, direction=TradeDirection.SHORT)
            if pivot_high is not None:
                stop_loss_price = pivot_high
                risk = stop_loss_price - entry_price
                if risk <= 0: # Invalid SL (e.g. pivot_high <= entry_price)
                    logger.warning(f"Invalid SL for SHORT {api_symbol}: entry={entry_price}, pivot_high={pivot_high}. Skipping signal.")
                    return None
                take_profit_price = entry_price - (risk * tp_sl_ratio)
            else:
                logger.warning(f"Could not determine pivot high for SHORT SL for {api_symbol}. Skipping signal.")
                return None

        if stop_loss_price is None or take_profit_price is None:
            logger.warning(f"Failed to calculate SL/TP for {api_symbol}. Skipping signal.")
            return None
        
        # Ensure SL and TP are not the same as entry or inverted
        if (signal_direction == TradeDirection.LONG and (stop_loss_price >= entry_price or take_profit_price <= entry_price)) or \
           (signal_direction == TradeDirection.SHORT and (stop_loss_price <= entry_price or take_profit_price >= entry_price)):
            logger.warning(f"Calculated SL/TP invalid for {api_symbol}: E={entry_price}, SL={stop_loss_price}, TP={take_profit_price}. Skipping signal.")
            return None

        self.last_signal_time[api_symbol] = current_time
        logger.info(f"Generated signal for {api_symbol}: {signal_direction.value} at {entry_price}, SL={stop_loss_price}, TP={take_profit_price}")

        # Construct signal kline from the latest data point in the DataFrame
        signal_kline_data = latest.to_dict()
        # Ensure all fields required by Kline model are present, add defaults if necessary
        signal_kline_obj = Kline(
            timestamp=int(latest.name), # timestamp is the index
            open=signal_kline_data.get("open"),
            high=signal_kline_data.get("high"),
            low=signal_kline_data.get("low"),
            close=signal_kline_data.get("close"),
            volume=signal_kline_data.get("volume"),
            is_closed=signal_kline_data.get("is_closed", True), # Assume closed if it triggered signal logic
            symbol=api_symbol,
            interval=signal_interval
        )

        return TradeSignal(
            symbol=api_symbol,
            config_symbol=config_symbol,
            contract_type=pair_config["contract_type"],
            direction=signal_direction,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            strategy_name="V1_SMA_Crossover",
            signal_kline=signal_kline_obj,
            details={
                "sma_short_at_signal": latest[sma_short_col],
                "sma_long_at_signal": latest[sma_long_col],
                "sma_short_previous": previous[sma_short_col],
                "sma_long_previous": previous[sma_long_col],
                "pivot_used_for_sl": pivot_low if signal_direction == TradeDirection.LONG else pivot_high
            }
        )

