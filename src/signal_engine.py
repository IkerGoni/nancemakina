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
        """Retrieves configuration for a specific trading pair, falling back to global defaults when needed."""
        config = self.config_manager.get_config()
        global_v1_config = config.get("global_settings", {}).get("v1_strategy", {})
        pair_config = config.get("pairs", {}).get(config_symbol, {})
        
        # Merge global defaults with pair-specific settings
        merged_config = {
            "sma_short_period": pair_config.get("sma_short_period", global_v1_config.get("sma_short_period", 21)),
            "sma_long_period": pair_config.get("sma_long_period", global_v1_config.get("sma_long_period", 200)),
            "min_signal_interval_minutes": pair_config.get("min_signal_interval_minutes", global_v1_config.get("min_signal_interval_minutes", 15)),
            "tp_sl_ratio": pair_config.get("tp_sl_ratio", global_v1_config.get("tp_sl_ratio", 3.0)),
            "leverage": pair_config.get("leverage", global_v1_config.get("default_leverage", 10)),
            "margin_mode": pair_config.get("margin_mode", global_v1_config.get("margin_mode", "ISOLATED")),
            "contract_type": pair_config.get("contract_type", "USD_M"),  # Default to USD-M
            "fallback_sl_percentage": pair_config.get("fallback_sl_percentage", global_v1_config.get("fallback_sl_percentage", 0.02))  # Add fallback SL percentage
        }
        
        # Handle margin amount based on contract type
        if merged_config["contract_type"] == "USD_M":
            merged_config["margin_usdt"] = pair_config.get("margin_usdt", global_v1_config.get("default_margin_usdt", 50.0))
        else:  # COIN_M
            merged_config["margin_coin"] = pair_config.get("margin_coin", global_v1_config.get("default_margin_coin", 0.01))
            
        return merged_config

    def _find_recent_pivot(self, klines_df: pd.DataFrame, lookback: int = 10, direction: TradeDirection = TradeDirection.LONG) -> Optional[float]:
        """Finds the most recent significant pivot low (for LONG) or high (for SHORT)."""
        logger.debug(f"_find_recent_pivot called with lookback={lookback}, direction={direction.value}")
        
        if klines_df is None or klines_df.empty or len(klines_df) < 3:
            logger.debug(f"Insufficient data for pivot: df exists: {klines_df is not None}, empty: {klines_df.empty if klines_df is not None else True}, len: {len(klines_df) if klines_df is not None else 0}")
            return None

        # Ensure we only look at a recent window of closed candles for pivots
        # The signal candle itself is usually the last one, so we look before it.
        relevant_klines = klines_df[klines_df["is_closed"]].iloc[-lookback-1:-1] # Look at N closed candles before the signal candle
        logger.debug(f"Found {len(relevant_klines)} closed candles in the lookback window")
        
        if len(relevant_klines) < 3:
             relevant_klines = klines_df[klines_df["is_closed"]].iloc[-len(klines_df):-1] # use all available if not enough
             logger.debug(f"Using all available {len(relevant_klines)} closed candles")
             if len(relevant_klines) < 3:
                logger.debug("Still insufficient data for pivot calculation (need at least 3 candles)")
                return None

        if direction == TradeDirection.LONG:
            # Find pivot lows: low[i] < low[i-1] and low[i] < low[i+1]
            # For simplicity, let\"s find the minimum low in the lookback window as a proxy for recent support
            # A more robust pivot detection would use scipy.signal.find_peaks or similar
            min_low = relevant_klines["low"].min()
            logger.debug(f"For LONG direction, found pivot low at {min_low}")
            return min_low
        else: # TradeDirection.SHORT
            # Find pivot highs: high[i] > high[i-1] and high[i] > high[i+1]
            # For simplicity, let\"s find the maximum high in the lookback window as a proxy for recent resistance
            max_high = relevant_klines["high"].max()
            logger.debug(f"For SHORT direction, found pivot high at {max_high}")
            return max_high

    async def check_signal(self, api_symbol: str, config_symbol: str) -> Optional[TradeSignal]:
        """Checks for a V1 SMA crossover signal for the given API symbol (e.g., BTCUSDT)."""
        # DEBUG: Entry to check_signal
        logger.debug(f"Entering check_signal for {api_symbol}, timestamp: {time.time()}")
        
        pair_config = self._get_pair_specific_config(config_symbol)
        if not pair_config: 
            logger.debug(f"No configuration found for {config_symbol} in SignalEngineV1")
            return None

        # V1 strategy uses 1-minute timeframe for signals
        signal_interval = "1m"
        df = self.data_processor.get_indicator_dataframe(api_symbol, signal_interval)

        if df is None or df.empty or len(df) < 2:
            logger.debug(f"Not enough data for {api_symbol} {signal_interval} to generate signal. df exists: {df is not None}, empty: {df.empty if df is not None else True}, len: {len(df) if df is not None else 0}")
            return None

        # Get the latest two candles for crossover detection
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        sma_short_col = "sma_short"
        sma_long_col = "sma_long"

        if not all(col in latest.index and col in previous.index for col in [sma_short_col, sma_long_col, "close", "is_closed"]):
            logger.debug(f"SMA data not available for {api_symbol} {signal_interval}. Columns in latest: {list(latest.index)}")
            return None
        
        # Ensure latest candle data is present and SMAs are calculated
        if pd.isna(latest[sma_short_col]) or pd.isna(latest[sma_long_col]) or \
           pd.isna(previous[sma_short_col]) or pd.isna(previous[sma_long_col]):
            logger.debug(f"SMA values are NA for {api_symbol} {signal_interval}. Latest: {latest[sma_short_col]}, {latest[sma_long_col]}. Previous: {previous[sma_short_col]}, {previous[sma_long_col]}")
            return None

        # DEBUG: Log SMA values for debugging
        logger.debug(f"SMA values for {api_symbol}: Latest short={latest[sma_short_col]}, long={latest[sma_long_col]}. Previous short={previous[sma_short_col]}, long={previous[sma_long_col]}")

        # --- Significance Filter (Time-based) ---
        min_interval_seconds = pair_config["min_signal_interval_minutes"] * 60
        current_time = time.time()
        if api_symbol in self.last_signal_time and (current_time - self.last_signal_time[api_symbol]) < min_interval_seconds:
            logger.debug(f"Signal for {api_symbol} too soon. Last signal at {self.last_signal_time[api_symbol]}, current: {current_time}, min_interval: {min_interval_seconds}")
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

        # DEBUG: Log crossover detection
        logger.debug(f"Crossover detection for {api_symbol}: crossed_up={crossed_up}, crossed_down={crossed_down}")

        signal_direction: Optional[TradeDirection] = None
        if crossed_up:
            signal_direction = TradeDirection.LONG
            logger.debug(f"LONG signal detected for {api_symbol}")
        elif crossed_down:
            signal_direction = TradeDirection.SHORT
            logger.debug(f"SHORT signal detected for {api_symbol}")
        else:
            logger.debug(f"No crossover detected for {api_symbol}")
            return None # No crossover

        # TEMPORARY DEBUG CODE: Bypass SL/TP calculation for testing crossovers
        if signal_direction:
            logger.info(f"TEMP DEBUG: Crossover ({signal_direction.value}) detected for {api_symbol} at entry {latest['close']}.")
            logger.info(f"TEMP DEBUG: Prev SMAs: S={previous[sma_short_col]}, L={previous[sma_long_col]}. Curr SMAs: S={latest[sma_short_col]}, L={latest[sma_long_col]}")
            
            # Construct a minimal Kline for the signal
            temp_kline = Kline(
                timestamp=int(latest.name),
                open=latest["open"],
                high=latest["high"],
                low=latest["low"],
                close=latest["close"],
                volume=latest["volume"],
                is_closed=latest["is_closed"],
                symbol=api_symbol,
                interval=signal_interval
            )
            
            # Use dummy SL/TP values based on direction
            dummy_sl = latest["close"] * (0.99 if signal_direction == TradeDirection.LONG else 1.01)  # 1% away
            dummy_tp = latest["close"] * (1.01 if signal_direction == TradeDirection.LONG else 0.99)  # 1% away
            
            return TradeSignal(
                symbol=api_symbol,
                config_symbol=config_symbol,
                contract_type=pair_config["contract_type"],
                direction=signal_direction,
                entry_price=latest["close"],
                stop_loss_price=dummy_sl,
                take_profit_price=dummy_tp,
                strategy_name="V1_SMA_Crossover",
                signal_kline=temp_kline,
                details={
                    "temp_debug": True,
                    "sma_short_at_signal": latest[sma_short_col],
                    "sma_long_at_signal": latest[sma_long_col],
                    "sma_short_previous": previous[sma_short_col],
                    "sma_long_previous": previous[sma_long_col]
                }
            )
        
        # Original SL/TP calculation
        entry_price = latest["close"] # Current close price as entry
        logger.debug(f"Entry price for {api_symbol}: {entry_price}")
        
        stop_loss_price: Optional[float] = None
        take_profit_price: Optional[float] = None
        tp_sl_ratio = pair_config["tp_sl_ratio"]
        fallback_sl_percentage = pair_config["fallback_sl_percentage"]  # Get fallback SL percentage

        # Standard pivot lookback for production use
        pivot_lookback = 30
        logger.debug(f"Using standard pivot_lookback={pivot_lookback}")

        if signal_direction == TradeDirection.LONG:
            pivot_low = self._find_recent_pivot(df, lookback=pivot_lookback, direction=TradeDirection.LONG)
            logger.debug(f"LONG signal - find_recent_pivot result for {api_symbol}: pivot_low={pivot_low}")
            
            if pivot_low is not None:
                # Add a small buffer to SL, e.g., a few ticks or a percentage
                # For simplicity, direct use for now. Buffer can be added from config.
                stop_loss_price = pivot_low 
                risk = entry_price - stop_loss_price
                logger.debug(f"Calculated risk for LONG {api_symbol}: entry={entry_price}, pivot_low={pivot_low}, risk={risk}")
                
                if risk <= 0: # Invalid SL (e.g. pivot_low >= entry_price)
                    logger.warning(f"Invalid SL for LONG {api_symbol}: entry={entry_price}, pivot_low={pivot_low}, risk={risk}. Using fallback SL.")
                    # Use fallback SL instead of skipping the signal
                    stop_loss_price = entry_price * (1 - fallback_sl_percentage)
                    risk = entry_price - stop_loss_price
                    logger.warning(f"Using fallback %-based SL for LONG {api_symbol}. New SL: {stop_loss_price}, risk: {risk}")
                take_profit_price = entry_price + (risk * tp_sl_ratio)
                logger.debug(f"Calculated TP for LONG {api_symbol}: TP={take_profit_price} (risk={risk} * ratio={tp_sl_ratio})")
            else:
                logger.warning(f"Could not determine pivot low for LONG SL for {api_symbol}. Using fallback SL.")
                # Implement fallback SL calculation
                stop_loss_price = entry_price * (1 - fallback_sl_percentage)
                risk = entry_price - stop_loss_price
                take_profit_price = entry_price + (risk * tp_sl_ratio)
                logger.warning(f"Using fallback %-based SL for LONG {api_symbol}. SL: {stop_loss_price}, TP: {take_profit_price}")
        else: # TradeDirection.SHORT
            pivot_high = self._find_recent_pivot(df, lookback=pivot_lookback, direction=TradeDirection.SHORT)
            logger.debug(f"SHORT signal - find_recent_pivot result for {api_symbol}: pivot_high={pivot_high}")
            
            if pivot_high is not None:
                stop_loss_price = pivot_high
                risk = stop_loss_price - entry_price
                logger.debug(f"Calculated risk for SHORT {api_symbol}: entry={entry_price}, pivot_high={pivot_high}, risk={risk}")
                
                if risk <= 0: # Invalid SL (e.g. pivot_high <= entry_price)
                    logger.warning(f"Invalid SL for SHORT {api_symbol}: entry={entry_price}, pivot_high={pivot_high}, risk={risk}. Using fallback SL.")
                    # Use fallback SL instead of skipping the signal
                    stop_loss_price = entry_price * (1 + fallback_sl_percentage)
                    risk = stop_loss_price - entry_price
                    logger.warning(f"Using fallback %-based SL for SHORT {api_symbol}. New SL: {stop_loss_price}, risk: {risk}")
                take_profit_price = entry_price - (risk * tp_sl_ratio)
                logger.debug(f"Calculated TP for SHORT {api_symbol}: TP={take_profit_price} (risk={risk} * ratio={tp_sl_ratio})")
            else:
                logger.warning(f"Could not determine pivot high for SHORT SL for {api_symbol}. Using fallback SL.")
                # Implement fallback SL calculation
                stop_loss_price = entry_price * (1 + fallback_sl_percentage)
                risk = stop_loss_price - entry_price
                take_profit_price = entry_price - (risk * tp_sl_ratio)
                logger.warning(f"Using fallback %-based SL for SHORT {api_symbol}. SL: {stop_loss_price}, TP: {take_profit_price}")

        if stop_loss_price is None or take_profit_price is None:
            logger.warning(f"Failed to calculate SL/TP for {api_symbol}. Skipping signal.")
            return None
        
        # Final validation of calculated SL/TP values
        if (signal_direction == TradeDirection.LONG and (stop_loss_price >= entry_price or take_profit_price <= entry_price)) or \
           (signal_direction == TradeDirection.SHORT and (stop_loss_price <= entry_price or take_profit_price >= entry_price)):
            logger.warning(f"Calculated SL/TP invalid for {api_symbol}: Direction={signal_direction.value}, Entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}. Skipping signal.")
            return None

        # Ensure we haven't issued a signal for this symbol recently
        current_time = int(time.time())
        min_interval_seconds = pair_config["min_signal_interval_minutes"] * 60
        if api_symbol in self.last_signal_time and (current_time - self.last_signal_time[api_symbol]) < min_interval_seconds:
            logger.debug(f"Signal for {api_symbol} too soon. Last signal: {self.last_signal_time[api_symbol]}, current: {current_time}")
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

        logger.debug(f"Successfully created TradeSignal for {api_symbol}: {signal_direction.value} at {entry_price}")
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

