import logging
import time
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List

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
        
        # Merge core parameters from global defaults with pair-specific settings
        merged_config = {
            "sma_short_period": pair_config.get("sma_short_period", global_v1_config.get("sma_short_period", 21)),
            "sma_long_period": pair_config.get("sma_long_period", global_v1_config.get("sma_long_period", 200)),
            "min_signal_interval_minutes": pair_config.get("min_signal_interval_minutes", global_v1_config.get("min_signal_interval_minutes", 15)),
            "tp_sl_ratio": pair_config.get("tp_sl_ratio", global_v1_config.get("tp_sl_ratio", 3.0)),
            "leverage": pair_config.get("leverage", global_v1_config.get("default_leverage", 10)),
            "margin_mode": pair_config.get("margin_mode", global_v1_config.get("margin_mode", "ISOLATED")),
            "contract_type": pair_config.get("contract_type", "USD_M"),  # Default to USD-M
            "fallback_sl_percentage": pair_config.get("fallback_sl_percentage", global_v1_config.get("fallback_sl_percentage", 0.02)),
        }
        
        # Handle margin amount based on contract type
        if merged_config["contract_type"] == "USD_M":
            merged_config["margin_usdt"] = pair_config.get("margin_usdt", global_v1_config.get("default_margin_usdt", 50.0))
        else:  # COIN_M
            merged_config["margin_coin"] = pair_config.get("margin_coin", global_v1_config.get("default_margin_coin", 0.01))
        
        # --- Merge Filter Configurations ---
        # Get global filter configs with defaults
        global_filters = global_v1_config.get("filters", {})
        pair_filters = pair_config.get("filters", {})
        
        # Initialize filters structure
        merged_config["filters"] = {
            "buffer_time": {
                "enabled": False,
                "candles": 2
            },
            "volume": {
                "enabled": False,
                "lookback_periods": 20,
                "min_threshold": 1.5
            },
            "volatility": {
                "enabled": False,
                "indicator": "atr",
                "lookback_periods": 14,
                "min_threshold": 0.002,
                "max_threshold": 0.05
            }
        }
        
        # Buffer Time Filter
        buffer_time_global = global_filters.get("buffer_time", {})
        buffer_time_pair = pair_filters.get("buffer_time", {})
        merged_config["filters"]["buffer_time"] = {
            "enabled": buffer_time_pair.get("enabled", buffer_time_global.get("enabled", False)),
            "candles": buffer_time_pair.get("candles", buffer_time_global.get("candles", 2))
        }
        
        # Volume Filter
        volume_global = global_filters.get("volume", {})
        volume_pair = pair_filters.get("volume", {})
        merged_config["filters"]["volume"] = {
            "enabled": volume_pair.get("enabled", volume_global.get("enabled", False)),
            "lookback_periods": volume_pair.get("lookback_periods", volume_global.get("lookback_periods", 20)),
            "min_threshold": volume_pair.get("min_threshold", volume_global.get("min_threshold", 1.5))
        }
        
        # Volatility Filter
        volatility_global = global_filters.get("volatility", {})
        volatility_pair = pair_filters.get("volatility", {})
        merged_config["filters"]["volatility"] = {
            "enabled": volatility_pair.get("enabled", volatility_global.get("enabled", False)),
            "indicator": volatility_pair.get("indicator", volatility_global.get("indicator", "atr")),
            "lookback_periods": volatility_pair.get("lookback_periods", volatility_global.get("lookback_periods", 14)),
            "min_threshold": volatility_pair.get("min_threshold", volatility_global.get("min_threshold", 0.002)),
            "max_threshold": volatility_pair.get("max_threshold", volatility_global.get("max_threshold", 0.05))
        }
        
        # --- Merge SL Options ---
        global_sl_options = global_v1_config.get("sl_options", {})
        pair_sl_options = pair_config.get("sl_options", {})
        
        # Initialize SL options structure with default values
        merged_config["sl_options"] = {
            "primary_type": pair_sl_options.get("primary_type", global_sl_options.get("primary_type", "pivot")),
            "pivot_settings": {
                "lookback_candles": pair_sl_options.get("pivot_settings", {}).get("lookback_candles", 
                                  global_sl_options.get("pivot_settings", {}).get("lookback_candles", 30))
            },
            "percentage_settings": {
                "value": pair_sl_options.get("percentage_settings", {}).get("value", 
                        global_sl_options.get("percentage_settings", {}).get("value", 
                        merged_config["fallback_sl_percentage"]))  # Default to fallback_sl_percentage if not set
            },
            "enable_fallback": pair_sl_options.get("enable_fallback", global_sl_options.get("enable_fallback", True)),
            "fallback_type": pair_sl_options.get("fallback_type", global_sl_options.get("fallback_type", "percentage"))
        }
        
        # --- Merge TP Options ---
        global_tp_options = global_v1_config.get("tp_options", {})
        pair_tp_options = pair_config.get("tp_options", {})
        
        # Initialize TP options structure with default values
        merged_config["tp_options"] = {
            "primary_type": pair_tp_options.get("primary_type", global_tp_options.get("primary_type", "rr_ratio")),
            "rr_ratio_settings": {
                "value": pair_tp_options.get("rr_ratio_settings", {}).get("value", 
                        global_tp_options.get("rr_ratio_settings", {}).get("value", 
                        merged_config["tp_sl_ratio"]))  # Default to tp_sl_ratio if not set
            },
            "fixed_percentage_settings": {
                "value": pair_tp_options.get("fixed_percentage_settings", {}).get("value", 
                        global_tp_options.get("fixed_percentage_settings", {}).get("value", 0.06))  # Default 6%
            }
        }
        
        logger.debug(f"Merged config for {config_symbol}: {merged_config}")
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

    def _passes_buffer_time_filter(self, df: pd.DataFrame, buffer_config: Dict[str, Any], direction: TradeDirection) -> bool:
        """
        Checks if the crossover has been maintained for the specified number of candles.
        
        Args:
            df: DataFrame containing price and indicator data
            buffer_config: Configuration for buffer time filter
            direction: Direction of the trade (LONG or SHORT)
            
        Returns:
            True if filter passes, False if it fails
        """
        if not buffer_config.get("enabled", False):
            logger.debug("Buffer time filter not enabled, passing")
            return True
            
        required_candles = buffer_config.get("candles", 2)
        logger.debug(f"Buffer time filter checking for {required_candles} candles of consistent crossover")
        
        if len(df) < required_candles + 1:  # +1 for the candle before crossover
            logger.warning(f"Not enough candles to check buffer time filter. Need at least {required_candles + 1}, got {len(df)}")
            return False
            
        # Get the required number of most recent candles plus one before
        relevant_candles = df.iloc[-(required_candles+1):]
        sma_short_col = "sma_short"
        sma_long_col = "sma_long"
        
        # Verify all required candles have the crossover maintained
        if direction == TradeDirection.LONG:
            # First candle should show SMA short <= SMA long (not yet crossed)
            # All subsequent candles should show SMA short > SMA long (crossed and maintained)
            first_candle = relevant_candles.iloc[0]
            subsequent_candles = relevant_candles.iloc[1:]
            
            if not (first_candle[sma_short_col] <= first_candle[sma_long_col]):
                logger.debug("Buffer time filter failed: first candle already shows crossover up")
                return False
                
            for i, candle in subsequent_candles.iterrows():
                if not (candle[sma_short_col] > candle[sma_long_col]):
                    logger.debug(f"Buffer time filter failed: candle at index {i} doesn't maintain crossover up")
                    return False
                    
        else:  # TradeDirection.SHORT
            # First candle should show SMA short >= SMA long (not yet crossed down)
            # All subsequent candles should show SMA short < SMA long (crossed down and maintained)
            first_candle = relevant_candles.iloc[0]
            subsequent_candles = relevant_candles.iloc[1:]
            
            if not (first_candle[sma_short_col] >= first_candle[sma_long_col]):
                logger.debug("Buffer time filter failed: first candle already shows crossover down")
                return False
                
            for i, candle in subsequent_candles.iterrows():
                if not (candle[sma_short_col] < candle[sma_long_col]):
                    logger.debug(f"Buffer time filter failed: candle at index {i} doesn't maintain crossover down")
                    return False
        
        logger.debug(f"Buffer time filter passed: crossover maintained for {required_candles} candles")
        return True
        
    def _calculate_percentage_sl(self, entry_price: float, direction: TradeDirection, percentage_value: float) -> float:
        """
        Calculate stop loss based on a percentage from entry price.
        
        Args:
            entry_price: Entry price of the trade
            direction: Direction of the trade (LONG or SHORT)
            percentage_value: Percentage to use for calculation (e.g., 0.02 for 2%)
            
        Returns:
            Calculated stop loss price
        """
        logger.debug(f"Calculating percentage-based SL with entry={entry_price}, direction={direction.value}, percentage={percentage_value}")
        
        if direction == TradeDirection.LONG:
            # For long positions, SL is below entry
            stop_loss_price = entry_price * (1 - percentage_value)
        else:  # TradeDirection.SHORT
            # For short positions, SL is above entry
            stop_loss_price = entry_price * (1 + percentage_value)
            
        logger.debug(f"Calculated percentage-based SL: {stop_loss_price}")
        return stop_loss_price
        
    def _calculate_stop_loss(self, entry_price: float, direction: TradeDirection, df: pd.DataFrame, sl_config: Dict[str, Any]) -> Optional[float]:
        """
        Calculate stop loss price based on configuration.
        
        Args:
            entry_price: Entry price of the trade
            direction: Direction of the trade (LONG or SHORT)
            df: DataFrame containing price data for pivot calculation
            sl_config: Configuration for stop loss calculation
            
        Returns:
            Calculated stop loss price or None if calculation fails
        """
        primary_type = sl_config.get("primary_type", "pivot")
        logger.debug(f"Calculating stop loss with primary_type={primary_type}, direction={direction.value}")
        
        stop_loss_price = None
        
        # Primary calculation method
        if primary_type == "pivot":
            # Use pivot point method
            lookback_candles = sl_config.get("pivot_settings", {}).get("lookback_candles", 30)
            logger.debug(f"Using pivot method with lookback_candles={lookback_candles}")
            
            pivot_value = self._find_recent_pivot(df, lookback=lookback_candles, direction=direction)
            
            if pivot_value is not None:
                stop_loss_price = pivot_value
                # Validate pivot-based SL
                if (direction == TradeDirection.LONG and stop_loss_price >= entry_price) or \
                   (direction == TradeDirection.SHORT and stop_loss_price <= entry_price):
                    logger.warning(f"Invalid pivot-based SL: direction={direction.value}, entry={entry_price}, SL={stop_loss_price}")
                    stop_loss_price = None
            else:
                logger.warning(f"Failed to find valid pivot point for SL calculation")
        
        elif primary_type == "percentage":
            # Use percentage method
            percentage_value = sl_config.get("percentage_settings", {}).get("value", 0.02)
            logger.debug(f"Using percentage method with value={percentage_value}")
            
            stop_loss_price = self._calculate_percentage_sl(entry_price, direction, percentage_value)
        
        else:
            logger.warning(f"Unknown SL primary_type: {primary_type}")
        
        # Fallback if primary method failed
        if stop_loss_price is None and sl_config.get("enable_fallback", True):
            fallback_type = sl_config.get("fallback_type", "percentage")
            logger.debug(f"Primary SL calculation failed, using fallback_type={fallback_type}")
            
            if fallback_type == "percentage":
                percentage_value = sl_config.get("percentage_settings", {}).get("value", 0.02)
                logger.debug(f"Using fallback percentage method with value={percentage_value}")
                
                stop_loss_price = self._calculate_percentage_sl(entry_price, direction, percentage_value)
            else:
                logger.warning(f"Unknown fallback_type: {fallback_type}")
        
        # Final validation
        if stop_loss_price is not None:
            if (direction == TradeDirection.LONG and stop_loss_price >= entry_price) or \
               (direction == TradeDirection.SHORT and stop_loss_price <= entry_price):
                logger.warning(f"Invalid final SL: direction={direction.value}, entry={entry_price}, SL={stop_loss_price}")
                return None
        
        logger.debug(f"Final calculated SL: {stop_loss_price}")
        return stop_loss_price
        
    def _calculate_take_profit(self, entry_price: float, stop_loss_price: float, direction: TradeDirection, tp_config: Dict[str, Any]) -> Optional[float]:
        """
        Calculate take profit price based on configuration.
        
        Args:
            entry_price: Entry price of the trade
            stop_loss_price: Calculated stop loss price
            direction: Direction of the trade (LONG or SHORT)
            tp_config: Configuration for take profit calculation
            
        Returns:
            Calculated take profit price or None if calculation fails
        """
        primary_type = tp_config.get("primary_type", "rr_ratio")
        logger.debug(f"Calculating take profit with primary_type={primary_type}, direction={direction.value}")
        
        take_profit_price = None
        
        if primary_type == "rr_ratio":
            # Calculate TP based on risk-reward ratio
            rr_ratio = tp_config.get("rr_ratio_settings", {}).get("value", 3.0)
            logger.debug(f"Using risk-reward ratio method with value={rr_ratio}")
            
            # Calculate risk (distance from entry to stop loss)
            if direction == TradeDirection.LONG:
                risk = entry_price - stop_loss_price
                if risk <= 0:
                    logger.warning(f"Invalid risk for LONG: entry={entry_price}, SL={stop_loss_price}")
                    return None
                take_profit_price = entry_price + (risk * rr_ratio)
            else:  # TradeDirection.SHORT
                risk = stop_loss_price - entry_price
                if risk <= 0:
                    logger.warning(f"Invalid risk for SHORT: entry={entry_price}, SL={stop_loss_price}")
                    return None
                take_profit_price = entry_price - (risk * rr_ratio)
        
        elif primary_type == "fixed_percentage":
            # Calculate TP based on fixed percentage from entry
            percentage_value = tp_config.get("fixed_percentage_settings", {}).get("value", 0.06)
            logger.debug(f"Using fixed percentage method with value={percentage_value}")
            
            if direction == TradeDirection.LONG:
                take_profit_price = entry_price * (1 + percentage_value)
            else:  # TradeDirection.SHORT
                take_profit_price = entry_price * (1 - percentage_value)
        
        else:
            logger.warning(f"Unknown TP primary_type: {primary_type}")
        
        # Final validation
        if take_profit_price is not None:
            if (direction == TradeDirection.LONG and take_profit_price <= entry_price) or \
               (direction == TradeDirection.SHORT and take_profit_price >= entry_price):
                logger.warning(f"Invalid TP: direction={direction.value}, entry={entry_price}, TP={take_profit_price}")
                return None
        
        logger.debug(f"Final calculated TP: {take_profit_price}")
        return take_profit_price

    def _passes_volume_filter(self, latest_kline_data: pd.Series, volume_filter_config: Dict[str, Any]) -> bool:
        """
        Checks if the current volume meets the required threshold compared to average volume.
        
        Args:
            latest_kline_data: Series containing the latest candle data including volume and avg_volume
            volume_filter_config: Configuration for volume filter
            
        Returns:
            True if filter passes, False if it fails
        """
        if not volume_filter_config.get("enabled", False):
            logger.debug("Volume filter not enabled, passing")
            return True
            
        current_volume = latest_kline_data.get("volume")
        avg_volume = latest_kline_data.get("avg_volume")
        
        # Check if we have valid volume data
        if pd.isna(avg_volume) or avg_volume == 0:
            logger.warning(f"Average volume is not available or zero. avg_volume: {avg_volume}")
            # Return True to not block signals when data isn't ready yet
            return True
            
        threshold_ratio = volume_filter_config.get("min_threshold", 1.5)
        volume_requirement = avg_volume * threshold_ratio
        
        # Check if current volume meets the minimum threshold
        passes = current_volume >= volume_requirement
        
        logger.debug(f"Volume filter: current={current_volume}, average={avg_volume}, " 
                    f"required={volume_requirement} (avg * {threshold_ratio}), passes={passes}")
        
        return passes

    def _passes_volatility_filter(self, latest_kline_data: pd.Series, volatility_filter_config: Dict[str, Any]) -> bool:
        """
        Checks if the current volatility (ATR) meets the specified conditions.
        
        Args:
            latest_kline_data: Series containing the latest candle data including close price and atr
            volatility_filter_config: Configuration for volatility filter
            
        Returns:
            True if filter passes, False if it fails
        """
        if not volatility_filter_config.get("enabled", False):
            logger.debug("Volatility filter not enabled, passing")
            return True
            
        atr = latest_kline_data.get("atr")
        close_price = latest_kline_data.get("close")
        
        # Check if we have valid ATR and price data
        if pd.isna(atr) or pd.isna(close_price) or close_price == 0:
            logger.warning(f"ATR or close price is not available. ATR: {atr}, Close: {close_price}")
            # Return True to not block signals when data isn't ready yet
            return True
            
        # Calculate ATR as percentage of price
        atr_percentage = (atr / close_price) * 100
        
        # Get filter configuration
        condition_type = volatility_filter_config.get("indicator", "atr")
        min_threshold = volatility_filter_config.get("min_threshold", 0.002)
        max_threshold = volatility_filter_config.get("max_threshold", 0.05)
        
        # Convert thresholds to percentage if they're not already (for readability in logs)
        min_threshold_pct = min_threshold * 100 if min_threshold < 1 else min_threshold
        max_threshold_pct = max_threshold * 100 if max_threshold < 1 else max_threshold
        
        # Implement different condition types
        passes = False
        
        if condition_type == "atr":
            # Check if ATR percentage is within desired range
            min_condition = atr_percentage >= min_threshold_pct
            max_condition = atr_percentage <= max_threshold_pct
            passes = min_condition and max_condition
            
            logger.debug(f"Volatility filter: ATR={atr}, Close={close_price}, ATR%={atr_percentage:.2f}%, " 
                         f"min={min_threshold_pct:.2f}%, max={max_threshold_pct:.2f}%, "
                         f"min_passed={min_condition}, max_passed={max_condition}, overall={passes}")
        else:
            logger.warning(f"Unknown volatility condition type: {condition_type}")
            passes = True  # Default to pass for unknown types
            
        return passes

    async def check_signal(self, api_symbol: str, config_symbol: str) -> Optional[TradeSignal]:
        """Checks for a V1 SMA crossover signal for the given API symbol (e.g., BTCUSDT)."""
        # 1. Get pair config with the new nested configurations
        pair_config = self._get_pair_specific_config(config_symbol)
        logger.debug(f"Checking signal for {api_symbol} with config: {pair_config}")

        # 2. Fetch DataFrame from data_processor
        signal_interval = "1m"  # V1 strategy uses 1-minute timeframe for signals
        df = self.data_processor.get_indicator_dataframe(api_symbol, signal_interval)

        # Perform initial data checks
        if df is None or df.empty or len(df) < 2:
            logger.debug(f"Not enough data for {api_symbol} {signal_interval} to generate signal. df exists: {df is not None}, empty: {df.empty if df is not None else True}, len: {len(df) if df is not None else 0}")
            return None

        # Get the latest two candles for crossover detection
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        sma_short_col = "sma_short"
        sma_long_col = "sma_long"

        # Check if required columns exist and are not NaN
        if not all(col in latest.index and col in previous.index for col in [sma_short_col, sma_long_col, "close", "is_closed"]):
            logger.debug(f"Required columns not available for {api_symbol} {signal_interval}.")
            return None
        
        # Ensure latest candle data is present and SMAs are calculated
        if pd.isna(latest[sma_short_col]) or pd.isna(latest[sma_long_col]) or \
           pd.isna(previous[sma_short_col]) or pd.isna(previous[sma_long_col]):
            logger.debug(f"SMA values are NA for {api_symbol} {signal_interval}. Latest: {latest[sma_short_col]}, {latest[sma_long_col]}. Previous: {previous[sma_short_col]}, {previous[sma_long_col]}")
            return None

        # 3. Core SMA Crossover detection
        logger.debug(f"SMA values for {api_symbol}: Latest short={latest[sma_short_col]}, long={latest[sma_long_col]}. Previous short={previous[sma_short_col]}, long={previous[sma_long_col]}")
        
        crossed_up = (previous[sma_short_col] <= previous[sma_long_col] and
                      latest[sma_short_col] > latest[sma_long_col])
        crossed_down = (previous[sma_short_col] >= previous[sma_long_col] and
                        latest[sma_short_col] < latest[sma_long_col])

        logger.debug(f"Crossover detection for {api_symbol}: crossed_up={crossed_up}, crossed_down={crossed_down}")

        signal_direction: Optional[TradeDirection] = None
        if crossed_up:
            signal_direction = TradeDirection.LONG
            logger.debug(f"Detected LONG signal for {api_symbol}: Short SMA crossed above Long SMA")
        elif crossed_down:
            signal_direction = TradeDirection.SHORT
            logger.debug(f"Detected SHORT signal for {api_symbol}: Short SMA crossed below Long SMA")
        else:
            logger.debug(f"No crossover detected for {api_symbol}")
            return None  # No crossover

        # 4. Min signal interval check
        min_interval_seconds = pair_config["min_signal_interval_minutes"] * 60
        current_time = time.time()
        if api_symbol in self.last_signal_time and (current_time - self.last_signal_time[api_symbol]) < min_interval_seconds:
            logger.debug(f"Signal for {api_symbol} too soon. Last signal at {self.last_signal_time[api_symbol]}, current: {current_time}, min_interval: {min_interval_seconds}")
            return None

        # 5. Buffer Time Filter
        buffer_time_config = pair_config["filters"]["buffer_time"]
        if buffer_time_config["enabled"]:
            if not self._passes_buffer_time_filter(df, buffer_time_config, signal_direction):
                logger.info(f"Buffer time filter failed for {api_symbol} {signal_direction.value}")
                return None
            logger.debug(f"Buffer time filter passed for {api_symbol} {signal_direction.value}")

        # 6. Volume Filter
        volume_config = pair_config["filters"]["volume"]
        if volume_config["enabled"]:
            if not self._passes_volume_filter(latest, volume_config):
                logger.info(f"Volume filter failed for {api_symbol}")
                return None
            logger.debug(f"Volume filter passed for {api_symbol} {signal_direction.value}")

        # 7. Volatility Filter
        volatility_config = pair_config["filters"]["volatility"]
        if volatility_config["enabled"]:
            if not self._passes_volatility_filter(latest, volatility_config):
                logger.info(f"Volatility filter failed for {api_symbol}")
                return None
            logger.debug(f"Volatility filter passed for {api_symbol} {signal_direction.value}")

        # 8. Entry price
        entry_price = latest["close"]
        logger.debug(f"Entry price for {api_symbol}: {entry_price}")

        # 9. SL Calculation
        stop_loss_price = self._calculate_stop_loss(
            entry_price=entry_price,
            direction=signal_direction,
            df=df,
            sl_config=pair_config["sl_options"]
        )
        
        if stop_loss_price is None:
            logger.warning(f"Failed to calculate valid stop loss for {api_symbol}. Skipping signal.")
            return None

        # 10. TP Calculation
        take_profit_price = self._calculate_take_profit(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            direction=signal_direction,
            tp_config=pair_config["tp_options"]
        )
        
        if take_profit_price is None:
            logger.warning(f"Failed to calculate valid take profit for {api_symbol}. Skipping signal.")
            return None

        # 11. Final SL/TP Validation
        if (signal_direction == TradeDirection.LONG and (stop_loss_price >= entry_price or take_profit_price <= entry_price)) or \
           (signal_direction == TradeDirection.SHORT and (stop_loss_price <= entry_price or take_profit_price >= entry_price)):
            logger.warning(f"Calculated SL/TP invalid for {api_symbol}: Direction={signal_direction.value}, Entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}. Skipping signal.")
            return None

        # Final validation passed, update last signal time
        self.last_signal_time[api_symbol] = current_time
        logger.info(f"Generated signal for {api_symbol}: {signal_direction.value} at {entry_price}, SL={stop_loss_price}, TP={take_profit_price}")

        # Construct signal kline from the latest data point in the DataFrame
        signal_kline_data = latest.to_dict()
        signal_kline_obj = Kline(
            timestamp=int(latest.name),  # timestamp is the index
            open=signal_kline_data.get("open"),
            high=signal_kline_data.get("high"),
            low=signal_kline_data.get("low"),
            close=signal_kline_data.get("close"),
            volume=signal_kline_data.get("volume"),
            is_closed=signal_kline_data.get("is_closed", True),
            symbol=api_symbol,
            interval=signal_interval
        )

        # Include filter and SL/TP details in signal
        sl_method = pair_config["sl_options"]["primary_type"]
        tp_method = pair_config["tp_options"]["primary_type"]
        
        # Determine which pivot was used for SL if applicable
        pivot_used = None
        if sl_method == "pivot":
            pivot_used = self._find_recent_pivot(df, 
                                               lookback=pair_config["sl_options"]["pivot_settings"]["lookback_candles"], 
                                               direction=signal_direction)

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
                "pivot_used_for_sl": pivot_used,
                "sl_method": sl_method,
                "tp_method": tp_method,
                "buffer_time_filter_enabled": buffer_time_config["enabled"],
                "volume_filter_enabled": volume_config["enabled"],
                "volatility_filter_enabled": volatility_config["enabled"],
                "volume_at_signal": latest.get("volume"),
                "avg_volume_at_signal": latest.get("avg_volume"),
                "atr_at_signal": latest.get("atr"),
                "atr_percentage_of_price": None if pd.isna(latest.get("atr")) or latest.get("close") == 0 
                                           else (latest.get("atr") / latest.get("close")) * 100
            }
        )

