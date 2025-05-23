import pandas as pd
from collections import deque, defaultdict
import logging
from typing import Dict, Deque, Optional, List, Tuple, Callable

from src.models import Kline
from src.config_loader import ConfigManager
from src.connectors import convert_symbol_to_api_format # For consistent symbol usage

logger = logging.getLogger(__name__)

# Define a maximum length for k-line deques to prevent memory issues
# Needs to be at least max(sma_long_period) + some buffer for calculations
# For SMA 200, we need at least 200 candles. Let's use a bit more.
MAX_KLINE_BUFFER_LENGTH = 300 

class DataProcessor:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        # Structure: self.kline_buffers[api_symbol][interval] = deque([Kline, ...])
        self.kline_buffers: Dict[str, Dict[str, Deque[Kline]]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=MAX_KLINE_BUFFER_LENGTH)))
        # Structure: self.indicator_data[api_symbol][interval] = pd.DataFrame with columns [timestamp, open, high, low, close, volume, sma_short, sma_long]
        self.indicator_data: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(lambda: defaultdict(pd.DataFrame))
        
        self.config_manager.register_callback(self._handle_config_update) # Listen for config changes
        self._active_pairs_timeframes: Dict[str, List[str]] = {}
        self._initialize_buffers_from_config()

    def _handle_config_update(self, new_config: Dict):
        logger.info("DataProcessor: Configuration updated. Re-initializing buffers and settings.")
        self._initialize_buffers_from_config(new_config)

    def _initialize_buffers_from_config(self, config: Optional[Dict] = None):
        if config is None:
            config = self.config_manager.get_config()
        
        new_active_pairs_timeframes: Dict[str, List[str]] = {}
        pairs_config = config.get("pairs", {})
        global_settings = config.get("global_settings", {})
        default_timeframes = global_settings.get("v1_strategy", {}).get("indicator_timeframes", ["1m"])

        for pair_symbol_config, pair_details in pairs_config.items():
            if pair_details.get("enabled", False):
                api_symbol = convert_symbol_to_api_format(pair_symbol_config)
                timeframes_for_pair = pair_details.get("indicator_timeframes", default_timeframes)
                new_active_pairs_timeframes[api_symbol] = timeframes_for_pair
                # Ensure buffers exist for newly active pairs/timeframes
                for tf in timeframes_for_pair:
                    if tf not in self.kline_buffers[api_symbol]:
                        self.kline_buffers[api_symbol][tf] = deque(maxlen=MAX_KLINE_BUFFER_LENGTH)
                    if tf not in self.indicator_data[api_symbol]:
                        self.indicator_data[api_symbol][tf] = pd.DataFrame()
        
        # Cleanup buffers for pairs/timeframes that are no longer active (optional, to save memory)
        # For simplicity, we are not removing old buffers now, but this could be added.
        self._active_pairs_timeframes = new_active_pairs_timeframes
        logger.debug(f"DataProcessor initialized/updated for pairs and timeframes: {self._active_pairs_timeframes}")

    async def process_kline(self, kline: Kline, market_type: Optional[str] = None):
        """ Process an incoming Kline object. market_type is for context if needed. """
        api_symbol = kline.symbol # Assuming kline.symbol is already in API format (e.g., BTCUSDT)
        interval = kline.interval

        if api_symbol not in self._active_pairs_timeframes or interval not in self._active_pairs_timeframes.get(api_symbol, []):
            # logger.debug(f"Skipping kline for inactive pair/timeframe: {api_symbol} {interval}")
            return

        # Add to deque. If it's an update to the last candle, replace it.
        # Kline data from Binance WS usually gives the current (non-closed) candle and then the closed one.
        # We are interested in closed candles for SMA calculation primarily.
        
        buffer = self.kline_buffers[api_symbol][interval]
        if kline.is_closed:
            if buffer and buffer[-1].timestamp == kline.timestamp:
                if not buffer[-1].is_closed:
                    buffer[-1] = kline # Replace the non-closed one with the closed one
            else:
                buffer.append(kline)
            self._update_indicators(api_symbol, interval)
        else: # Kline is not closed (current, ongoing candle)
            if buffer and buffer[-1].timestamp == kline.timestamp:
                buffer[-1] = kline # Update the last (current) candle
            else:
                buffer.append(kline) # Add as new current candle

    def _update_indicators(self, api_symbol: str, interval: str):
        kline_deque = self.kline_buffers[api_symbol][interval]
        if not kline_deque:
            return
        
        df_data = {
            "timestamp": [k.timestamp for k in kline_deque],
            "open": [k.open for k in kline_deque],
            "high": [k.high for k in kline_deque],
            "low": [k.low for k in kline_deque],
            "close": [k.close for k in kline_deque],
            "volume": [k.volume for k in kline_deque],
            "is_closed": [k.is_closed for k in kline_deque]
        }
        df = pd.DataFrame(df_data)
        df.set_index("timestamp", inplace=True, drop=False) # Keep timestamp column too
        df.sort_index(inplace=True) # Ensure time order
        df = df[~df.index.duplicated(keep="last")] # Remove duplicate timestamps, keep last update

        if df.empty:
            return

        config = self.config_manager.get_config()
        global_settings = config.get("global_settings", {})
        v1_strategy_config = global_settings.get("v1_strategy", {})
        sma_short_period = v1_strategy_config.get("sma_short_period", 21)
        sma_long_period = v1_strategy_config.get("sma_long_period", 200)

        # Get Volume & Volatility settings from global config
        volume_config = v1_strategy_config.get("filters", {}).get("volume", {})
        volatility_config = v1_strategy_config.get("filters", {}).get("volatility", {})
        
        # Retrieve periods for Average Volume and ATR from config
        avg_vol_period = volume_config.get("lookback_periods", 20)
        atr_period = volatility_config.get("lookback_periods", 14)
        
        logger.debug(f"Indicator periods from config - SMA short: {sma_short_period}, SMA long: {sma_long_period}, "
                   f"Avg Volume: {avg_vol_period}, ATR: {atr_period}")
        
        # Calculate SMAs
        # Special handling for SMA period 1
        if sma_short_period == 1:
            logger.debug("Using direct close prices for SMA_short period 1")
            df["sma_short"] = df["close"]
        elif len(df) >= sma_short_period:
            df["sma_short"] = df["close"].rolling(window=sma_short_period).mean()
        else:
            df["sma_short"] = pd.NA

        # Special handling for SMA period 2
        if sma_long_period == 2 and len(df) >= 2:
            logger.debug("Manual calculation for SMA_long period 2")
            # Manual calculation for SMA(2) to ensure it's correct
            # For the last row, it's (current close + previous close) / 2
            last_two_closes = df["close"].tail(2).values
            if len(last_two_closes) == 2:
                last_sma2 = (last_two_closes[0] + last_two_closes[1]) / 2
                logger.debug(f"Last two closes: {last_two_closes[0]}, {last_two_closes[1]}, SMA(2): {last_sma2}")
            df["sma_long"] = df["close"].rolling(window=2).mean()
        elif len(df) >= sma_long_period:
            df["sma_long"] = df["close"].rolling(window=sma_long_period).mean()
        else:
            df["sma_long"] = pd.NA
        
        # Calculate Average Volume
        if len(df) >= avg_vol_period:
            df["avg_volume"] = df["volume"].rolling(window=avg_vol_period).mean()
            logger.debug(f"Calculated Average Volume for {api_symbol}/{interval} over {avg_vol_period} periods")
        else:
            df["avg_volume"] = pd.NA
            logger.debug(f"Not enough data for Average Volume calculation, need {avg_vol_period} candles, got {len(df)}")
        
        # Calculate ATR (Average True Range)
        # TR = max(High - Low, abs(High - Previous_Close), abs(Low - Previous_Close))
        if len(df) >= 2:  # Need at least 2 candles for TR calculation due to Previous_Close
            # Calculate components of True Range
            high_low = df["high"] - df["low"]
            high_close_prev = abs(df["high"] - df["close"].shift(1))
            low_close_prev = abs(df["low"] - df["close"].shift(1))
            
            # True Range is the maximum of the three components
            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            
            # Calculate ATR as SMA of True Range
            if len(df) >= atr_period + 1:  # +1 because first TR value will be NaN due to shift
                df["atr"] = tr.rolling(window=atr_period).mean()
                logger.debug(f"Calculated ATR for {api_symbol}/{interval} over {atr_period} periods")
            else:
                df["atr"] = pd.NA
                logger.debug(f"Not enough data for ATR calculation, need {atr_period+1} candles, got {len(df)}")
        else:
            df["atr"] = pd.NA
            logger.debug(f"Not enough data for TR calculation, need at least 2 candles, got {len(df)}")
        
        self.indicator_data[api_symbol][interval] = df

    def get_latest_kline(self, api_symbol: str, interval: str) -> Optional[Kline]:
        if api_symbol in self.kline_buffers and interval in self.kline_buffers[api_symbol]:
            if self.kline_buffers[api_symbol][interval]:
                return self.kline_buffers[api_symbol][interval][-1]
        return None

    def get_indicator_dataframe(self, api_symbol: str, interval: str) -> Optional[pd.DataFrame]:
        if api_symbol in self.indicator_data and interval in self.indicator_data[api_symbol]:
            return self.indicator_data[api_symbol][interval].copy() # Return a copy
        return None

    def get_latest_indicators(self, api_symbol: str, interval: str) -> Optional[pd.Series]:
        df = self.get_indicator_dataframe(api_symbol, interval)
        if df is not None and not df.empty:
            return df.iloc[-1]
        return None

