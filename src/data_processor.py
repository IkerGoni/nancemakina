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
        sma_short_period = config.get("global_settings", {}).get("v1_strategy", {}).get("sma_short_period", 21)
        sma_long_period = config.get("global_settings", {}).get("v1_strategy", {}).get("sma_long_period", 200)

        if len(df) >= sma_short_period:
            df["sma_short"] = df["close"].rolling(window=sma_short_period).mean()
        else:
            df["sma_short"] = pd.NA

        if len(df) >= sma_long_period:
            df["sma_long"] = df["close"].rolling(window=sma_long_period).mean()
        else:
            df["sma_long"] = pd.NA
        
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

