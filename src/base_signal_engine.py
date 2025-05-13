import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import pandas as pd

from src.config_loader import ConfigManager
from src.data_processor import DataProcessor
from src.models import TradeSignal

logger = logging.getLogger(__name__)

class BaseSignalEngine(ABC):
    """
    Abstract Base Class for signal engines.
    All signal engines should inherit from this class and implement the required methods.
    """
    
    def __init__(self, config_manager: ConfigManager, data_processor: DataProcessor):
        """
        Initialize the base signal engine.
        
        Args:
            config_manager: Configuration manager instance
            data_processor: Data processor instance
        """
        self.config_manager = config_manager
        self.data_processor = data_processor
        
    @abstractmethod
    async def check_signal(self, api_symbol: str, config_symbol: str) -> Optional[TradeSignal]:
        """
        Check for a trading signal for the given symbol.
        
        Args:
            api_symbol: Symbol in API format (e.g., "BTCUSDT")
            config_symbol: Symbol in config format (e.g., "BTC_USDT")
            
        Returns:
            TradeSignal if a valid signal is found, None otherwise
        """
        pass
        
    @abstractmethod
    async def check_signal_with_df(self, api_symbol: str, config_symbol: str, point_in_time_df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Check for a trading signal using the provided DataFrame directly.
        This allows passing a specific historical context for backtesting.
        
        Args:
            api_symbol: Symbol in API format (e.g., "BTCUSDT")
            config_symbol: Symbol in config format (e.g., "BTC_USDT")
            point_in_time_df: DataFrame containing price and indicator data up to the point in time being evaluated
            
        Returns:
            TradeSignal if a valid signal is found, None otherwise
        """
        pass
    
    @abstractmethod
    def _get_pair_specific_config(self, config_symbol: str) -> Dict[str, Any]:
        """
        Get pair-specific configuration, merging with global defaults.
        
        Args:
            config_symbol: Symbol in config format (e.g., "BTC_USDT")
            
        Returns:
            Dictionary containing the merged configuration
        """
        pass 