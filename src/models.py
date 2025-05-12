from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
import time
from enum import Enum

# Enum for trade direction
class TradeDirection(str, Enum):
    LONG = "BUY"
    SHORT = "SELL"

class Kline(BaseModel):
    timestamp: int # Kline start time in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_asset_volume: Optional[float] = None # Volume in quote asset
    number_of_trades: Optional[int] = None
    is_closed: bool # True if this kline is closed
    symbol: Optional[str] = None # Trading pair symbol, e.g., BTCUSDT
    interval: Optional[str] = None # Kline interval, e.g., 1m, 5m, 1h

class PairConfig(BaseModel):
    enabled: bool = True
    symbol: str # e.g. BTC_USDT (config format), will be converted for API/WS
    contract_type: Literal["USDT_M", "COIN_M"] = "USDT_M"

    # V1 Strategy specific (can be overridden from global)
    sma_short_period: Optional[int] = None
    sma_long_period: Optional[int] = None
    min_signal_interval_minutes: Optional[int] = None
    tp_sl_ratio: Optional[float] = None
    margin_usdt: Optional[float] = None # For USDT-M
    margin_coin: Optional[float] = None # For COIN-M, in base currency units
    leverage: Optional[int] = None
    margin_mode: Optional[Literal["ISOLATED", "CROSSED"]] = None

    # V2 Strategy specific (can be overridden from global)
    strategy_v2_enabled: Optional[bool] = None
    # ... other V2 params like ema_short_period_1m, stochrsi_k_1m etc.

    # Filter overrides
    volume_filter_enabled: Optional[bool] = None
    # ... other filter params ...

    # Risk management overrides
    dynamic_sizing_enabled: Optional[bool] = None
    risk_percent_per_trade: Optional[float] = None

    @validator("symbol")
    def symbol_format(cls, value):
        # Basic validation, can be expanded
        if "_" not in value and not value.endswith("PERP") and value.isupper():
            pass 
        elif "_" not in value:
            raise ValueError("Symbol in config should generally be in format BASE_QUOTE, e.g., BTC_USDT or end with PERP for COIN-M e.g. BTCUSD_PERP")
        return value

class TradeSignal(BaseModel):
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    symbol: str # e.g., BTCUSDT (API format)
    config_symbol: str # e.g., BTC_USDT (Config format, for logging/reference)
    contract_type: Literal["USDT_M", "COIN_M"]
    direction: TradeDirection
    entry_price: float # Estimated entry price at signal time (e.g., current close)
    stop_loss_price: float
    take_profit_price: float
    strategy_name: str = "V1_SMA_Crossover" # Or V2_MultiTF_EMA, etc.
    signal_kline: Optional[Kline] = None # The kline that triggered the signal
    details: Optional[Dict[str, Any]] = None # e.g., SMA values, pivot points used for SL

class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL" 
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS" 
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT" 
    TAKE_PROFIT = "TAKE_PROFIT" 
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT" 
    LIMIT_MAKER = "LIMIT_MAKER"
    STOP = "STOP" 
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class Order(BaseModel):
    order_id: str # Exchange order ID
    client_order_id: Optional[str] = None # Custom order ID
    symbol: str # API format, e.g., BTCUSDT
    type: OrderType
    side: OrderSide
    quantity: float # Order quantity
    price: Optional[float] = None # For LIMIT orders
    stop_price: Optional[float] = None # For STOP_MARKET, TAKE_PROFIT_MARKET etc.
    status: OrderStatus
    timestamp: int # Order creation/update timestamp (milliseconds)
    avg_fill_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    reduce_only: Optional[bool] = False
    commission: Optional[float] = None
    commission_asset: Optional[str] = None

class Position(BaseModel):
    symbol: str # API format, e.g., BTCUSDT
    contract_type: Literal["USDT_M", "COIN_M"]
    side: TradeDirection # LONG or SHORT
    entry_price: float
    quantity: float # Size of the position
    margin_type: Optional[Literal["ISOLATED", "CROSSED"]] = None
    leverage: Optional[int] = None
    unrealized_pnl: Optional[float] = None
    liquidation_price: Optional[float] = None
    mark_price: Optional[float] = None
    entry_timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    current_sl_price: Optional[float] = None
    current_tp_price: Optional[float] = None
    trailing_sl_active: Optional[bool] = False

# Example usage (not part of the module, just for illustration)
if __name__ == "__main__":
    kline_data = {
        "timestamp": int(time.time() * 1000),
        "open": 40000.0, "high": 40100.0, "low": 39900.0, "close": 40050.0,
        "volume": 100.0, "is_closed": True, "symbol": "BTCUSDT", "interval": "1m"
    }
    kline_obj = Kline(**kline_data)
    print(f"Kline: {kline_obj.json(indent=2)}")

    pair_cfg_data = {"symbol": "BTC_USDT", "leverage": 20, "margin_usdt": 100}
    pair_cfg_obj = PairConfig(**pair_cfg_data)
    print(f"PairConfig: {pair_cfg_obj.json(indent=2)}")

    signal_data = {
        "symbol": "BTCUSDT", "config_symbol": "BTC_USDT", "contract_type": "USDT_M",
        "direction": TradeDirection.LONG, "entry_price": 40050.0,
        "stop_loss_price": 39800.0, "take_profit_price": 40800.0,
        "details": {"sma_short": 40020, "sma_long": 40000}
    }
    signal_obj = TradeSignal(**signal_data)
    print(f"TradeSignal: {signal_obj.json(indent=2)}")

    order_data = {
        "order_id": "12345", "symbol": "BTCUSDT", "type": OrderType.MARKET, "side": OrderSide.BUY,
        "quantity": 0.001, "status": OrderStatus.FILLED, "timestamp": int(time.time() * 1000),
        "avg_fill_price": 40055.0, "filled_quantity": 0.001
    }
    order_obj = Order(**order_data)
    print(f"Order: {order_obj.json(indent=2)}")

    position_data = {
        "symbol": "BTCUSDT", "contract_type": "USDT_M", "side": TradeDirection.LONG, "entry_price": 40055.0,
        "quantity": 0.001, "leverage": 20, "margin_type": "ISOLATED"
    }
    position_obj = Position(**position_data)
    print(f"Position: {position_obj.json(indent=2)}")

