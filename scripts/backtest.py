#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import yaml
import matplotlib.pyplot as plt
from pydantic import BaseModel
import asyncio
from decimal import Decimal, getcontext, ROUND_DOWN

# Set decimal precision for financial calculations
getcontext().prec = 28

# Import necessary models and utilities
from src.models import Kline, TradeSignal, TradeDirection
from src.config_loader import ConfigManager
from src.data_processor import DataProcessor
from src.signal_engine import SignalEngineV1

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('backtest')

class BacktestPosition(BaseModel):
    symbol: str
    entry_price: float
    entry_time: int
    quantity: float
    direction: TradeDirection
    stop_loss: float
    take_profit: float
    status: str = "OPEN"  # OPEN, CLOSED_TP, CLOSED_SL, CLOSED_EXIT
    exit_price: Optional[float] = None
    exit_time: Optional[int] = None
    profit_loss: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    
class BacktestResults(BaseModel):
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit_percent: float
    avg_loss_percent: float
    max_drawdown: float
    profit_factor: float
    total_fees: float
    
class Backtester:
    def __init__(self, config_path: str, data_file: str):
        self.config_path = config_path
        self.data_file = data_file
        self.config_manager = ConfigManager(config_path)
        self.data_processor = DataProcessor(self.config_manager)
        self.signal_engine = SignalEngineV1(self.config_manager, self.data_processor)
        
        # Backtest specific properties
        self.initial_balance = Decimal('10000.0')  # Default USDT
        self.current_balance = self.initial_balance
        self.positions: List[BacktestPosition] = []
        self.trade_history: List[BacktestPosition] = []
        self.equity_curve: List[Tuple[int, float]] = []  # (timestamp, balance)
        
        # Load backtest settings from config
        self._load_backtest_config()
        
    def _load_backtest_config(self):
        """Load backtesting specific configuration"""
        config = self.config_manager.get_config()
        backtest_settings_from_config = config.get("backtesting")  # Get potential None or dict

        # Ensure backtest_config is a dictionary to safely call .get() on it
        backtest_config = backtest_settings_from_config if isinstance(backtest_settings_from_config, dict) else {}
        
        self.initial_balance = Decimal(str(backtest_config.get("initial_balance_usdt", 10000.0)))
        self.current_balance = self.initial_balance
        self.fee_percent = Decimal(str(backtest_config.get("fee_percent", 0.04))) / Decimal('100')  # Convert to decimal
        self.slippage_percent = Decimal(str(backtest_config.get("slippage_percent", 0.01))) / Decimal('100')  # Convert to decimal
        
    def load_historical_data(self) -> List[Kline]:
        """Load historical data from CSV file and convert to Kline objects"""
        logger.info(f"Loading historical data from {self.data_file}")
        
        try:
            # Load data from CSV
            df = pd.read_csv(self.data_file)
            
            # Convert 'Open time' from string to timestamp
            df['timestamp'] = pd.to_datetime(df['Open time']).astype(np.int64) // 10**6  # Convert to milliseconds
            
            # Create list of Kline objects
            klines = []
            for _, row in df.iterrows():
                kline = Kline(
                    timestamp=int(row['timestamp']),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    quote_asset_volume=float(row.get('Taker buy base asset volume', 0)),
                    number_of_trades=int(row.get('Number of trades', 0)),
                    is_closed=True,  # All historical data is considered closed
                    symbol="BTCUSDT",  # Default symbol, will be overridden in process_data_async
                    interval="1m"  # Default interval, will be overridden in process_data_async
                )
                klines.append(kline)
            
            logger.info(f"Loaded {len(klines)} data points")
            return klines
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    async def process_data_async(self, klines: List[Kline], symbol: str = "BTCUSDT", interval: str = "1m") -> pd.DataFrame:
        """
        Process historical Kline data through the data processor
        """
        logger.info(f"Processing {len(klines)} klines for {symbol} {interval}")
        
        # Update klines with symbol and interval
        for kline in klines:
            kline.symbol = symbol
            kline.interval = interval
            
            # Process each kline through the data processor
            await self.data_processor.process_kline(kline)
        
        logger.info(f"Finished processing klines and calculating indicators")
        
        # Return the dataframe with indicators
        return self.data_processor.indicator_data[symbol][interval]
        
    async def generate_signals(self, df: pd.DataFrame, symbol: str = "BTCUSDT", config_symbol: str = "BTC_USDT") -> List[TradeSignal]:
        """
        Generate trading signals for the historical data
        """
        logger.info(f"Generating signals for {symbol}")
        
        signals = []
        
        # Disable the time-based filtering temporarily for backtest
        # Store the original last_signal_time
        original_last_signal_time = self.signal_engine.last_signal_time.copy()
        self.signal_engine.last_signal_time = {}
        
        # Iterate through each row (simulating the passage of time)
        for i in range(2, len(df)):  # Start from 2 to have previous candle data
            # Get current and previous rows
            current_row = df.iloc[i]
            
            # Check if we have valid SMA calculations
            if pd.notna(current_row['sma_short']) and pd.notna(current_row['sma_long']):
                # Generate signal for current state
                signal = await self.signal_engine.check_signal(symbol, config_symbol)
                
                if signal:
                    signals.append(signal)
                    logger.info(f"Signal generated at {datetime.fromtimestamp(signal.timestamp/1000)}: "
                               f"{signal.direction.value} at {signal.entry_price}, "
                               f"SL={signal.stop_loss_price}, TP={signal.take_profit_price}")
        
        # Restore the original last_signal_time
        self.signal_engine.last_signal_time = original_last_signal_time
        
        logger.info(f"Generated {len(signals)} signals")
        return signals

    def simulate_trading(self, signals: List[TradeSignal], df: pd.DataFrame):
        """
        Simulate trading based on generated signals
        """
        logger.info(f"Starting trading simulation with {len(signals)} signals")
        
        # Handle the case where no signals were generated
        if not signals:
            logger.warning("No signals provided to simulate_trading. Exiting simulation.")
            # Initialize the equity curve with initial balance
            if df is not None and not df.empty:
                self.equity_curve = [(df.iloc[0]['timestamp'], float(self.initial_balance))]
            else:
                self.equity_curve = [(int(datetime.now().timestamp() * 1000), float(self.initial_balance))]
            self.final_balance = float(self.current_balance)
            self.total_fees = Decimal('0.0')
            return [], self.equity_curve
        
        # Get configuration
        config = self.config_manager.get_config()
        pair_configs = config.get("pairs", {})
        global_v1_config = config.get("global_settings", {}).get("v1_strategy", {})
        
        # Get symbol specific config
        symbol_config = None
        for cfg_sym, details in pair_configs.items():
            if cfg_sym.upper() == signals[0].config_symbol.upper():
                symbol_config = details
                break
        
        # Check if symbol_config was found
        if not symbol_config:
            logger.error(f"Configuration for {signals[0].config_symbol} not found. Cannot simulate trades.")
            # Initialize the equity curve with initial balance
            if df is not None and not df.empty:
                self.equity_curve = [(df.iloc[0]['timestamp'], float(self.initial_balance))]
            else:
                self.equity_curve = [(int(datetime.now().timestamp() * 1000), float(self.initial_balance))]
            self.final_balance = float(self.current_balance)
            self.total_fees = Decimal('0.0')
            return [], self.equity_curve
        
        # Set leverage and margin
        leverage = Decimal(str(symbol_config.get("leverage", global_v1_config.get("default_leverage", 10))))
        fixed_margin_usdt = Decimal(str(symbol_config.get("margin_usdt", global_v1_config.get("default_margin_usdt", 50.0))))
        
        # Track current positions
        open_positions = []
        closed_positions = []
        
        # Track equity curve
        equity_curve = [(df.iloc[0]['timestamp'], float(self.initial_balance))]
        
        # Iterate through each candle
        current_balance = self.initial_balance
        total_fees = Decimal('0.0')
        
        for i in range(len(df)):
            current_candle = df.iloc[i]
            current_time = current_candle['timestamp']
            
            # Check for signals at this timestamp
            new_signals = [s for s in signals if s.timestamp == current_time]
            
            # Process open positions first (check for TP/SL)
            for pos in list(open_positions):
                # Check if TP hit
                if (pos.direction == TradeDirection.LONG and current_candle['high'] >= pos.take_profit) or \
                   (pos.direction == TradeDirection.SHORT and current_candle['low'] <= pos.take_profit):
                    # Close at take profit
                    pos.status = "CLOSED_TP"
                    pos.exit_price = pos.take_profit
                    pos.exit_time = current_time
                    
                    # Calculate P&L using Decimal for precision
                    if pos.direction == TradeDirection.LONG:
                        pos.profit_loss = float(
                            (Decimal(str(pos.exit_price)) - Decimal(str(pos.entry_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    else:  # SHORT
                        pos.profit_loss = float(
                            (Decimal(str(pos.entry_price)) - Decimal(str(pos.exit_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    
                    # Account for fees
                    entry_fee = Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    exit_fee = Decimal(str(pos.exit_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    fee = entry_fee + exit_fee
                    pos.profit_loss = float(Decimal(str(pos.profit_loss)) - fee)
                    total_fees += fee
                    
                    # Update balance
                    current_balance += Decimal(str(pos.profit_loss))
                    
                    # Calculate percent P&L
                    pos.profit_loss_percent = float(
                        (Decimal(str(pos.profit_loss)) / 
                        (Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)))) * Decimal('100')
                    )
                    
                    # Log
                    logger.info(f"TP HIT: {pos.direction.value} {pos.symbol} closed at {pos.exit_price}, "
                                f"P&L: ${pos.profit_loss:.2f} ({pos.profit_loss_percent:.2f}%)")
                    
                    # Move to closed positions
                    closed_positions.append(pos)
                    open_positions.remove(pos)
                
                # Check if SL hit
                elif (pos.direction == TradeDirection.LONG and current_candle['low'] <= pos.stop_loss) or \
                     (pos.direction == TradeDirection.SHORT and current_candle['high'] >= pos.stop_loss):
                    # Close at stop loss
                    pos.status = "CLOSED_SL"
                    pos.exit_price = pos.stop_loss
                    pos.exit_time = current_time
                    
                    # Calculate P&L using Decimal for precision
                    if pos.direction == TradeDirection.LONG:
                        pos.profit_loss = float(
                            (Decimal(str(pos.exit_price)) - Decimal(str(pos.entry_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    else:  # SHORT
                        pos.profit_loss = float(
                            (Decimal(str(pos.entry_price)) - Decimal(str(pos.exit_price))) * 
                            Decimal(str(pos.quantity)) * leverage
                        )
                    
                    # Account for fees
                    entry_fee = Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    exit_fee = Decimal(str(pos.exit_price)) * Decimal(str(pos.quantity)) * self.fee_percent
                    fee = entry_fee + exit_fee
                    pos.profit_loss = float(Decimal(str(pos.profit_loss)) - fee)
                    total_fees += fee
                    
                    # Update balance
                    current_balance += Decimal(str(pos.profit_loss))
                    
                    # Calculate percent P&L
                    pos.profit_loss_percent = float(
                        (Decimal(str(pos.profit_loss)) / 
                        (Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)))) * Decimal('100')
                    )
                    
                    # Log
                    logger.info(f"SL HIT: {pos.direction.value} {pos.symbol} closed at {pos.exit_price}, "
                                f"P&L: ${pos.profit_loss:.2f} ({pos.profit_loss_percent:.2f}%)")
                    
                    # Move to closed positions
                    closed_positions.append(pos)
                    open_positions.remove(pos)
            
            # Process new signals and open positions
            for signal in new_signals:
                # Calculate position size
                position_size = fixed_margin_usdt / Decimal(str(signal.entry_price))
                
                # Apply slippage to entry
                entry_price = Decimal(str(signal.entry_price))
                if signal.direction == TradeDirection.LONG:
                    entry_price *= (Decimal('1') + self.slippage_percent)  # Higher for buys
                else:
                    entry_price *= (Decimal('1') - self.slippage_percent)  # Lower for sells
                
                # Create new position
                new_position = BacktestPosition(
                    symbol=signal.symbol,
                    entry_price=float(entry_price),
                    entry_time=current_time,
                    quantity=float(position_size),
                    direction=signal.direction,
                    stop_loss=signal.stop_loss_price,
                    take_profit=signal.take_profit_price,
                    status="OPEN"
                )
                
                # Calculate entry fee
                entry_fee = entry_price * position_size * self.fee_percent
                total_fees += entry_fee
                
                # Log
                logger.info(f"OPEN: {signal.direction.value} {signal.symbol} at {float(entry_price)}, "
                            f"Size: {float(position_size)}, SL: {signal.stop_loss_price}, TP: {signal.take_profit_price}")
                
                # Add to open positions
                open_positions.append(new_position)
            
            # Track equity at each step
            equity_curve.append((current_time, float(current_balance)))
        
        # Close any remaining open positions at the last price
        last_candle = df.iloc[-1]
        for pos in list(open_positions):
            pos.status = "CLOSED_EXIT"
            pos.exit_price = float(last_candle['close'])
            pos.exit_time = last_candle['timestamp']
            
            # Calculate P&L using Decimal
            if pos.direction == TradeDirection.LONG:
                pos.profit_loss = float(
                    (Decimal(str(pos.exit_price)) - Decimal(str(pos.entry_price))) * 
                    Decimal(str(pos.quantity)) * leverage
                )
            else:  # SHORT
                pos.profit_loss = float(
                    (Decimal(str(pos.entry_price)) - Decimal(str(pos.exit_price))) * 
                    Decimal(str(pos.quantity)) * leverage
                )
            
            # Account for fees
            entry_fee = Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)) * self.fee_percent
            exit_fee = Decimal(str(pos.exit_price)) * Decimal(str(pos.quantity)) * self.fee_percent
            fee = entry_fee + exit_fee
            pos.profit_loss = float(Decimal(str(pos.profit_loss)) - fee)
            total_fees += fee
            
            # Update balance
            current_balance += Decimal(str(pos.profit_loss))
            
            # Calculate percent P&L
            pos.profit_loss_percent = float(
                (Decimal(str(pos.profit_loss)) / 
                (Decimal(str(pos.entry_price)) * Decimal(str(pos.quantity)))) * Decimal('100')
            )
            
            # Log
            logger.info(f"CLOSE AT END: {pos.direction.value} {pos.symbol} closed at {pos.exit_price}, "
                        f"P&L: ${pos.profit_loss:.2f} ({pos.profit_loss_percent:.2f}%)")
            
            # Move to closed positions
            closed_positions.append(pos)
        
        # Save results
        self.positions = closed_positions
        self.equity_curve = equity_curve
        self.final_balance = float(current_balance)
        self.total_fees = float(total_fees)
        
        logger.info(f"Trading simulation completed. Final balance: ${float(current_balance):.2f}, "
                    f"P&L: ${float(current_balance - self.initial_balance):.2f}, "
                    f"Total fees: ${float(total_fees):.2f}")
        
        return closed_positions, equity_curve

    def analyze_results(self):
        """
        Analyze backtest results and generate performance metrics
        """
        if not self.positions:
            logger.warning("No positions to analyze")
            return None
        
        # Extract performance metrics
        total_trades = len(self.positions)
        winning_trades = len([p for p in self.positions if p.profit_loss and p.profit_loss > 0])
        losing_trades = len([p for p in self.positions if p.profit_loss and p.profit_loss <= 0])
        
        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average profit/loss
        profit_trades = [p.profit_loss_percent for p in self.positions if p.profit_loss and p.profit_loss > 0]
        loss_trades = [p.profit_loss_percent for p in self.positions if p.profit_loss and p.profit_loss <= 0]
        
        avg_profit = sum(profit_trades) / len(profit_trades) if profit_trades else 0
        avg_loss = sum(loss_trades) / abs(len(loss_trades)) if loss_trades else 0
        
        # Calculate drawdown
        equity_values = [balance for _, balance in self.equity_curve]
        max_dd = 0
        peak = equity_values[0]
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Calculate profit factor
        total_profit = sum([p.profit_loss for p in self.positions if p.profit_loss and p.profit_loss > 0])
        total_loss = abs(sum([p.profit_loss for p in self.positions if p.profit_loss and p.profit_loss <= 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Create results object
        results = BacktestResults(
            initial_balance=float(self.initial_balance),
            final_balance=self.final_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_profit_percent=avg_profit,
            avg_loss_percent=avg_loss,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            total_fees=self.total_fees
        )
        
        # Log summary
        logger.info("\n===== BACKTEST RESULTS =====")
        logger.info(f"Initial Balance: ${float(self.initial_balance):.2f}")
        logger.info(f"Final Balance: ${self.final_balance:.2f}")
        logger.info(f"Net Profit/Loss: ${self.final_balance - float(self.initial_balance):.2f} "
                    f"({((self.final_balance / float(self.initial_balance)) - 1) * 100:.2f}%)")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades} ({win_rate*100:.2f}%)")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Average Profit: {avg_profit:.2f}%")
        logger.info(f"Average Loss: {avg_loss:.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Maximum Drawdown: {max_dd*100:.2f}%")
        logger.info(f"Total Fees: ${self.total_fees:.2f}")
        logger.info("===========================\n")
        
        return results
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve and save to file if path provided"""
        if not self.equity_curve:
            logger.warning("No equity data to plot")
            return
        
        # Convert timestamps to datetime for better readability
        times = [datetime.fromtimestamp(ts/1000) for ts, _ in self.equity_curve]
        balances = [balance for _, balance in self.equity_curve]
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, balances)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Balance (USDT)')
        plt.grid(True)
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()

async def run_backtest(config_path: str, data_file: str, symbol: str = "BTCUSDT", config_symbol: str = "BTC_USDT"):
    """
    Run a complete backtest
    
    Args:
        config_path: Path to configuration file
        data_file: Path to historical data CSV file
        symbol: Trading symbol in API format (e.g., "BTCUSDT")
        config_symbol: Trading symbol in config format (e.g., "BTC_USDT")
    """
    # Initialize backtester
    backtester = Backtester(config_path, data_file)
    
    # Load and process data
    kline_objects = backtester.load_historical_data()
    processed_df = await backtester.process_data_async(kline_objects, symbol=symbol)
    
    # Generate signals
    signals = await backtester.generate_signals(processed_df, symbol=symbol, config_symbol=config_symbol)
    
    # Simulate trading
    positions, equity_curve = backtester.simulate_trading(signals, processed_df)
    
    # Analyze results
    results = backtester.analyze_results()
    
    # Create basic results if no trades were executed
    if results is None:
        logger.info("No trade results to analyze. Creating default results.")
        results = BacktestResults(
            initial_balance=float(backtester.initial_balance),
            final_balance=float(backtester.current_balance),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_profit_percent=0.0,
            avg_loss_percent=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            total_fees=0.0
        )
    
    # Plot equity curve
    results_dir = Path("backtest_results")
    results_dir.mkdir(exist_ok=True)
    backtester.plot_equity_curve(
        save_path=str(results_dir / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    )
    
    return results, positions, equity_curve

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot Backtester')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to historical data CSV file')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol in API format')
    parser.add_argument('--config-symbol', type=str, default='BTC_USDT', help='Trading symbol in config format')
    
    args = parser.parse_args()
    
    # Run backtest
    asyncio.run(run_backtest(args.config, args.data, args.symbol, args.config_symbol)) 