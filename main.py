import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm
import warnings
import sys
warnings.filterwarnings('ignore')

def log_message(msg):
    print(msg)
    sys.stdout.flush()

def debug_log(msg):
    print(f"DEBUG: {msg}")
    sys.stdout.flush()

class AdvancedPremiumArbitrageStrategy:
    def __init__(self, initial_capital=100000, base_risk_per_trade=0.03):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.base_risk_per_trade = base_risk_per_trade
        self.positions = {}
        self.trades_history = []
        self.ticker_metrics = {}
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'max_drawdown': 0
        }
        self.intraday_trade_count = {}

    def initialize_ticker_metrics(self, ticker):
        if ticker not in self.ticker_metrics:
            self.ticker_metrics[ticker] = {
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'profit_factor': 1.0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_trades': 0,
                'intraday_trades': 0
            }

    def calculate_position_size(self, ticker, option_price, volatility, win_rate):
        self.initialize_ticker_metrics(ticker)
        metrics = self.ticker_metrics[ticker]
        
        # More aggressive base risk for higher returns
        base_risk = self.capital * self.base_risk_per_trade * 1.5
        
        # Adjust for intraday trading frequency
        if ticker in self.intraday_trade_count:
            daily_trades = self.intraday_trade_count[ticker].get(datetime.now().date(), 0)
            frequency_factor = max(0.7, 1 - (daily_trades * 0.05))  # Less reduction in size
        else:
            frequency_factor = 1.0
        
        # More aggressive volatility adjustment
        vol_factor = np.clip(volatility / 40, 0.7, 2.5)  # Higher max factor
        
        # Performance-based adjustment
        streak_factor = 1.0
        if metrics['consecutive_wins'] >= 2:
            streak_factor = 1.5  # More aggressive scaling on wins
        elif metrics['consecutive_losses'] >= 2:
            streak_factor = 0.8
        
        # Calculate final position size
        adjusted_risk = base_risk * vol_factor * streak_factor * frequency_factor
        contracts = int(adjusted_risk / (option_price * 100))
        
        # Scale based on capital growth
        capital_factor = max(1.0, (self.capital / self.initial_capital) ** 1.5)  # More aggressive scaling
        contracts = int(contracts * capital_factor)
        
        return max(1, min(contracts, 30))  # Increased max contracts

    def calculate_intraday_indicators(self, data):
        close = data['Close']
        volume = data['Volume']
        
        # Fast moving averages
        ema_1m = close.ewm(span=1, adjust=False).mean()
        ema_5m = close.ewm(span=5, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Volume analysis
        volume_sma = volume.rolling(window=5).mean()
        volume_ratio = volume / volume_sma
        
        return {
            'ema_1m': ema_1m,
            'ema_5m': ema_5m,
            'rsi': rsi,
            'volume_ratio': volume_ratio
        }

    def calculate_intraday_volatility_score(self, data):
        returns = data['Close'].pct_change()
        vol_5m = returns.rolling(5).std() * np.sqrt(390)
        vol_15m = returns.rolling(15).std() * np.sqrt(390)
        vol_ratio = vol_5m / vol_15m
        
        vol_regime = np.where(vol_ratio > 1.2, 'high',
                            np.where(vol_ratio < 0.8, 'low', 'normal'))
        
        vol_cluster = returns.abs().rolling(5).mean() > returns.abs().rolling(15).mean()
        
        return vol_ratio.iloc[-1], vol_regime[-1], vol_cluster.iloc[-1]

    def should_sell_premium(self, ticker, data):
        """Enhanced premium selling decision with more aggressive entry"""
        indicators = self.calculate_intraday_indicators(data)
        vol_ratio, vol_regime, vol_cluster = self.calculate_intraday_volatility_score(data)
        
        # Market condition analysis
        current_price = data['Close'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1]
        volume_ratio = indicators['volume_ratio'].iloc[-1]
        
        # More aggressive entry conditions
        high_volatility = vol_ratio > 1.1  # Lower threshold
        volume_surge = volume_ratio > 1.3  # Lower threshold
        
        # Trend analysis
        ema_1m = indicators['ema_1m'].iloc[-1]
        ema_5m = indicators['ema_5m'].iloc[-1]
        trend_strength = abs(ema_1m - ema_5m) / ema_5m
        
        # Combined entry conditions
        technical_condition = (
            high_volatility and
            volume_surge and
            trend_strength > 0.0008 and  # Lower threshold
            (25 < current_rsi < 75)  # Wider RSI range
        )
        
        # Trade frequency check
        current_date = datetime.now().date()
        if ticker not in self.intraday_trade_count:
            self.intraday_trade_count[ticker] = {}
        daily_trades = self.intraday_trade_count[ticker].get(current_date, 0)
        within_trade_limit = daily_trades < 15  # Increased trade limit
        
        return technical_condition and within_trade_limit

    def should_buy_back(self, ticker, entry_price, current_price, data):
        """Enhanced exit strategy with faster profit taking"""
        profit_pct = (entry_price - current_price) / entry_price
        
        # Dynamic profit targets
        if self.ticker_metrics[ticker]['consecutive_wins'] >= 2:
            profit_target = 0.15  # Take profits faster after wins
        else:
            profit_target = 0.2
            
        # Tighter stop loss
        stop_loss = -0.3
        
        # Time-based exit (shorter holding periods)
        minutes_held = (data.index[-1] - self.positions[ticker]['entry_date']).total_seconds() / 60
        time_exit = minutes_held > 45  # Reduced holding time
        
        return profit_pct >= profit_target or profit_pct <= stop_loss or time_exit

    def execute_trade(self, ticker, action, price, contracts, timestamp):
        trade = {
            'ticker': ticker,
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'contracts': contracts,
            'value': price * contracts * 100
        }
        self.trades_history.append(trade)
        
        current_date = timestamp.date()
        if ticker not in self.intraday_trade_count:
            self.intraday_trade_count[ticker] = {}
        if current_date not in self.intraday_trade_count[ticker]:
            self.intraday_trade_count[ticker][current_date] = 0
        
        if action == 'sell':
            self.capital += trade['value']
            self.intraday_trade_count[ticker][current_date] += 1
            log_message(f"\nSold {contracts} contracts of {ticker} at ${price:.2f}")
            log_message(f"Daily trades for {ticker}: {self.intraday_trade_count[ticker][current_date]}")
            self.positions[ticker] = {
                'contracts': contracts,
                'entry_price': price,
                'entry_date': timestamp
            }
        else:  # buy
            self.capital -= trade['value']
            log_message(f"\nBought back {contracts} contracts of {ticker} at ${price:.2f}")
            
            # Calculate trade performance
            entry_trade = None
            for t in reversed(self.trades_history[:-1]):
                if t['ticker'] == ticker and t['action'] == 'sell':
                    entry_trade = t
                    break
                    
            if entry_trade:
                profit = entry_trade['value'] - trade['value']
                self.metrics['total_profit'] += profit
                self.metrics['total_trades'] += 1
                
                metrics = self.ticker_metrics[ticker]
                metrics['total_trades'] += 1
                
                if profit > 0:
                    self.metrics['winning_trades'] += 1
                    metrics['consecutive_wins'] += 1
                    metrics['consecutive_losses'] = 0
                    metrics['avg_win'] = (metrics['avg_win'] * (metrics['total_trades'] - 1) + profit) / metrics['total_trades']
                    log_message(f"Trade Profit: ${profit:.2f} ({(profit/entry_trade['value'])*100:.1f}%)")
                else:
                    self.metrics['losing_trades'] += 1
                    metrics['consecutive_wins'] = 0
                    metrics['consecutive_losses'] += 1
                    metrics['avg_loss'] = (metrics['avg_loss'] * (metrics['total_trades'] - 1) + profit) / metrics['total_trades']
                    log_message(f"Trade Loss: ${profit:.2f} ({(profit/entry_trade['value'])*100:.1f}%)")
                
                log_message(f"Current Capital: ${self.capital:,.2f}")
                log_message(f"Return: {((self.capital - self.initial_capital) / self.initial_capital * 100):.2f}%")

class AdvancedBacktester:
    def __init__(self, strategy, ticker):
        self.strategy = strategy
        self.ticker = ticker
        self.results = {}

    def get_recent_data(self):
        """Get most recent 8 days of intraday data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=8)
        
        try:
            data = yf.download(
                self.ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1m',
                progress=False
            )
            return data
        except Exception as e:
            log_message(f"Error fetching data for {self.ticker}: {str(e)}")
            return None

    def simulate_option_data(self, data):
        """Enhanced option data simulation with proper array handling and debugging"""
        try:
            debug_log(f"Starting option data simulation for {self.ticker}")
            debug_log(f"Input data shape: {data.shape}")
            
            # Calculate returns and volatility
            returns = data['Close'].pct_change().fillna(0)
            debug_log(f"Returns shape before conversion: {returns.shape}")
            
            # Convert returns to 1D numpy array
            returns_array = returns.to_numpy().flatten()
            debug_log(f"Returns array shape after flatten: {returns_array.shape}")
            
            # Calculate volatility
            intraday_vol = pd.Series(
                (returns.rolling(5).std() * np.sqrt(390)).to_numpy().flatten(),
                index=data.index
            ).fillna(method='bfill')
            
            # Calculate implied volatility
            mean_vol = float(intraday_vol.mean())
            vol_diff = pd.Series(mean_vol - intraday_vol, index=data.index)
            random_noise = pd.Series(np.random.normal(0, 0.03, size=len(data)), index=data.index)
            iv = intraday_vol + 0.1 * vol_diff + random_noise
            
            # Calculate option prices
            atm_factor = pd.Series(1 - np.abs(returns_array), index=data.index)
            base_prices = pd.Series(data['Close'].to_numpy().flatten() * 0.05, index=data.index)
            option_prices = base_prices * (1 + iv) * atm_factor
            
            # Calculate time decay
            time_decay = pd.Series(index=data.index)
            for idx in data.index:
                hour = idx.hour
                minute = idx.minute
                minutes_to_close = (16 - hour) * 60 - minute
                time_decay[idx] = np.exp(-0.0001 * (390 - minutes_to_close))
            
            # Apply time decay
            option_prices = option_prices * time_decay
            
            # Create result DataFrame
            result = pd.DataFrame({
                'implied_volatility': iv * 100,  # Convert to percentage
                'option_price': option_prices
            }, index=data.index)
            
            debug_log(f"Final result shape: {result.shape}")
            return result
            
        except Exception as e:
            debug_log(f"Error in simulate_option_data: {str(e)}")
            debug_log(f"Error occurred at line: {sys.exc_info()[2].tb_lineno}")
            raise

    def run(self):
        log_message(f"\nProcessing {self.ticker}...")
        
        data = self.get_recent_data()
        if data is None or len(data) < 100:
            log_message(f"Insufficient data for {self.ticker}")
            return
            
        option_data = self.simulate_option_data(data)
        data = pd.concat([data, option_data], axis=1)
        
        self.strategy.initialize_ticker_metrics(self.ticker)
        
        for i in tqdm(range(20, len(data)), desc=f"Processing {self.ticker}"):
            current_data = data.iloc[:i+1]
            
            if self.ticker in self.strategy.positions:
                current_price = current_data['option_price'].iloc[-1]
                position = self.strategy.positions[self.ticker]
                
                if self.strategy.should_buy_back(self.ticker, position['entry_price'], 
                                               current_price, current_data):
                    self.strategy.execute_trade(self.ticker, 'buy', current_price,
                                             position['contracts'], current_data.index[-1])
                    del self.strategy.positions[self.ticker]
            
            elif self.ticker not in self.strategy.positions:
                if self.strategy.should_sell_premium(self.ticker, current_data):
                    price = current_data['option_price'].iloc[-1]
                    volatility = (data['Close'].pct_change().std() * np.sqrt(252 * 390) * 100).item()
                    contracts = self.strategy.calculate_position_size(
                        self.ticker, price, volatility,
                        self.strategy.ticker_metrics[self.ticker]['profit_factor']
                    )
                    self.strategy.execute_trade(self.ticker, 'sell', price, contracts,
                                             current_data.index[-1])
        
        self._calculate_performance()

    def _calculate_performance(self):
        equity_curve = [self.strategy.initial_capital]
        daily_returns = []
        
        for trade in self.strategy.trades_history:
            if trade['action'] == 'sell':
                equity_curve.append(equity_curve[-1] + trade['value'])
            else:
                equity_curve.append(equity_curve[-1] - trade['value'])
                daily_returns.append((equity_curve[-1] - equity_curve[-2]) / equity_curve[-2])
        
        equity_curve = np.array(equity_curve)
        drawdowns = np.maximum.accumulate(equity_curve) - equity_curve
        self.strategy.metrics['max_drawdown'] = drawdowns.max()
        
        if len(daily_returns) > 0:
            returns_array = np.array(daily_returns)
            sharpe_ratio = (np.mean(returns_array) - 0.02/252) / (np.std(returns_array) * np.sqrt(252))
            downside_returns = returns_array[returns_array < 0]
            sortino_ratio = (np.mean(returns_array) - 0.02/252) / (np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 0 else np.nan
            dd_end = np.argmax(drawdowns)
            dd_start = np.argmax(equity_curve[:dd_end+1])
            max_dd_duration = (dd_end - dd_start) if dd_end > dd_start else 0
        else:
            sharpe_ratio = sortino_ratio = max_dd_duration = np.nan
        
        self.results = {
            'final_capital': self.strategy.capital,
            'total_return': (self.strategy.capital - self.strategy.initial_capital) / self.strategy.initial_capital * 100,
            'annual_return': ((1 + self.strategy.capital/self.strategy.initial_capital) ** (252/8) - 1) * 100 if len(daily_returns) > 0 else 0,
            'win_rate': self.strategy.metrics['winning_trades'] / self.strategy.metrics['total_trades'] if self.strategy.metrics['total_trades'] > 0 else 0,
            'max_drawdown': self.strategy.metrics['max_drawdown'],
            'max_drawdown_pct': (self.strategy.metrics['max_drawdown'] / self.strategy.initial_capital) * 100,
            'max_drawdown_duration': max_dd_duration,
            'total_trades': self.strategy.metrics['total_trades'],
            'avg_profit_per_trade': self.strategy.metrics['total_profit'] / self.strategy.metrics['total_trades'] if self.strategy.metrics['total_trades'] > 0 else 0,
            'profit_factor': sum(t['value'] for t in self.strategy.trades_history if t['action'] == 'sell') / abs(sum(t['value'] for t in self.strategy.trades_history if t['action'] == 'buy')) if self.strategy.metrics['total_trades'] > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }

if __name__ == "__main__":
    log_message("\nInitializing Advanced Intraday Premium Arbitrage Strategy Backtest")
    log_message("=" * 50)
    
    log_message("Using most recent 8 days of intraday data")
    log_message("High-frequency trading with 1-minute resolution")
    
    # Initialize strategy with single ticker
    ticker = "MSTR"  # Example ticker
    strategy = AdvancedPremiumArbitrageStrategy(initial_capital=100000)
    
    # Setup and run backtest
    backtester = AdvancedBacktester(strategy=strategy, ticker=ticker)
    
    # Run backtest
    backtester.run()
    
    # Print results
    log_message("\nBacktest Results:")
    log_message("=" * 50)
    log_message(f"Final Capital: ${backtester.results['final_capital']:,.2f}")
    log_message(f"Total Return: {backtester.results['total_return']:.2f}%")
    log_message(f"Annualized Return: {backtester.results['annual_return']:.2f}%")
    log_message(f"Win Rate: {backtester.results['win_rate']*100:.2f}%")
    log_message(f"Max Drawdown: ${backtester.results['max_drawdown']:,.2f} ({backtester.results['max_drawdown_pct']:.2f}%)")
    log_message(f"Max Drawdown Duration: {backtester.results['max_drawdown_duration']:.0f} periods")
    log_message(f"Total Trades: {backtester.results['total_trades']}")
    log_message(f"Average Profit per Trade: ${backtester.results['avg_profit_per_trade']:,.2f}")
    log_message(f"Profit Factor: {backtester.results['profit_factor']:.2f}")
    log_message(f"Sharpe Ratio: {backtester.results['sharpe_ratio']:.2f}")
    log_message(f"Sortino Ratio: {backtester.results['sortino_ratio']:.2f}")
