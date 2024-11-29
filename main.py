import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import sys
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Download data with retries
def download_with_retry(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval='1m',
                progress=False,
                group_by='ticker'
            )
            if len(data) > 0:
                # Flatten columns if MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(0)
                return data
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed, retrying in 5 seconds...")
                time.sleep(5)
            else:
                logging.error(f"Failed to download data after {max_retries} attempts: {str(e)}")
    return None

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
        self.ml_model = None
        self.scaler = None
        self.ticker = None  # To store the current ticker

    def initialize_ticker_metrics(self, ticker):
        self.ticker = ticker
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

    def train_ml_model(self, historical_data, ticker):
        """
        Train a Random Forest model using historical data.
        """
        logging.info(f"Training machine learning model for {ticker}...")

        # Ensure 'Close' is a Series
        if 'Close' in historical_data.columns:
            close = historical_data['Close']
        else:
            logging.error("'Close' column not found in historical_data.")
            return

        if isinstance(close, pd.DataFrame):
            close = close.squeeze()

        # Feature engineering
        historical_data['returns'] = close.pct_change()
        historical_data['rolling_mean'] = close.rolling(window=20).mean()
        historical_data['rolling_std'] = close.rolling(window=20).std()
        historical_data['z_score'] = (close - historical_data['rolling_mean']) / historical_data['rolling_std']
        historical_data['bollinger_upper'] = historical_data['rolling_mean'] + 2 * historical_data['rolling_std']
        historical_data['bollinger_lower'] = historical_data['rolling_mean'] - 2 * historical_data['rolling_std']
        historical_data['bollinger_bandwidth'] = historical_data['bollinger_upper'] - historical_data['bollinger_lower']

        # Target variable: Favorable market conditions (1) or not (0)
        historical_data['favorable'] = (
            (historical_data['z_score'].abs() < 1.0) &  # Within 1 standard deviation
            (historical_data['bollinger_bandwidth'] > close * 0.02)  # Sufficient volatility
        ).astype(int)

        # Prepare data for ML
        features = ['returns', 'rolling_mean', 'rolling_std', 'z_score', 'bollinger_bandwidth']
        target = 'favorable'
        data = historical_data.dropna()  # Drop rows with NaN values

        if data.empty:
            logging.error("Not enough data to train the model.")
            return

        X = data[features]
        y = data[target]
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Use GridSearchCV for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_scaled, y)
        self.ml_model = grid_search.best_estimator_

        accuracy = self.ml_model.score(X_scaled, y)
        logging.info(f"Model trained with accuracy: {accuracy:.2f}")

    def calculate_position_size(self, ticker, option_price, volatility):
        self.initialize_ticker_metrics(ticker)
        metrics = self.ticker_metrics[ticker]
        
        base_risk = self.capital * self.base_risk_per_trade

        # Adjust risk based on volatility
        vol_factor = np.clip(volatility / 40, 0.7, 2.5)
        adjusted_risk = base_risk * vol_factor

        # Calculate the number of contracts
        contracts = int(adjusted_risk / (option_price * 100))

        # Ensure at least 1 contract and limit maximum
        return max(1, min(contracts, 30))

    def calculate_intraday_indicators(self, data):
        if 'Close' in data.columns:
            close = data['Close']
        else:
            logging.error("'Close' column not found in data.")
            return

        if 'Volume' in data.columns:
            volume = data['Volume']
        else:
            logging.error("'Volume' column not found in data.")
            return

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

    def should_sell_premium(self, data, indicators):
        """
        Decide whether to sell premiums using a combination of statistics and ML predictions.
        """
        close = data['Close']
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        z_score = (close - rolling_mean) / rolling_std
        z_score = z_score.iloc[-1]

        # Predict favorable conditions using ML
        features = np.array([[
            close.pct_change().iloc[-1],
            rolling_mean.iloc[-1],
            rolling_std.iloc[-1],
            z_score,
            (rolling_mean.iloc[-1] + 2 * rolling_std.iloc[-1]) - (rolling_mean.iloc[-1] - 2 * rolling_std.iloc[-1])
        ]])
        prediction = self.ml_model.predict(self.scaler.transform(features))[0]

        # Additional technical indicators
        current_rsi = indicators['rsi'].iloc[-1]
        volume_ratio = indicators['volume_ratio'].iloc[-1]

        return prediction == 1 and (25 < current_rsi < 75) and volume_ratio > 1

    def should_buy_back(self, ticker, entry_price, current_price, data):
        profit_pct = (entry_price - current_price) / entry_price
        stop_loss = -0.3  # 30% stop loss
        profit_target = 0.2  # 20% profit target

        # Time-based exit (shorter holding periods)
        minutes_held = (data.index[-1] - self.positions[ticker]['entry_date']).total_seconds() / 60
        time_exit = minutes_held > 45

        return profit_pct >= profit_target or profit_pct <= stop_loss or time_exit

    def execute_trade(self, ticker, action, price, contracts, timestamp):
        trade_value = price * contracts * 100
        trade = {
            'ticker': ticker,
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'contracts': contracts,
            'value': trade_value
        }
        self.trades_history.append(trade)

        if action == 'sell':
            self.capital += trade_value
            self.positions[ticker] = {
                'contracts': contracts,
                'entry_price': price,
                'entry_date': timestamp
            }
            logging.info(f"Sold {contracts} contracts of {ticker} at ${price:.2f} on {timestamp}")
        else:  # buy
            self.capital -= trade_value
            position = self.positions.pop(ticker, None)
            if position:
                profit = (position['entry_price'] - price) * contracts * 100
                self.metrics['total_profit'] += profit
                self.metrics['total_trades'] += 1
                if profit > 0:
                    self.metrics['winning_trades'] += 1
                else:
                    self.metrics['losing_trades'] += 1
                logging.info(f"Bought back {contracts} contracts of {ticker} at ${price:.2f} on {timestamp}")
                logging.info(f"Trade {'Profit' if profit > 0 else 'Loss'}: ${profit:.2f}")

class AdvancedBacktester:
    def __init__(self, strategy, ticker):
        self.strategy = strategy
        self.ticker = ticker
        self.results = {}

    def get_recent_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=8)
        logging.info(f"Downloading data for {self.ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        data = download_with_retry(self.ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if data is None or len(data) < 100:
            logging.warning(f"Insufficient data for {self.ticker}")
            return None
        return data

    def simulate_option_data(self, data):
        """
        Simulate option prices based on underlying stock data.
        """
        close_prices = data['Close']
        returns = close_prices.pct_change()
        implied_volatility = returns.rolling(window=30).std() * np.sqrt(252)  # Annualized volatility
        option_prices = implied_volatility * close_prices * 0.1  # Simplified option pricing model
        simulated_data = pd.DataFrame({
            'implied_volatility': implied_volatility * 100,
            'option_price': option_prices
        }, index=data.index)
        return simulated_data

    def run(self):
        logging.info(f"Starting backtest for {self.ticker}...")
        data = self.get_recent_data()
        if data is None:
            return

        # Train ML model
        self.strategy.train_ml_model(data.copy(), self.ticker)
        if self.strategy.ml_model is None:
            logging.error("Machine learning model was not trained.")
            return

        # Simulate option data
        option_data = self.simulate_option_data(data)
        data = data.join(option_data)

        # Initialize indicators
        indicators = self.strategy.calculate_intraday_indicators(data)
        if indicators is None:
            return

        for i in range(20, len(data)):
            current_data = data.iloc[:i+1]
            current_indicators = {key: val.iloc[:i+1] for key, val in indicators.items()}
            timestamp = current_data.index[-1]
            current_price = current_data['option_price'].iloc[-1]
            volatility = current_data['implied_volatility'].iloc[-1]

            if self.ticker in self.strategy.positions:
                position = self.strategy.positions[self.ticker]
                if self.strategy.should_buy_back(self.ticker, position['entry_price'], current_price, current_data):
                    self.strategy.execute_trade(self.ticker, 'buy', current_price, position['contracts'], timestamp)
            else:
                if self.strategy.should_sell_premium(current_data, current_indicators):
                    contracts = self.strategy.calculate_position_size(self.ticker, current_price, volatility)
                    self.strategy.execute_trade(self.ticker, 'sell', current_price, contracts, timestamp)

        # Calculate performance metrics
        self._calculate_performance()

    def _calculate_performance(self):
        total_return = (self.strategy.capital - self.strategy.initial_capital) / self.strategy.initial_capital * 100
        logging.info(f"Backtest completed for {self.ticker}. Total Return: {total_return:.2f}%")
        logging.info(f"Total Trades: {self.strategy.metrics['total_trades']}")
        logging.info(f"Winning Trades: {self.strategy.metrics['winning_trades']}")
        logging.info(f"Losing Trades: {self.strategy.metrics['losing_trades']}")
        self.results['total_return'] = total_return

if __name__ == "__main__":
    logging.info("Initializing Advanced Premium Arbitrage Strategy Backtest")
    ticker = "GME"  # You can change this to any ticker symbol
    strategy = AdvancedPremiumArbitrageStrategy(initial_capital=20000)
    backtester = AdvancedBacktester(strategy=strategy, ticker=ticker)
    backtester.run()
