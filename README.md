# Trading Engine - Complete Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Using the System](#using-the-system)
6. [Creating Custom Rules](#creating-custom-rules)
7. [Backtesting](#backtesting)
8. [Dashboard Features](#dashboard-features)
9. [Troubleshooting](#troubleshooting)
10. [API Documentation](#api-documentation)

## 🎯 System Overview

The Trading Engine is a modular algorithmic trading system that:
- Connects to real-time market data feeds (Alpaca, Polygon.io, IEX Cloud)
- Evaluates custom trading rules without coding
- Generates automated alerts with entry/exit points
- Provides a real-time web dashboard
- Includes backtesting capabilities for strategy validation

### Key Components

1. **Trading Engine Core** (`trading_engine.py`)
   - Real-time data processing
   - Rule evaluation engine
   - Alert generation system
   - Database management
   - WebSocket server for live updates

2. **Configuration File** (`trading_config.yaml`)
   - Trading rules definition
   - API credentials
   - Risk management parameters
   - Watchlist management

3. **Web Dashboard** (React-based)
   - Live alert monitoring
   - Performance metrics
   - Historical data visualization
   - Search and filtering capabilities

4. **Backtest Module** (`backtest.py`)
   - Historical strategy testing
   - Performance analytics
   - Risk metrics calculation
   - Result visualization

## 📦 Prerequisites

### Required Software
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)
- Text editor (VS Code, Sublime Text, etc.)

### Required Accounts
Choose one of the following data providers:

1. **Alpaca Markets** (Recommended for beginners)
   - Free paper trading account
   - Real-time data included
   - Sign up at: https://alpaca.markets

2. **Polygon.io**
   - Free tier available (limited requests)
   - Professional data quality
   - Sign up at: https://polygon.io

3. **IEX Cloud**
   - Free tier with 50,000 messages/month
   - Sign up at: https://iexcloud.io

## 🚀 Installation

### Step 1: Set Up Project Directory

```bash
# Create project directory
mkdir trading-engine
cd trading-engine

# Create subdirectories
mkdir data logs config
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

Create a `requirements.txt` file:

```txt
# Core dependencies
aiohttp==3.9.1
pandas==2.0.3
numpy==1.24.3
pyyaml==6.0.1

# Web server
flask==3.0.0
flask-socketio==5.3.5
flask-cors==4.0.0
python-socketio==5.10.0

# Database
sqlite3  # Usually included with Python

# Market data
websocket-client==1.6.4
requests==2.31.0

# Optional - for specific providers
alpaca-py==0.13.3  # If using Alpaca
polygon-api-client==1.12.4  # If using Polygon

# Backtesting
yfinance==0.2.33
matplotlib==3.7.2
seaborn==0.12.2

# Technical indicators (optional)
ta-lib==0.4.28  # May require additional installation steps
scipy==1.11.4
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Download System Files

Create the main files in your project directory:

1. `trading_engine.py` - Main engine code
2. `trading_config.yaml` - Configuration file
3. `backtest.py` - Backtesting module
4. `dashboard.html` - Web interface (optional)

## ⚙️ Configuration

### Basic Configuration Structure

The `trading_config.yaml` file controls all aspects of the system:

```yaml
# API Configuration
api:
  provider: alpaca  # Options: alpaca, polygon, iex
  key: YOUR_API_KEY_HERE
  secret: YOUR_API_SECRET_HERE  # Only for Alpaca
  
# Watchlist - Stocks to monitor
watchlist:
  - AAPL   # Apple
  - TSLA   # Tesla
  - NVDA   # Nvidia
  - AMD    # AMD
  - SPY    # S&P 500 ETF
  
# Scan interval in seconds
scan_interval: 5  # Check for signals every 5 seconds

# Trading Rules (see detailed section below)
rules:
  - name: "Rule Name"
    enabled: true
    conditions: [...]
    action: [...]

# Risk Management
risk_management:
  max_daily_loss_pct: 0.02  # Stop trading after 2% daily loss
  max_position_size_pct: 0.20  # Max 20% of capital per trade
  trailing_stop_enabled: true
  trailing_stop_pct: 0.015  # 1.5% trailing stop

# Alerts Configuration
alerts:
  sound_enabled: true
  email:
    enabled: false
    smtp_server: smtp.gmail.com
    smtp_port: 587
    from_email: your-email@gmail.com
    to_email: alerts@gmail.com
```

### API Provider Setup

#### Alpaca Configuration
```yaml
api:
  provider: alpaca
  key: PKtest_YOUR_KEY  # Paper trading key starts with PKtest_
  secret: YOUR_SECRET_KEY
  base_url: https://paper-api.alpaca.markets  # Paper trading URL
```

#### Polygon Configuration
```yaml
api:
  provider: polygon
  key: YOUR_POLYGON_API_KEY
  # No secret needed for Polygon
```

#### IEX Cloud Configuration
```yaml
api:
  provider: iex
  key: pk_YOUR_PUBLISHABLE_KEY  # or sk_ for secret key
  sandbox: true  # Use sandbox for testing
```

## 🎮 Using the System

### Starting the Trading Engine

#### Method 1: Direct Launch
```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run the engine
python trading_engine.py
```

#### Method 2: Create a Launch Script

Create `start_trading.py`:

```python
#!/usr/bin/env python
import subprocess
import webbrowser
import time
import os
import sys

def start_trading_engine():
    """Start the trading engine with proper error handling"""
    
    print("=" * 60)
    print("🚀 TRADING ENGINE STARTUP")
    print("=" * 60)
    
    # Check configuration file exists
    if not os.path.exists('trading_config.yaml'):
        print("❌ Error: trading_config.yaml not found!")
        print("Please create the configuration file first.")
        sys.exit(1)
    
    # Start the backend
    print("\n📊 Starting trading engine backend...")
    try:
        backend = subprocess.Popen(
            [sys.executable, 'trading_engine.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for startup
        time.sleep(3)
        
        # Check if process is still running
        if backend.poll() is not None:
            print("❌ Backend failed to start!")
            sys.exit(1)
            
        print("✅ Backend started successfully!")
        
        # Open dashboard
        print("\n🌐 Opening dashboard in browser...")
        webbrowser.open('http://localhost:5000')
        
        print("\n" + "=" * 60)
        print("✅ TRADING ENGINE IS RUNNING")
        print("=" * 60)
        print("\n📍 Dashboard URL: http://localhost:5000")
        print("📍 API Endpoint: http://localhost:5000/api")
        print("\n⚠️  Press Ctrl+C to stop the engine")
        print("=" * 60)
        
        # Keep running
        backend.wait()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down trading engine...")
        backend.terminate()
        print("✅ Trading engine stopped successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_trading_engine()
```

Run with:
```bash
python start_trading.py
```

### System Operations

#### 1. Pre-Market Preparation (Before 9:30 AM ET)

```bash
# Check system status
python trading_engine.py --status

# Update watchlist in config
# Edit trading_config.yaml and add/remove tickers

# Run pre-market scan
python trading_engine.py --premarket
```

#### 2. Live Trading Hours (9:30 AM - 4:00 PM ET)

The system will automatically:
- Scan all tickers in your watchlist
- Evaluate rules every 5 seconds (configurable)
- Generate alerts when conditions are met
- Update the dashboard in real-time
- Save alerts to the database

#### 3. After-Hours Analysis

```bash
# Generate daily report
python trading_engine.py --report today

# Export alerts to CSV
python trading_engine.py --export alerts.csv

# Review performance
python trading_engine.py --stats
```

### Monitoring the System

#### Dashboard Access
Open your browser and navigate to:
```
http://localhost:5000
```

#### Dashboard Features:
- **Live Status Indicator**: Shows if engine is connected
- **Alert Feed**: Real-time alerts with all details
- **Statistics Panel**: Win rate, total alerts, confidence scores
- **Search Bar**: Filter by ticker symbol
- **Setup Filter**: Filter by strategy type

#### API Endpoints

You can also access data programmatically:

```bash
# Get recent alerts
curl http://localhost:5000/api/alerts

# Get market data for a ticker
curl http://localhost:5000/api/market/AAPL

# Get system status
curl http://localhost:5000/api/status
```

## 📝 Creating Custom Rules

### Rule Structure

Each rule in `trading_config.yaml` has three main parts:

```yaml
rules:
  - name: "My Custom Rule"        # Unique identifier
    enabled: true                  # Can be toggled on/off
    conditions:                    # ALL conditions must be true
      - field: "field_name"
        operator: "comparison"
        value: threshold
    action:                        # What to do when triggered
      type: buy                    # buy or sell
      stop_loss_pct: 0.02         # 2% stop loss
      target_pct: 0.05            # 5% profit target
      confidence: 0.75            # Confidence score (0-1)
```

### Available Fields for Conditions

#### Market Data Fields
- `market.price` - Current price
- `market.volume` - Current volume
- `market.bid` - Bid price
- `market.ask` - Ask price
- `market.high` - Day's high
- `market.low` - Day's low

#### Calculated Indicators
- `indicators.sma_5` - 5-period moving average
- `indicators.sma_20` - 20-period moving average
- `indicators.volume_avg_5` - 5-period volume average
- `indicators.volume_ratio` - Current volume / average volume
- `indicators.price_change_pct` - Price change percentage
- `indicators.rsi` - Relative Strength Index
- `indicators.macd` - MACD value

#### Stock Information
- `info.float_shares` - Float shares
- `info.short_interest` - Short interest ratio
- `info.market_cap` - Market capitalization
- `info.avg_volume_10d` - 10-day average volume

### Operators

- `>` - Greater than
- `<` - Less than
- `>=` - Greater than or equal
- `<=` - Less than or equal
- `==` - Equal to
- `!=` - Not equal to

### Example Rules

#### 1. Simple Price Breakout
```yaml
- name: "Price Breakout"
  enabled: true
  conditions:
    - field: market.price
      operator: ">"
      value: 100
    - field: indicators.volume_ratio
      operator: ">"
      value: 1.5
  action:
    type: buy
    stop_loss_pct: 0.02
    target_pct: 0.05
    confidence: 0.70
```

#### 2. Volume Spike with RSI
```yaml
- name: "Oversold Bounce"
  enabled: true
  conditions:
    - field: indicators.rsi
      operator: "<"
      value: 30
    - field: indicators.volume_ratio
      operator: ">"
      value: 2.0
    - field: market.price
      operator: ">"
      value: 5  # Minimum price $5
  action:
    type: buy
    stop_loss_pct: 0.015
    target_pct: 0.03
    confidence: 0.65
```

#### 3. Multi-Timeframe Setup
```yaml
- name: "Multi-Timeframe Momentum"
  enabled: true
  conditions:
    - field: market.price
      operator: ">"
      value: indicators.sma_5
    - field: indicators.sma_5
      operator: ">"
      value: indicators.sma_20
    - field: indicators.volume_ratio
      operator: ">"
      value: 1.2
  action:
    type: buy
    stop_loss_pct: 0.025
    target_pct: 0.08
    confidence: 0.80
```

## 📊 Backtesting

### Running a Backtest

```bash
# Basic backtest
python backtest.py

# Backtest specific tickers
python backtest.py --tickers AAPL,TSLA,NVDA

# Backtest specific date range
python backtest.py --start 2024-01-01 --end 2024-12-31

# Backtest with custom capital
python backtest.py --capital 25000
```

### Backtest Configuration

Add to `trading_config.yaml`:

```yaml
backtest:
  enabled: true
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_capital: 10000
  position_size_pct: 0.10  # Use 10% per position
  max_positions: 5         # Max 5 concurrent positions
  commission: 0.001        # 0.1% commission per trade
  slippage: 0.001         # 0.1% slippage assumption
```

### Interpreting Backtest Results

The backtest will generate:

1. **Summary Statistics**
   - Total return
   - Win rate
   - Profit factor
   - Sharpe ratio
   - Maximum drawdown

2. **Trade Analysis**
   - Average win/loss
   - Best/worst trade
   - Average holding time
   - Win/loss streaks

3. **Visual Reports**
   - Equity curve
   - Drawdown chart
   - P&L distribution
   - Monthly returns heatmap

### Example Backtest Output

```
==================================================
📊 BACKTEST RESULTS
==================================================

💼 Overall Performance:
  • Initial Capital: $10,000.00
  • Final Capital: $12,543.67
  • Total Return: +25.44%
  • Sharpe Ratio: 1.85
  • Max Drawdown: -8.32%

📈 Trade Statistics:
  • Total Trades: 147
  • Winning Trades: 89 (60.5%)
  • Losing Trades: 58 (39.5%)
  • Profit Factor: 1.92

💰 Average Performance:
  • Average Win: +$127.34
  • Average Loss: -$68.92
  • Average Hold Time: 2.3 days

🎯 Best/Worst:
  • Best Trade: +8.7% (NVDA)
  • Worst Trade: -3.2% (TSLA)
==================================================
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. API Connection Issues

**Problem**: "API connection failed"
```bash
# Check API credentials
python -c "import yaml; print(yaml.safe_load(open('trading_config.yaml'))['api'])"

# Test API connection
python test_connection.py
```

**Solution**:
- Verify API keys are correct
- Check if you're using paper trading keys for testing
- Ensure your IP is whitelisted (if required)

#### 2. No Alerts Generated

**Problem**: System running but no alerts

**Checklist**:
- Check if market is open (9:30 AM - 4:00 PM ET)
- Verify rules are enabled in config
- Lower threshold values for testing
- Check system logs: `tail -f logs/trading.log`

#### 3. Dashboard Not Loading

**Problem**: Can't access http://localhost:5000

**Solution**:
```bash
# Check if port 5000 is in use
netstat -an | grep 5000

# Use alternative port
python trading_engine.py --port 8080
```

#### 4. High CPU Usage

**Problem**: System using too much CPU

**Solution**:
- Increase `scan_interval` in config (e.g., from 5 to 10 seconds)
- Reduce number of tickers in watchlist
- Disable complex indicators if not needed

#### 5. Database Errors

**Problem**: "Database is locked" or similar

**Solution**:
```bash
# Reset database
mv trading_alerts.db trading_alerts.db.backup
python trading_engine.py --init-db
```

### Debug Mode

Enable detailed logging:

```python
# In trading_engine.py, change:
logging.basicConfig(level=logging.DEBUG)

# Or run with debug flag:
python trading_engine.py --debug
```

### Performance Optimization

#### 1. Optimize Watchlist
```yaml
# Instead of scanning 100 stocks, focus on high-probability setups
watchlist:
  # Liquid, high-volume stocks only
  - SPY
  - QQQ
  - AAPL
  - TSLA
  - NVDA
```

#### 2. Optimize Rules
```yaml
# Add pre-filters to reduce processing
rules:
  - name: "Optimized Rule"
    enabled: true
    pre_filters:  # Check these first (faster)
      - field: market.price
        operator: ">"
        value: 1  # Skip penny stocks
      - field: market.volume
        operator: ">"
        value: 100000  # Minimum volume
    conditions:  # Then check complex conditions
      - field: indicators.rsi
        operator: "<"
        value: 30
```

#### 3. Use Caching
```yaml
# Cache settings
cache:
  enabled: true
  ttl: 60  # Cache for 60 seconds
  max_size: 1000  # Max items in cache
```

## 📚 Advanced Features

### 1. Custom Indicators

Add custom indicators in `trading_engine.py`:

```python
def calculate_custom_indicator(self, ticker: str) -> float:
    """Calculate your custom indicator"""
    buffer = self.data_buffer.get(ticker, {})
    prices = list(buffer.get('prices', []))
    
    if len(prices) < 10:
        return 0
    
    # Your custom calculation
    custom_value = sum(prices[-10:]) / 10 * 1.5
    
    return custom_value
```

### 2. Machine Learning Integration

```python
# Add ML predictions to rules
import joblib

class MLPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    def predict_signal(self, features):
        prediction = self.model.predict([features])[0]
        return prediction > 0.5  # Binary signal
```

### 3. Multi-Timeframe Analysis

```yaml
# Configure multiple timeframes
timeframes:
  - 1min   # Scalping
  - 5min   # Day trading
  - 15min  # Swing entries
  - 1hour  # Trend confirmation
```

### 4. News Integration

```python
# Add news sentiment analysis
import requests

def get_news_sentiment(ticker):
    # Fetch news from API
    news = fetch_news(ticker)
    
    # Analyze sentiment
    sentiment = analyze_sentiment(news)
    
    return sentiment  # -1 to 1 scale
```

## 🎯 Best Practices

### 1. Risk Management
- Never risk more than 1-2% per trade
- Use stop losses on every position
- Diversify across multiple strategies
- Keep position sizes consistent

### 2. Testing Protocol
1. Backtest strategy for at least 6 months
2. Paper trade for 2-4 weeks
3. Start with minimum position sizes
4. Scale up gradually with success

### 3. Monitoring
- Review alerts daily
- Track win rate weekly
- Adjust rules monthly
- Full strategy review quarterly

### 4. Documentation
- Document all rule changes
- Keep trade journal
- Screenshot important setups
- Note market conditions

## 📞 Support Resources

### Official Documentation
- **Alpaca**: https://alpaca.markets/docs/
- **Polygon**: https://polygon.io/docs/
- **IEX Cloud**: https://iexcloud.io/docs/

### Community Resources
- Reddit: r/algotrading
- Discord: Algorithmic Trading servers
- GitHub: Search for similar projects

### Educational Resources
- **Technical Analysis**: TradingView Education
- **Python Finance**: QuantConnect Learning
- **Risk Management**: Investopedia

## ⚠️ Legal Disclaimer

**IMPORTANT**: This software is provided for educational purposes only. 

- Trading involves substantial risk of loss
- Past performance doesn't guarantee future results  
- Not financial advice - consult professionals
- Test thoroughly before using real money
- You are responsible for your trading decisions

## 🔄 Updates and Maintenance

### Keeping the System Updated

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Backup before updates
cp trading_config.yaml trading_config.yaml.backup
cp trading_alerts.db trading_alerts.db.backup

# Test after updates
python test_suite.py
```

### Version Control

```bash
# Initialize git repository
git init

# Track changes
git add .
git commit -m "Initial setup"

# Create branches for experiments
git checkout -b new-strategy
```

---

**Version**: 1.0.0  
**Last Updated**: January 2025  
**License**: MIT

For questions and support, please refer to the documentation or community resources listed above.
