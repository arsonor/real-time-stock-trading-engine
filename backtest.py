"""
Module de Backtest pour le Trading Engine
Permet de tester les stratégies sur des données historiques
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

@dataclass
class Trade:
    """Structure pour représenter un trade"""
    ticker: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: int
    stop_loss: float
    target: float
    setup_type: str
    status: str  # 'open', 'closed_profit', 'closed_loss', 'closed_target'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
@dataclass
class BacktestResults:
    """Résultats du backtest"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    best_trade: float
    worst_trade: float
    avg_hold_time: timedelta
    trades: List[Trade]
    equity_curve: pd.Series

class Backtester:
    """Classe principale pour le backtesting"""
    
    def __init__(self, config_file: str, initial_capital: float = 10000):
        self.config = self.load_config(config_file)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.open_positions = {}
        self.equity_curve = []
        self.daily_returns = []
        
    def load_config(self, config_file: str) -> Dict:
        """Charger la configuration"""
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                return yaml.safe_load(f)
            return json.load(f)
    
    def load_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Charger les données historiques"""
        # Ici, vous pouvez utiliser yfinance, Alpha Vantage, ou vos propres données
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1h')
        
        # Ajouter des indicateurs techniques
        df = self.add_technical_indicators(df)
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter les indicateurs techniques nécessaires"""
        # SMA
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volume moyen
        df['Volume_Avg_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Avg_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_Avg_20']
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        
        # ATR
        df['ATR'] = self.calculate_atr(df)
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Support/Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        # Gap
        df['Gap_Pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
        
        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculer le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculer le MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculer l'ATR"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculer les Bandes de Bollinger"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def evaluate_rules(self, row: pd.Series, ticker: str) -> Optional[Dict]:
        """Évaluer les règles de trading pour une ligne de données"""
        for rule in self.config.get('rules', []):
            if not rule.get('enabled', True):
                continue
            
            all_conditions_met = True
            
            for condition in rule.get('conditions', []):
                field = condition.get('field')
                operator = condition.get('operator')
                value = condition.get('value')
                
                # Mapper les champs aux colonnes du DataFrame
                field_mapping = {
                    'market.price': 'Close',
                    'indicators.volume_ratio': 'Volume_Ratio',
                    'indicators.rsi': 'RSI',
                    'indicators.sma_5': 'SMA_5',
                    'indicators.sma_20': 'SMA_20',
                    'indicators.gap_pct': 'Gap_Pct',
                    'market.volume': 'Volume'
                }
                
                if field in field_mapping:
                    field_value = row[field_mapping[field]]
                    
                    # Gérer les valeurs NaN
                    if pd.isna(field_value):
                        all_conditions_met = False
                        break
                    
                    # Évaluer la condition
                    if operator == '>' and not (field_value > value):
                        all_conditions_met = False
                        break
                    elif operator == '<' and not (field_value < value):
                        all_conditions_met = False
                        break
                    elif operator == '>=' and not (field_value >= value):
                        all_conditions_met = False
                        break
                    elif operator == '<=' and not (field_value <= value):
                        all_conditions_met = False
                        break
                else:
                    all_conditions_met = False
                    break
            
            if all_conditions_met:
                return rule
        
        return None
    
    def execute_trade(self, ticker: str, timestamp: datetime, price: float, 
                     rule: Dict) -> Optional[Trade]:
        """Exécuter un trade"""
        # Calculer la taille de position
        position_size_pct = self.config.get('backtest', {}).get('position_size_pct', 0.1)
        position_value = self.current_capital * position_size_pct
        shares = int(position_value / price)
        
        if shares <= 0:
            return None
        
        # Créer le trade
        action = rule.get('action', {})
        trade = Trade(
            ticker=ticker,
            entry_time=timestamp,
            exit_time=None,
            entry_price=price,
            exit_price=None,
            position_size=shares,
            stop_loss=price * (1 - action.get('stop_loss_pct', 0.02)),
            target=price * (1 + action.get('target_pct', 0.05)),
            setup_type=rule.get('name', 'Unknown'),
            status='open'
        )
        
        # Déduire le capital
        self.current_capital -= shares * price
        
        return trade
    
    def check_exit_conditions(self, trade: Trade, current_price: float, 
                            timestamp: datetime) -> bool:
        """Vérifier les conditions de sortie"""
        if trade.status != 'open':
            return False
        
        should_exit = False
        exit_reason = ''
        
        # Check stop loss
        if current_price <= trade.stop_loss:
            should_exit = True
            exit_reason = 'closed_loss'
            trade.exit_price = trade.stop_loss
        
        # Check target
        elif current_price >= trade.target:
            should_exit = True
            exit_reason = 'closed_target'
            trade.exit_price = trade.target
        
        # Check trailing stop (si activé)
        elif self.config.get('risk_management', {}).get('trailing_stop_enabled', False):
            trailing_pct = self.config.get('risk_management', {}).get('trailing_stop_pct', 0.015)
            max_price = current_price  # Simplification - devrait tracker le max depuis l'entrée
            trailing_stop = max_price * (1 - trailing_pct)
            if current_price <= trailing_stop:
                should_exit = True
                exit_reason = 'closed_trailing'
                trade.exit_price = current_price
        
        if should_exit:
            trade.exit_time = timestamp
            trade.status = exit_reason
            trade.exit_price = trade.exit_price or current_price
            
            # Calculer P&L
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
            trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
            
            # Récupérer le capital
            self.current_capital += trade.position_size * trade.exit_price
            
            return True
        
        return False
    
    def run_backtest(self, tickers: List[str], start_date: str, end_date: str) -> BacktestResults:
        """Exécuter le backtest"""
        print(f"🚀 Démarrage du backtest du {start_date} au {end_date}")
        print(f"💰 Capital initial: ${self.initial_capital:,.2f}")
        print(f"📊 Tickers: {', '.join(tickers)}")
        print("-" * 50)
        
        all_trades = []
        
        for ticker in tickers:
            print(f"\n📈 Backtest de {ticker}...")
            
            # Charger les données
            df = self.load_historical_data(ticker, start_date, end_date)
            
            if df.empty:
                print(f"⚠️ Pas de données pour {ticker}")
                continue
            
            open_trade = None
            
            # Parcourir chaque ligne
            for idx, row in df.iterrows():
                current_price = row['Close']
                
                # Vérifier les conditions de sortie pour les positions ouvertes
                if open_trade:
                    if self.check_exit_conditions(open_trade, current_price, idx):
                        all_trades.append(open_trade)
                        print(f"  ❌ Sortie: {open_trade.setup_type} à ${open_trade.exit_price:.2f} "
                              f"(P&L: {open_trade.pnl_pct:+.2f}%)")
                        open_trade = None
                
                # Vérifier les conditions d'entrée si pas de position ouverte
                if not open_trade:
                    rule = self.evaluate_rules(row, ticker)
                    if rule:
                        trade = self.execute_trade(ticker, idx, current_price, rule)
                        if trade:
                            open_trade = trade
                            print(f"  ✅ Entrée: {trade.setup_type} à ${trade.entry_price:.2f}")
                
                # Enregistrer l'équité
                position_value = 0
                if open_trade:
                    position_value = open_trade.position_size * current_price
                total_equity = self.current_capital + position_value
                self.equity_curve.append(total_equity)
        
        # Fermer les positions encore ouvertes
        if open_trade:
            open_trade.exit_time = df.index[-1]
            open_trade.exit_price = df.iloc[-1]['Close']
            open_trade.status = 'closed_end'
            open_trade.pnl = (open_trade.exit_price - open_trade.entry_price) * open_trade.position_size
            open_trade.pnl_pct = ((open_trade.exit_price - open_trade.entry_price) / open_trade.entry_price) * 100
            self.current_capital += open_trade.position_size * open_trade.exit_price
            all_trades.append(open_trade)
        
        # Calculer les statistiques
        results = self.calculate_statistics(all_trades)
        
        # Afficher le résumé
        self.print_summary(results)
        
        # Générer les graphiques
        self.plot_results(results)
        
        return results
    
    def calculate_statistics(self, trades: List[Trade]) -> BacktestResults:
        """Calculer les statistiques du backtest"""
        if not trades:
            return BacktestResults(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                total_pnl=0, total_return_pct=0, sharpe_ratio=0,
                max_drawdown=0, best_trade=0, worst_trade=0,
                avg_hold_time=timedelta(0), trades=[], 
                equity_curve=pd.Series(self.equity_curve)
            )
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # Win rate
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        # Average win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Best/Worst trade
        best_trade = max(t.pnl_pct for t in trades) if trades else 0
        worst_trade = min(t.pnl_pct for t in trades) if trades else 0
        
        # Average hold time
        hold_times = [(t.exit_time - t.entry_time) for t in trades if t.exit_time]
        avg_hold_time = np.mean(hold_times) if hold_times else timedelta(0)
        
        # Sharpe ratio (simplifié)
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        return BacktestResults(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_hold_time=avg_hold_time,
            trades=trades,
            equity_curve=equity_series
        )
    
    def print_summary(self, results: BacktestResults):
        """Afficher le résumé des résultats"""
        print("\n" + "=" * 60)
        print("📊 RÉSULTATS DU BACKTEST")
        print("=" * 60)
        
        print(f"\n💼 Performance Globale:")
        print(f"  • Capital final: ${self.current_capital:,.2f}")
        print(f"  • P&L Total: ${results.total_pnl:+,.2f}")
        print(f"  • Retour Total: {results.total_return_pct:+.2f}%")
        print(f"  • Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"  • Max Drawdown: {results.max_drawdown:.2f}%")
        
        print(f"\n📈 Statistiques des Trades:")
        print(f"  • Nombre total: {results.total_trades}")
        print(f"  • Trades gagnants: {results.winning_trades}")
        print(f"  • Trades perdants: {results.losing_trades}")
        print(f"  • Taux de réussite: {results.win_rate:.1f}%")
        print(f"  • Profit Factor: {results.profit_factor:.2f}")
        
        print(f"\n💰 Moyennes:")
        print(f"  • Gain moyen: ${results.avg_win:+,.2f}")
        print(f"  • Perte moyenne: ${results.avg_loss:+,.2f}")
        print(f"  • Durée moyenne: {results.avg_hold_time}")
        
        print(f"\n🎯 Extrêmes:")
        print(f"  • Meilleur trade: {results.best_trade:+.2f}%")
        print(f"  • Pire trade: {results.worst_trade:+.2f}%")
        
        print("\n" + "=" * 60)
    
    def plot_results(self, results: BacktestResults):
        """Générer les graphiques de résultats"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Résultats du Backtest', fontsize=16)
        
        # 1. Courbe d'équité
        ax1 = axes[0, 0]
        results.equity_curve.plot(ax=ax1, color='blue', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5)
        ax1.set_title('Évolution du Capital')
        ax1.set_xlabel('Temps')
        ax1.set_ylabel('Capital ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution des P&L
        ax2 = axes[0, 1]
        pnl_pcts = [t.pnl_pct for t in results.trades]
        if pnl_pcts:
            ax2.hist(pnl_pcts, bins=30, color='skyblue', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax2.set_title('Distribution des P&L (%)')
            ax2.set_xlabel('P&L (%)')
            ax2.set_ylabel('Fréquence')
        
        # 3. P&L cumulé par setup
        ax3 = axes[1, 0]
        setup_pnl = defaultdict(float)
        for trade in results.trades:
            setup_pnl[trade.setup_type] += trade.pnl
        
        if setup_pnl:
            setups = list(setup_pnl.keys())
            pnls = list(setup_pnl.values())
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax3.barh(setups, pnls, color=colors)
            ax3.set_title('P&L par Type de Setup')
            ax3.set_xlabel('P&L ($)')
            ax3.set_ylabel('Setup')
        
        # 4. Drawdown
        ax4 = axes[1, 1]
        rolling_max = results.equity_curve.expanding().max()
        drawdown = (results.equity_curve - rolling_max) / rolling_max * 100
        drawdown.plot(ax=ax4, color='red', linewidth=2, fill=True, alpha=0.3)
        ax4.set_title('Drawdown')
        ax4.set_xlabel('Temps')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=100)
        plt.show()
        print("\n📊 Graphiques sauvegardés dans 'backtest_results.png'")
    
    def export_results(self, results: BacktestResults, filename: str = 'backtest_results.json'):
        """Exporter les résultats en JSON"""
        export_data = {
            'summary': {
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'win_rate': results.win_rate,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss,
                'profit_factor': results.profit_factor,
                'total_pnl': results.total_pnl,
                'total_return_pct': results.total_return_pct,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'best_trade': results.best_trade,
                'worst_trade': results.worst_trade
            },
            'trades': [
                {
                    'ticker': t.ticker,
                    'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'position_size': t.position_size,
                    'setup_type': t.setup_type,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'status': t.status
                }
                for t in results.trades
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n💾 Résultats exportés dans '{filename}'")

# ==================== UTILISATION ====================

if __name__ == "__main__":
    # Configuration
    config_file = 'trading_config.yaml'
    
    # Créer le backtester
    backtester = Backtester(config_file, initial_capital=10000)
    
    # Liste des tickers à tester
    tickers = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'META']
    
    # Période de test
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    # Lancer le backtest
    results = backtester.run_backtest(tickers, start_date, end_date)
    
    # Exporter les résultats
    backtester.export_results(results)