"""
Trading Engine - Moteur principal de détection de signaux
"""

import asyncio
import json
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import deque
import aiohttp
import websocket
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sqlite3
import threading
from queue import Queue

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== STRUCTURES DE DONNÉES ====================

@dataclass
class MarketData:
    """Structure pour les données de marché"""
    ticker: str
    price: float
    volume: int
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    vwap: float = 0.0
    
@dataclass
class StockInfo:
    """Informations fondamentales sur une action"""
    ticker: str
    float_shares: float
    short_interest: float
    market_cap: float
    sector: str = ""
    avg_volume_10d: int = 0
    pre_market_high: float = 0.0
    pre_market_low: float = 0.0
    
@dataclass
class Alert:
    """Structure pour les alertes générées"""
    ticker: str
    timestamp: datetime
    setup_type: str
    entry_price: float
    stop_loss: float
    target: float
    confidence_score: float
    reason: str
    current_price: float
    volume: int

# ==================== CONNECTEURS API ====================

class DataConnector:
    """Classe de base pour les connecteurs de données"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        
    async def connect(self):
        """Établir la connexion"""
        self.session = aiohttp.ClientSession()
        
    async def disconnect(self):
        """Fermer la connexion"""
        if self.session:
            await self.session.close()
            
    async def get_quote(self, ticker: str) -> Optional[MarketData]:
        """Récupérer une cotation"""
        raise NotImplementedError
        
    async def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """Récupérer les infos fondamentales"""
        raise NotImplementedError

class AlpacaConnector(DataConnector):
    """Connecteur pour l'API Alpaca"""
    
    BASE_URL = "https://data.alpaca.markets/v2"
    WS_URL = "wss://stream.data.alpaca.markets/v2"
    
    def __init__(self, api_key: str, api_secret: str):
        super().__init__(api_key)
        self.api_secret = api_secret
        self.ws = None
        
    async def connect(self):
        await super().connect()
        # Headers pour l'authentification
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
    async def get_quote(self, ticker: str) -> Optional[MarketData]:
        """Récupérer la dernière cotation"""
        try:
            url = f"{self.BASE_URL}/stocks/{ticker}/quotes/latest"
            async with self.session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    quote = data.get('quote', {})
                    return MarketData(
                        ticker=ticker,
                        price=(quote.get('bp', 0) + quote.get('ap', 0)) / 2,
                        volume=quote.get('as', 0),
                        timestamp=datetime.fromisoformat(quote.get('t', '')),
                        bid=quote.get('bp', 0),
                        ask=quote.get('ap', 0)
                    )
        except Exception as e:
            logger.error(f"Erreur récupération quote {ticker}: {e}")
        return None
        
    async def get_bars(self, ticker: str, timeframe: str = "1Min", limit: int = 100):
        """Récupérer les barres historiques"""
        try:
            url = f"{self.BASE_URL}/stocks/{ticker}/bars"
            params = {
                "timeframe": timeframe,
                "limit": limit
            }
            async with self.session.get(url, headers=self.headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return pd.DataFrame(data.get('bars', []))
        except Exception as e:
            logger.error(f"Erreur récupération bars {ticker}: {e}")
        return pd.DataFrame()

class PolygonConnector(DataConnector):
    """Connecteur pour Polygon.io"""
    
    BASE_URL = "https://api.polygon.io"
    
    async def get_quote(self, ticker: str) -> Optional[MarketData]:
        """Récupérer la dernière cotation"""
        try:
            url = f"{self.BASE_URL}/v2/last/trade/{ticker}"
            params = {"apiKey": self.api_key}
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get('results', {})
                    return MarketData(
                        ticker=ticker,
                        price=result.get('p', 0),
                        volume=result.get('s', 0),
                        timestamp=datetime.fromtimestamp(result.get('t', 0) / 1000)
                    )
        except Exception as e:
            logger.error(f"Erreur Polygon quote {ticker}: {e}")
        return None

# ==================== MOTEUR DE RÈGLES ====================

class RuleEngine:
    """Moteur d'évaluation des règles de trading"""
    
    def __init__(self, config_file: str):
        self.rules = []
        self.load_rules(config_file)
        self.data_buffer = {}  # Buffer pour les données historiques
        self.buffer_size = 100  # Nombre de points à conserver
        
    def load_rules(self, config_file: str):
        """Charger les règles depuis le fichier de configuration"""
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            self.rules = config.get('rules', [])
            logger.info(f"Chargé {len(self.rules)} règles")
        except Exception as e:
            logger.error(f"Erreur chargement règles: {e}")
            
    def update_buffer(self, ticker: str, data: MarketData):
        """Mettre à jour le buffer de données"""
        if ticker not in self.data_buffer:
            self.data_buffer[ticker] = {
                'prices': deque(maxlen=self.buffer_size),
                'volumes': deque(maxlen=self.buffer_size),
                'timestamps': deque(maxlen=self.buffer_size)
            }
        
        buffer = self.data_buffer[ticker]
        buffer['prices'].append(data.price)
        buffer['volumes'].append(data.volume)
        buffer['timestamps'].append(data.timestamp)
        
    def calculate_indicators(self, ticker: str) -> Dict[str, float]:
        """Calculer les indicateurs techniques"""
        indicators = {}
        
        if ticker not in self.data_buffer:
            return indicators
            
        buffer = self.data_buffer[ticker]
        prices = list(buffer['prices'])
        volumes = list(buffer['volumes'])
        
        if len(prices) >= 5:
            # Moyennes mobiles
            indicators['sma_5'] = np.mean(prices[-5:])
            indicators['volume_avg_5'] = np.mean(volumes[-5:])
            
        if len(prices) >= 20:
            indicators['sma_20'] = np.mean(prices[-20:])
            indicators['volume_avg_20'] = np.mean(volumes[-20:])
            
        if len(prices) >= 2:
            # Variation de prix
            indicators['price_change_pct'] = (prices[-1] - prices[-2]) / prices[-2] * 100
            indicators['volume_ratio'] = volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1
            
        # Prix actuel et volume
        indicators['current_price'] = prices[-1] if prices else 0
        indicators['current_volume'] = volumes[-1] if volumes else 0
        
        return indicators
        
    def evaluate_condition(self, condition: Dict, context: Dict) -> bool:
        """Évaluer une condition"""
        try:
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            # Récupérer la valeur du champ
            if '.' in field:
                # Navigation dans les objets imbriqués
                parts = field.split('.')
                current = context
                for part in parts:
                    current = current.get(part, 0)
                field_value = current
            else:
                field_value = context.get(field, 0)
                
            # Évaluer selon l'opérateur
            if operator == '>':
                return field_value > value
            elif operator == '<':
                return field_value < value
            elif operator == '>=':
                return field_value >= value
            elif operator == '<=':
                return field_value <= value
            elif operator == '==':
                return field_value == value
            elif operator == '!=':
                return field_value != value
            elif operator == 'in':
                return field_value in value
            elif operator == 'contains':
                return value in str(field_value)
                
        except Exception as e:
            logger.error(f"Erreur évaluation condition: {e}")
            
        return False
        
    def evaluate_rules(self, ticker: str, market_data: MarketData, 
                      stock_info: Optional[StockInfo] = None) -> List[Alert]:
        """Évaluer toutes les règles pour un ticker"""
        alerts = []
        
        # Mettre à jour le buffer
        self.update_buffer(ticker, market_data)
        
        # Calculer les indicateurs
        indicators = self.calculate_indicators(ticker)
        
        # Créer le contexte d'évaluation
        context = {
            'ticker': ticker,
            'market': asdict(market_data),
            'indicators': indicators
        }
        
        if stock_info:
            context['info'] = asdict(stock_info)
            
        # Évaluer chaque règle
        for rule in self.rules:
            if not rule.get('enabled', True):
                continue
                
            # Vérifier toutes les conditions
            all_conditions_met = True
            for condition in rule.get('conditions', []):
                if not self.evaluate_condition(condition, context):
                    all_conditions_met = False
                    break
                    
            if all_conditions_met:
                # Générer l'alerte
                action = rule.get('action', {})
                alert = Alert(
                    ticker=ticker,
                    timestamp=datetime.now(),
                    setup_type=rule.get('name', 'Unknown'),
                    entry_price=market_data.price,
                    stop_loss=market_data.price * (1 - action.get('stop_loss_pct', 0.02)),
                    target=market_data.price * (1 + action.get('target_pct', 0.05)),
                    confidence_score=action.get('confidence', 0.5),
                    reason=rule.get('description', ''),
                    current_price=market_data.price,
                    volume=market_data.volume
                )
                alerts.append(alert)
                logger.info(f"Alert générée: {ticker} - {rule.get('name')}")
                
        return alerts

# ==================== GESTIONNAIRE PRINCIPAL ====================

class TradingEngine:
    """Moteur de trading principal"""
    
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.rule_engine = RuleEngine(config_file)
        self.connector = None
        self.alerts_queue = Queue()
        self.market_data_cache = {}
        self.stock_info_cache = {}
        self.db_conn = None
        self.setup_database()
        
    def load_config(self, config_file: str) -> Dict:
        """Charger la configuration"""
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                return yaml.safe_load(f)
            return json.load(f)
            
    def setup_database(self):
        """Initialiser la base de données"""
        self.db_conn = sqlite3.connect('trading_alerts.db', check_same_thread=False)
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                timestamp DATETIME,
                setup_type TEXT,
                entry_price REAL,
                stop_loss REAL,
                target REAL,
                confidence_score REAL,
                reason TEXT,
                current_price REAL,
                volume INTEGER
            )
        ''')
        self.db_conn.commit()
        
    async def initialize_connector(self):
        """Initialiser le connecteur de données"""
        api_config = self.config.get('api', {})
        provider = api_config.get('provider', 'alpaca')
        
        if provider == 'alpaca':
            self.connector = AlpacaConnector(
                api_config.get('key'),
                api_config.get('secret')
            )
        elif provider == 'polygon':
            self.connector = PolygonConnector(api_config.get('key'))
        else:
            raise ValueError(f"Provider non supporté: {provider}")
            
        await self.connector.connect()
        logger.info(f"Connecté à {provider}")
        
    async def fetch_market_data(self, ticker: str) -> Optional[MarketData]:
        """Récupérer les données de marché"""
        if self.connector:
            return await self.connector.get_quote(ticker)
        return None
        
    async def fetch_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """Récupérer les infos fondamentales (à implémenter selon l'API)"""
        # Pour l'instant, retourner des données de test
        return StockInfo(
            ticker=ticker,
            float_shares=50_000_000,
            short_interest=0.15,
            market_cap=1_000_000_000,
            avg_volume_10d=5_000_000
        )
        
    def save_alert(self, alert: Alert):
        """Sauvegarder une alerte en base"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (ticker, timestamp, setup_type, entry_price,
                              stop_loss, target, confidence_score, reason,
                              current_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.ticker, alert.timestamp, alert.setup_type, alert.entry_price,
            alert.stop_loss, alert.target, alert.confidence_score, alert.reason,
            alert.current_price, alert.volume
        ))
        self.db_conn.commit()
        
    async def scan_ticker(self, ticker: str):
        """Scanner un ticker"""
        try:
            # Récupérer les données
            market_data = await self.fetch_market_data(ticker)
            if not market_data:
                return
                
            # Mettre en cache
            self.market_data_cache[ticker] = market_data
            
            # Récupérer les infos fondamentales (avec cache)
            if ticker not in self.stock_info_cache:
                stock_info = await self.fetch_stock_info(ticker)
                if stock_info:
                    self.stock_info_cache[ticker] = stock_info
            else:
                stock_info = self.stock_info_cache[ticker]
                
            # Évaluer les règles
            alerts = self.rule_engine.evaluate_rules(ticker, market_data, stock_info)
            
            # Traiter les alertes
            for alert in alerts:
                self.save_alert(alert)
                self.alerts_queue.put(alert)
                
        except Exception as e:
            logger.error(f"Erreur scan {ticker}: {e}")
            
    async def run_scanner(self):
        """Boucle principale de scan"""
        watchlist = self.config.get('watchlist', [])
        scan_interval = self.config.get('scan_interval', 5)
        
        logger.info(f"Démarrage du scanner sur {len(watchlist)} tickers")
        
        while True:
            try:
                # Scanner tous les tickers
                tasks = [self.scan_ticker(ticker) for ticker in watchlist]
                await asyncio.gather(*tasks)
                
                # Attendre avant le prochain scan
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de scan: {e}")
                await asyncio.sleep(scan_interval)

# ==================== API WEB ====================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

engine = None

@app.route('/api/alerts')
def get_alerts():
    """Récupérer l'historique des alertes"""
    conn = sqlite3.connect('trading_alerts.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM alerts 
        ORDER BY timestamp DESC 
        LIMIT 100
    ''')
    alerts = cursor.fetchall()
    conn.close()
    
    return jsonify([{
        'id': a[0],
        'ticker': a[1],
        'timestamp': a[2],
        'setup_type': a[3],
        'entry_price': a[4],
        'stop_loss': a[5],
        'target': a[6],
        'confidence_score': a[7],
        'reason': a[8],
        'current_price': a[9],
        'volume': a[10]
    } for a in alerts])

@app.route('/api/market/<ticker>')
def get_market_data(ticker):
    """Récupérer les données de marché en cache"""
    if engine and ticker in engine.market_data_cache:
        data = engine.market_data_cache[ticker]
        return jsonify({
            'ticker': data.ticker,
            'price': data.price,
            'volume': data.volume,
            'timestamp': data.timestamp.isoformat()
        })
    return jsonify({'error': 'Ticker not found'}), 404

@socketio.on('connect')
def handle_connect():
    """Gestion de la connexion WebSocket"""
    logger.info("Client connecté")
    emit('connected', {'data': 'Connected to Trading Engine'})

def broadcast_alerts():
    """Diffuser les alertes via WebSocket"""
    while True:
        if engine and not engine.alerts_queue.empty():
            alert = engine.alerts_queue.get()
            socketio.emit('new_alert', {
                'ticker': alert.ticker,
                'timestamp': alert.timestamp.isoformat(),
                'setup_type': alert.setup_type,
                'entry_price': alert.entry_price,
                'stop_loss': alert.stop_loss,
                'target': alert.target,
                'confidence_score': alert.confidence_score,
                'reason': alert.reason
            })
        else:
            asyncio.sleep(1)

# ==================== MAIN ====================

async def main():
    """Point d'entrée principal"""
    global engine
    
    # Charger la configuration
    config_file = 'trading_config.yaml'
    engine = TradingEngine(config_file)
    
    # Initialiser le connecteur
    await engine.initialize_connector()
    
    # Démarrer le serveur web dans un thread séparé
    web_thread = threading.Thread(
        target=lambda: socketio.run(app, host='0.0.0.0', port=5000)
    )
    web_thread.daemon = True
    web_thread.start()
    
    # Démarrer le broadcast des alertes
    broadcast_thread = threading.Thread(target=broadcast_alerts)
    broadcast_thread.daemon = True
    broadcast_thread.start()
    
    # Démarrer le scanner
    await engine.run_scanner()

if __name__ == "__main__":
    asyncio.run(main())