"""
Data Preprocessing for CryptoGuard-LLM

Handles feature extraction and data preprocessing for
cryptocurrency transaction data.

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses raw cryptocurrency transaction data.
    
    Handles:
    - Missing value imputation
    - Feature scaling
    - Categorical encoding
    - Temporal feature extraction
    """
    
    def __init__(
        self,
        numerical_scaler: str = 'standard',
        handle_missing: str = 'median'
    ):
        self.numerical_scaler = numerical_scaler
        self.handle_missing = handle_missing
        
        # Initialize scalers and encoders
        if numerical_scaler == 'standard':
            self.scaler = StandardScaler()
        elif numerical_scaler == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            
        self.imputer = SimpleImputer(strategy=handle_missing)
        self.label_encoders = {}
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str]):
        """Fit preprocessors on training data."""
        # Fit numerical scaler
        if self.scaler and numerical_cols:
            numerical_data = df[numerical_cols].values
            numerical_data = self.imputer.fit_transform(numerical_data)
            self.scaler.fit(numerical_data)
            
        # Fit label encoders for categorical columns
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(df[col].astype(str))
            
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.fitted = True
        
        logger.info(f"Fitted preprocessor on {len(df)} samples")
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessors."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
            
        df_transformed = df.copy()
        
        # Transform numerical columns
        if self.scaler and self.numerical_cols:
            numerical_data = df[self.numerical_cols].values
            numerical_data = self.imputer.transform(numerical_data)
            numerical_data = self.scaler.transform(numerical_data)
            df_transformed[self.numerical_cols] = numerical_data
            
        # Transform categorical columns
        for col in self.categorical_cols:
            df_transformed[col] = self.label_encoders[col].transform(
                df[col].astype(str)
            )
            
        return df_transformed
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str]
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, numerical_cols, categorical_cols)
        return self.transform(df)


class FeatureExtractor:
    """
    Extracts features from cryptocurrency transaction data.
    
    Features include:
    - Transaction features (value, gas, fees)
    - Wallet features (age, activity, balance)
    - Temporal features (time of day, day of week)
    - Network features (degree, clustering coefficient)
    """
    
    # Standard feature sets
    TRANSACTION_FEATURES = [
        'value', 'gas_price', 'gas_used', 'transaction_fee',
        'input_length', 'is_contract_creation', 'is_contract_call'
    ]
    
    WALLET_FEATURES = [
        'wallet_age_days', 'total_transactions', 'total_received',
        'total_sent', 'unique_counterparties', 'avg_transaction_value',
        'transaction_frequency', 'balance'
    ]
    
    TEMPORAL_FEATURES = [
        'hour_of_day', 'day_of_week', 'is_weekend',
        'time_since_last_tx', 'tx_count_last_hour', 'tx_count_last_day'
    ]
    
    def __init__(self):
        self.feature_stats = {}
        
    def extract_transaction_features(self, tx: Dict) -> Dict[str, float]:
        """Extract features from a single transaction."""
        features = {}
        
        # Basic transaction features
        features['value'] = float(tx.get('value', 0))
        features['gas_price'] = float(tx.get('gasPrice', 0))
        features['gas_used'] = float(tx.get('gasUsed', 0))
        features['transaction_fee'] = features['gas_price'] * features['gas_used']
        
        # Input data features
        input_data = tx.get('input', '0x')
        features['input_length'] = len(input_data) // 2 - 1  # Hex bytes
        features['is_contract_creation'] = 1 if tx.get('to') is None else 0
        features['is_contract_call'] = 1 if len(input_data) > 10 else 0
        
        return features
    
    def extract_wallet_features(
        self,
        address: str,
        transactions: List[Dict],
        current_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Extract features for a wallet address."""
        if current_time is None:
            current_time = datetime.now()
            
        features = {}
        
        if not transactions:
            return {f: 0.0 for f in self.WALLET_FEATURES}
            
        # Filter transactions for this address
        wallet_txs = [
            tx for tx in transactions
            if tx.get('from') == address or tx.get('to') == address
        ]
        
        if not wallet_txs:
            return {f: 0.0 for f in self.WALLET_FEATURES}
            
        # Calculate wallet age
        timestamps = [tx.get('timestamp', 0) for tx in wallet_txs]
        first_tx_time = datetime.fromtimestamp(min(timestamps))
        features['wallet_age_days'] = (current_time - first_tx_time).days
        
        # Transaction counts
        features['total_transactions'] = len(wallet_txs)
        
        # Value calculations
        received = sum(
            float(tx.get('value', 0))
            for tx in wallet_txs if tx.get('to') == address
        )
        sent = sum(
            float(tx.get('value', 0))
            for tx in wallet_txs if tx.get('from') == address
        )
        
        features['total_received'] = received
        features['total_sent'] = sent
        features['balance'] = received - sent
        
        # Counterparties
        counterparties = set()
        for tx in wallet_txs:
            if tx.get('from') == address:
                counterparties.add(tx.get('to'))
            else:
                counterparties.add(tx.get('from'))
        features['unique_counterparties'] = len(counterparties)
        
        # Averages
        values = [float(tx.get('value', 0)) for tx in wallet_txs]
        features['avg_transaction_value'] = np.mean(values) if values else 0
        
        # Frequency
        if features['wallet_age_days'] > 0:
            features['transaction_frequency'] = (
                features['total_transactions'] / features['wallet_age_days']
            )
        else:
            features['transaction_frequency'] = features['total_transactions']
            
        return features
    
    def extract_temporal_features(
        self,
        timestamp: Union[int, datetime],
        transaction_history: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """Extract temporal features from timestamp."""
        if isinstance(timestamp, int):
            dt = datetime.fromtimestamp(timestamp)
        else:
            dt = timestamp
            
        features = {}
        
        # Time-based features
        features['hour_of_day'] = dt.hour
        features['day_of_week'] = dt.weekday()
        features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
        
        # Historical features (if history provided)
        if transaction_history:
            # Time since last transaction
            past_timestamps = [
                tx.get('timestamp', 0)
                for tx in transaction_history
                if tx.get('timestamp', 0) < timestamp
            ]
            if past_timestamps:
                last_tx_time = max(past_timestamps)
                features['time_since_last_tx'] = timestamp - last_tx_time
            else:
                features['time_since_last_tx'] = 0
                
            # Transaction counts in windows
            hour_ago = timestamp - 3600
            day_ago = timestamp - 86400
            
            features['tx_count_last_hour'] = sum(
                1 for tx in transaction_history
                if hour_ago <= tx.get('timestamp', 0) < timestamp
            )
            features['tx_count_last_day'] = sum(
                1 for tx in transaction_history
                if day_ago <= tx.get('timestamp', 0) < timestamp
            )
        else:
            features['time_since_last_tx'] = 0
            features['tx_count_last_hour'] = 0
            features['tx_count_last_day'] = 0
            
        return features
    
    def extract_all_features(
        self,
        transaction: Dict,
        wallet_history: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """Extract all features for a transaction."""
        features = {}
        
        # Transaction features
        features.update(self.extract_transaction_features(transaction))
        
        # Wallet features (for sender)
        if wallet_history:
            sender = transaction.get('from')
            if sender:
                wallet_features = self.extract_wallet_features(
                    sender, wallet_history
                )
                features.update({f"sender_{k}": v for k, v in wallet_features.items()})
                
        # Temporal features
        timestamp = transaction.get('timestamp', 0)
        temporal_features = self.extract_temporal_features(
            timestamp, wallet_history
        )
        features.update(temporal_features)
        
        return features


class TextPreprocessor:
    """
    Preprocesses text data for NLP components.
    
    Handles:
    - Text cleaning
    - Tokenization
    - Special token handling for crypto-specific terms
    """
    
    # Crypto-specific terms to preserve
    CRYPTO_TERMS = [
        'bitcoin', 'btc', 'ethereum', 'eth', 'defi', 'nft', 'dao',
        'smart contract', 'wallet', 'mining', 'staking', 'yield',
        'airdrop', 'rug pull', 'pump', 'dump', 'hodl', 'fomo', 'fud'
    ]
    
    def __init__(self, lowercase: bool = True, remove_urls: bool = True):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        import re
        
        if not text or pd.isna(text):
            return ""
            
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
            
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep crypto addresses
        text = re.sub(r'[^\w\s0x]', ' ', text)
        
        return text
    
    def extract_addresses(self, text: str) -> List[str]:
        """Extract cryptocurrency addresses from text."""
        import re
        
        # Ethereum addresses (0x followed by 40 hex chars)
        eth_pattern = r'0x[a-fA-F0-9]{40}'
        eth_addresses = re.findall(eth_pattern, text)
        
        # Bitcoin addresses (various formats)
        btc_pattern = r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,59}'
        btc_addresses = re.findall(btc_pattern, text)
        
        return eth_addresses + btc_addresses
    
    def anonymize_addresses(self, text: str) -> str:
        """Replace addresses with anonymous tokens."""
        addresses = self.extract_addresses(text)
        
        for i, addr in enumerate(set(addresses)):
            text = text.replace(addr, f'[ADDRESS_{i}]')
            
        return text


if __name__ == '__main__':
    # Example usage
    print("DataPreprocessor - Feature extraction for cryptocurrency fraud detection")
    
    # Demo feature extraction
    extractor = FeatureExtractor()
    
    sample_tx = {
        'from': '0x1234567890abcdef1234567890abcdef12345678',
        'to': '0xabcdef1234567890abcdef1234567890abcdef12',
        'value': 1000000000000000000,  # 1 ETH in wei
        'gasPrice': 20000000000,
        'gasUsed': 21000,
        'input': '0x',
        'timestamp': 1704067200
    }
    
    features = extractor.extract_transaction_features(sample_tx)
    print(f"Extracted {len(features)} transaction features")
    for k, v in features.items():
        print(f"  {k}: {v}")
