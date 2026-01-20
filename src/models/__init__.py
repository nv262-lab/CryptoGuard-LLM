"""
CryptoGuard-LLM Models

Multi-modal deep learning models for cryptocurrency fraud detection.
"""

from .cryptoguard import CryptoGuardLLM, CryptoGuardConfig, FraudTypeClassifier
from .gnn import (
    HeterogeneousGraphAttentionNetwork,
    GraphAttentionLayer,
    TransactionGraphEncoder
)
from .bert_classifier import (
    BERTFraudClassifier,
    ThreatIntelligenceProcessor,
    SocialMediaAnalyzer
)
from .ensemble import (
    EnsembleClassifier,
    WeightedEnsemble,
    StackingEnsemble,
    ConfidenceCalibrator,
    MultiTaskHead
)

__all__ = [
    # Main model
    'CryptoGuardLLM',
    'CryptoGuardConfig',
    'FraudTypeClassifier',
    
    # GNN components
    'HeterogeneousGraphAttentionNetwork',
    'GraphAttentionLayer',
    'TransactionGraphEncoder',
    
    # NLP components
    'BERTFraudClassifier',
    'ThreatIntelligenceProcessor',
    'SocialMediaAnalyzer',
    
    # Ensemble components
    'EnsembleClassifier',
    'WeightedEnsemble',
    'StackingEnsemble',
    'ConfidenceCalibrator',
    'MultiTaskHead'
]

__version__ = '1.0.0'
