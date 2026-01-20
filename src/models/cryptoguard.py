"""
CryptoGuard-LLM: Main Model Implementation

A Multi-Modal Deep Learning Framework for Real-Time Cryptocurrency Fraud Detection

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .gnn import HeterogeneousGraphAttentionNetwork
from .bert_classifier import BERTFraudClassifier
from .ensemble import EnsembleClassifier


@dataclass
class CryptoGuardConfig:
    """Configuration for CryptoGuard-LLM model."""
    
    # GNN Configuration
    gnn_num_layers: int = 4
    gnn_hidden_dim: int = 256
    gnn_attention_heads: int = 8
    gnn_dropout: float = 0.3
    gnn_negative_slope: float = 0.2
    
    # BERT Configuration
    bert_model_name: str = 'bert-base-uncased'
    bert_max_length: int = 512
    bert_hidden_dim: int = 768
    
    # Ensemble Configuration
    ensemble_mlp_dims: List[int] = None
    num_classes: int = 2
    
    # Training Configuration
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    
    def __post_init__(self):
        if self.ensemble_mlp_dims is None:
            self.ensemble_mlp_dims = [384, 128, self.num_classes]


class CryptoGuardLLM(nn.Module):
    """
    CryptoGuard-LLM: Multi-Modal Deep Learning Framework for Cryptocurrency Fraud Detection
    
    This model integrates:
    1. Heterogeneous Graph Attention Network (HGAT) for transaction graph analysis
    2. Fine-tuned BERT for textual threat intelligence processing
    3. Ensemble classifier for final prediction
    
    Args:
        config (CryptoGuardConfig): Model configuration
    """
    
    def __init__(self, config: Optional[CryptoGuardConfig] = None):
        super().__init__()
        
        self.config = config or CryptoGuardConfig()
        
        # Initialize GNN component
        self.gnn = HeterogeneousGraphAttentionNetwork(
            num_layers=self.config.gnn_num_layers,
            hidden_dim=self.config.gnn_hidden_dim,
            attention_heads=self.config.gnn_attention_heads,
            dropout=self.config.gnn_dropout,
            negative_slope=self.config.gnn_negative_slope
        )
        
        # Initialize BERT classifier
        self.bert_classifier = BERTFraudClassifier(
            model_name=self.config.bert_model_name,
            max_length=self.config.bert_max_length,
            hidden_dim=self.config.bert_hidden_dim
        )
        
        # Initialize ensemble classifier
        # Input: GNN output (hidden_dim) + BERT output (hidden_dim)
        ensemble_input_dim = self.config.gnn_hidden_dim + self.config.bert_hidden_dim
        self.ensemble = EnsembleClassifier(
            input_dim=ensemble_input_dim,
            mlp_dims=self.config.ensemble_mlp_dims,
            num_classes=self.config.num_classes
        )
        
        # Anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(ensemble_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        graph_data: Dict[str, torch.Tensor],
        text_data: Dict[str, torch.Tensor],
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CryptoGuard-LLM model.
        
        Args:
            graph_data: Dictionary containing:
                - node_features: Node feature tensor
                - edge_index: Edge connectivity
                - edge_features: Edge feature tensor
                - batch: Batch assignment for nodes
            text_data: Dictionary containing:
                - input_ids: Tokenized text input
                - attention_mask: Attention mask for BERT
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing:
                - logits: Classification logits
                - probabilities: Softmax probabilities
                - fraud_type: Predicted fraud type
                - anomaly_score: Anomaly detection score
                - embeddings (optional): Intermediate embeddings
        """
        
        # Process graph data through GNN
        gnn_output = self.gnn(
            x=graph_data['node_features'],
            edge_index=graph_data['edge_index'],
            edge_attr=graph_data.get('edge_features'),
            batch=graph_data.get('batch')
        )
        
        # Process text data through BERT
        bert_output = self.bert_classifier.get_embeddings(
            input_ids=text_data['input_ids'],
            attention_mask=text_data['attention_mask']
        )
        
        # Concatenate representations
        combined_features = torch.cat([gnn_output, bert_output], dim=-1)
        
        # Ensemble classification
        logits = self.ensemble(combined_features)
        probabilities = F.softmax(logits, dim=-1)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector(combined_features)
        
        # Prepare output
        output = {
            'logits': logits,
            'probabilities': probabilities,
            'fraud_probability': probabilities[:, 1],
            'anomaly_score': anomaly_score.squeeze(-1),
            'predictions': torch.argmax(logits, dim=-1)
        }
        
        if return_embeddings:
            output['gnn_embeddings'] = gnn_output
            output['bert_embeddings'] = bert_output
            output['combined_embeddings'] = combined_features
            
        return output
    
    def predict(
        self,
        transactions: List[Dict],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Predict fraud for a list of transactions.
        
        Args:
            transactions: List of transaction dictionaries
            threshold: Classification threshold
            
        Returns:
            List of prediction dictionaries
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for tx in transactions:
                # Preprocess transaction
                graph_data, text_data = self._preprocess_transaction(tx)
                
                # Forward pass
                output = self.forward(graph_data, text_data)
                
                # Generate prediction
                fraud_prob = output['fraud_probability'].item()
                is_fraud = fraud_prob >= threshold
                
                pred = {
                    'transaction_hash': tx.get('hash', 'unknown'),
                    'is_fraud': is_fraud,
                    'probability': fraud_prob,
                    'anomaly_score': output['anomaly_score'].item(),
                    'fraud_type': self._get_fraud_type(output) if is_fraud else None,
                    'confidence': abs(fraud_prob - 0.5) * 2,  # 0-1 scale
                    'explanation': self._generate_explanation(tx, output)
                }
                predictions.append(pred)
                
        return predictions
    
    def _preprocess_transaction(
        self,
        transaction: Dict
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Preprocess a single transaction for model input."""
        # This is a placeholder - actual implementation would involve
        # graph construction and text tokenization
        raise NotImplementedError("Implement transaction preprocessing")
    
    def _get_fraud_type(self, output: Dict[str, torch.Tensor]) -> str:
        """Determine the type of fraud based on model output."""
        fraud_types = ['rug_pull', 'phishing', 'ponzi_scheme', 'exchange_hack', 'ransomware']
        # This is a simplified version - actual implementation would use
        # multi-label classification or additional classification heads
        return fraud_types[0]
    
    def _generate_explanation(
        self,
        transaction: Dict,
        output: Dict[str, torch.Tensor]
    ) -> str:
        """Generate human-readable explanation for the prediction."""
        prob = output['fraud_probability'].item()
        anomaly = output['anomaly_score'].item()
        
        if prob < 0.3:
            return "Transaction appears legitimate with normal patterns."
        elif prob < 0.7:
            return f"Transaction shows some suspicious patterns (anomaly score: {anomaly:.2f}). Manual review recommended."
        else:
            return f"High fraud probability detected (anomaly score: {anomaly:.2f}). Immediate investigation recommended."
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained model weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {checkpoint_path}")
        
    def save_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict] = None
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


class FraudTypeClassifier(nn.Module):
    """
    Multi-label classifier for fraud type detection.
    
    Predicts specific fraud categories:
    - Rug Pulls
    - Phishing
    - Ponzi Schemes
    - Exchange Hacks
    - Ransomware
    """
    
    FRAUD_TYPES = ['rug_pull', 'phishing', 'ponzi_scheme', 'exchange_hack', 'ransomware']
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, len(self.FRAUD_TYPES))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for each fraud type."""
        return self.classifier(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> List[List[str]]:
        """Predict fraud types for batch of inputs."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        predictions = []
        
        for prob in probs:
            types = [self.FRAUD_TYPES[i] for i, p in enumerate(prob) if p >= threshold]
            predictions.append(types if types else ['unknown'])
            
        return predictions


if __name__ == '__main__':
    # Example usage
    config = CryptoGuardConfig(
        gnn_num_layers=4,
        gnn_hidden_dim=256,
        gnn_attention_heads=8
    )
    
    model = CryptoGuardLLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
