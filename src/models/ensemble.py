"""
Ensemble Classifier for CryptoGuard-LLM

Combines outputs from GNN, BERT, and additional models for final fraud prediction.

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class EnsembleClassifier(nn.Module):
    """
    Ensemble classifier combining multiple model outputs.
    
    Uses a multi-layer perceptron to combine:
    - GNN graph embeddings
    - BERT text embeddings
    - Additional features (optional)
    
    Args:
        input_dim: Total input dimension (sum of all embedding dimensions)
        mlp_dims: List of MLP hidden dimensions
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        mlp_dims: List[int] = None,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if mlp_dims is None:
            mlp_dims = [384, 128, num_classes]
            
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(mlp_dims[:-1]):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        # Final classification layer
        layers.append(nn.Linear(prev_dim, mlp_dims[-1]))
        
        self.mlp = nn.Sequential(*layers)
        
        # Attention mechanism for feature weighting
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        apply_attention: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Combined feature tensor [batch_size, input_dim]
            apply_attention: Whether to apply feature attention
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        
        if apply_attention:
            attention_weights = self.feature_attention(x)
            x = x * attention_weights
            
        return self.mlp(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class WeightedEnsemble(nn.Module):
    """
    Weighted ensemble combining predictions from multiple models.
    
    Learns optimal weights for combining model predictions.
    """
    
    def __init__(
        self,
        num_models: int = 3,
        num_classes: int = 2,
        learn_weights: bool = True
    ):
        super().__init__()
        
        self.num_models = num_models
        self.num_classes = num_classes
        
        if learn_weights:
            self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        else:
            self.register_buffer('weights', torch.ones(num_models) / num_models)
            
    def forward(
        self,
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: List of prediction tensors [batch_size, num_classes]
            
        Returns:
            Combined predictions [batch_size, num_classes]
        """
        
        # Normalize weights
        weights = F.softmax(self.weights, dim=0)
        
        # Stack and weight predictions
        stacked = torch.stack(predictions, dim=-1)  # [batch, classes, models]
        weighted = stacked * weights.view(1, 1, -1)
        
        return weighted.sum(dim=-1)


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble using a meta-learner.
    
    First-level models make predictions, then a meta-learner
    combines them for final prediction.
    """
    
    def __init__(
        self,
        num_base_models: int = 3,
        num_classes: int = 2,
        meta_hidden_dim: int = 64
    ):
        super().__init__()
        
        self.num_base_models = num_base_models
        self.num_classes = num_classes
        
        # Meta-learner
        meta_input_dim = num_base_models * num_classes
        self.meta_learner = nn.Sequential(
            nn.Linear(meta_input_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(meta_hidden_dim, num_classes)
        )
        
    def forward(
        self,
        base_predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Combine base model predictions using meta-learner.
        
        Args:
            base_predictions: List of [batch_size, num_classes] tensors
            
        Returns:
            Final predictions [batch_size, num_classes]
        """
        
        # Concatenate base predictions
        concat_preds = torch.cat(base_predictions, dim=-1)
        
        # Meta-learner prediction
        return self.meta_learner(concat_preds)


class ConfidenceCalibrator(nn.Module):
    """
    Calibrates model confidence scores.
    
    Uses temperature scaling and learned calibration to
    produce well-calibrated probability estimates.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        init_temperature: float = 1.0
    ):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        
        # Learned calibration
        self.calibration_net = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(
        self,
        logits: torch.Tensor,
        use_temperature: bool = True,
        use_calibration_net: bool = False
    ) -> torch.Tensor:
        """
        Calibrate logits.
        
        Args:
            logits: Raw model logits [batch_size, num_classes]
            use_temperature: Apply temperature scaling
            use_calibration_net: Apply learned calibration
            
        Returns:
            Calibrated probabilities [batch_size, num_classes]
        """
        
        if use_temperature:
            logits = logits / self.temperature
            
        if use_calibration_net:
            logits = logits + self.calibration_net(logits)
            
        return F.softmax(logits, dim=-1)
    
    def get_confidence(
        self,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get calibrated confidence scores.
        
        Returns:
            predictions: Class predictions
            confidences: Confidence scores (0-1)
        """
        
        probs = self.forward(logits)
        confidences, predictions = probs.max(dim=-1)
        
        return predictions, confidences


class MultiTaskHead(nn.Module):
    """
    Multi-task classification head for fraud detection.
    
    Simultaneously predicts:
    - Binary fraud/legitimate
    - Fraud type (multi-class)
    - Severity score (regression)
    """
    
    FRAUD_TYPES = [
        'legitimate',
        'rug_pull',
        'phishing',
        'ponzi_scheme',
        'exchange_hack',
        'ransomware',
        'wash_trading',
        'other'
    ]
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Shared representation
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Binary classification head
        self.binary_head = nn.Linear(hidden_dim, 2)
        
        # Fraud type classification head
        self.type_head = nn.Linear(hidden_dim, len(self.FRAUD_TYPES))
        
        # Severity regression head
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with all task outputs
        """
        
        # Shared representation
        shared = self.shared_layer(x)
        
        # Binary classification
        binary_logits = self.binary_head(shared)
        binary_probs = F.softmax(binary_logits, dim=-1)
        
        # Fraud type classification
        type_logits = self.type_head(shared)
        type_probs = F.softmax(type_logits, dim=-1)
        
        # Severity prediction
        severity = self.severity_head(shared)
        
        return {
            'binary_logits': binary_logits,
            'binary_probs': binary_probs,
            'is_fraud': binary_probs[:, 1],
            'type_logits': type_logits,
            'type_probs': type_probs,
            'fraud_type': torch.argmax(type_probs, dim=-1),
            'severity': severity.squeeze(-1)
        }
    
    def get_fraud_type_name(self, type_idx: int) -> str:
        """Get fraud type name from index."""
        return self.FRAUD_TYPES[type_idx]


if __name__ == '__main__':
    # Example usage
    
    # Test EnsembleClassifier
    ensemble = EnsembleClassifier(
        input_dim=1024,
        mlp_dims=[384, 128, 2],
        num_classes=2
    )
    
    x = torch.randn(32, 1024)
    output = ensemble(x)
    print(f"Ensemble output shape: {output.shape}")
    
    # Test MultiTaskHead
    multi_task = MultiTaskHead(input_dim=256)
    x = torch.randn(32, 256)
    outputs = multi_task(x)
    
    print(f"Binary probs shape: {outputs['binary_probs'].shape}")
    print(f"Type probs shape: {outputs['type_probs'].shape}")
    print(f"Severity shape: {outputs['severity'].shape}")
