"""
Explainability Module for CryptoGuard-LLM

Provides interpretable explanations for fraud detection predictions.

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ExplainabilityModule:
    """
    Main explainability class for CryptoGuard-LLM.
    
    Provides:
    - Feature importance analysis
    - Attention visualization
    - Natural language explanations
    - Counterfactual analysis
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.feature_names = None
        
    def set_feature_names(self, names: List[str]):
        """Set human-readable feature names."""
        self.feature_names = names
        
    def explain_prediction(
        self,
        graph_data: Dict[str, torch.Tensor],
        text_data: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Generate explanation for a single prediction.
        
        Args:
            graph_data: Graph input data
            text_data: Text input data
            prediction: Model prediction
            
        Returns:
            Dictionary containing explanation components
        """
        explanation = {
            'prediction': 'Fraud' if prediction['predictions'].item() == 1 else 'Legitimate',
            'confidence': prediction['fraud_probability'].item(),
            'anomaly_score': prediction['anomaly_score'].item(),
            'factors': [],
            'natural_language': ''
        }
        
        # Get feature importance
        importance = self._compute_feature_importance(graph_data, text_data)
        explanation['feature_importance'] = importance
        
        # Get top contributing factors
        top_factors = self._get_top_factors(importance)
        explanation['factors'] = top_factors
        
        # Generate natural language explanation
        explanation['natural_language'] = self._generate_nl_explanation(
            explanation['prediction'],
            explanation['confidence'],
            top_factors
        )
        
        return explanation
    
    def _compute_feature_importance(
        self,
        graph_data: Dict[str, torch.Tensor],
        text_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute feature importance using gradient-based method."""
        importance = {}
        
        # Enable gradient computation
        self.model.eval()
        
        # For graph features
        if 'node_features' in graph_data:
            node_features = graph_data['node_features'].clone().requires_grad_(True)
            
            # Forward pass
            outputs = self.model(
                {**graph_data, 'node_features': node_features},
                text_data
            )
            
            # Backward pass to get gradients
            outputs['fraud_probability'].backward()
            
            # Compute importance as gradient magnitude
            gradients = node_features.grad.abs().mean(dim=0).cpu().numpy()
            
            for i, grad in enumerate(gradients):
                name = self.feature_names[i] if self.feature_names else f'feature_{i}'
                importance[f'graph_{name}'] = float(grad)
                
        return importance
    
    def _get_top_factors(
        self,
        importance: Dict[str, float],
        top_k: int = 5
    ) -> List[Dict]:
        """Get top contributing factors."""
        sorted_factors = sorted(
            importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        factors = []
        for name, value in sorted_factors[:top_k]:
            factors.append({
                'feature': name,
                'importance': value,
                'direction': 'increases' if value > 0 else 'decreases'
            })
            
        return factors
    
    def _generate_nl_explanation(
        self,
        prediction: str,
        confidence: float,
        factors: List[Dict]
    ) -> str:
        """Generate natural language explanation."""
        if confidence < 0.3:
            confidence_text = "low confidence"
        elif confidence < 0.7:
            confidence_text = "moderate confidence"
        else:
            confidence_text = "high confidence"
            
        explanation = f"The transaction is classified as {prediction} with {confidence_text} "
        explanation += f"(probability: {confidence:.1%}). "
        
        if factors:
            explanation += "Key factors: "
            factor_texts = []
            for f in factors[:3]:
                factor_texts.append(
                    f"{f['feature']} {f['direction']} fraud likelihood"
                )
            explanation += "; ".join(factor_texts) + "."
            
        return explanation


class GNNExplainer:
    """
    Explainability for Graph Neural Network predictions.
    
    Identifies important:
    - Nodes (wallets/contracts)
    - Edges (transactions)
    - Subgraph patterns
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_hops: int = 2,
        num_epochs: int = 100
    ):
        self.model = model
        self.num_hops = num_hops
        self.num_epochs = num_epochs
        
    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Explain prediction for a specific node.
        
        Args:
            node_idx: Index of node to explain
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            
        Returns:
            Explanation dictionary
        """
        # Initialize edge mask
        num_edges = edge_index.shape[1]
        edge_mask = nn.Parameter(torch.ones(num_edges))
        
        # Optimizer for mask
        optimizer = torch.optim.Adam([edge_mask], lr=0.01)
        
        # Get target prediction
        self.model.eval()
        with torch.no_grad():
            target_out = self.model.gnn(x, edge_index, edge_attr)
            target_pred = target_out[node_idx]
            
        # Optimize mask
        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Masked forward pass
            masked_edge_attr = edge_attr * edge_mask.sigmoid().unsqueeze(-1) if edge_attr is not None else None
            out = self.model.gnn(x, edge_index, masked_edge_attr)
            
            # Loss: prediction similarity + sparsity
            pred_loss = nn.functional.mse_loss(out[node_idx], target_pred)
            sparsity_loss = edge_mask.sigmoid().mean()
            loss = pred_loss + 0.1 * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
        # Get important edges
        edge_importance = edge_mask.sigmoid().detach().cpu().numpy()
        important_edges = np.where(edge_importance > 0.5)[0]
        
        return {
            'node_idx': node_idx,
            'edge_importance': edge_importance,
            'important_edges': important_edges.tolist(),
            'num_important_edges': len(important_edges)
        }
    
    def get_important_subgraph(
        self,
        node_idx: int,
        edge_index: torch.Tensor,
        edge_importance: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """Extract the important subgraph for a node."""
        # Get edges above threshold
        important_mask = edge_importance > threshold
        important_edge_index = edge_index[:, important_mask]
        
        # Get unique nodes in subgraph
        nodes = torch.unique(important_edge_index).tolist()
        
        return {
            'center_node': node_idx,
            'nodes': nodes,
            'edges': important_edge_index.tolist(),
            'num_nodes': len(nodes),
            'num_edges': important_edge_index.shape[1]
        }


class AttentionVisualizer:
    """
    Visualizes attention weights from transformer models.
    
    Helps understand which parts of text input
    contribute to fraud detection.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def get_attention_weights(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> np.ndarray:
        """
        Extract attention weights from BERT model.
        
        Args:
            model: BERT model
            input_ids: Tokenized input
            attention_mask: Attention mask
            
        Returns:
            Attention weights array
        """
        model.eval()
        
        with torch.no_grad():
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
        # Get attention from last layer, average over heads
        attentions = outputs.attentions[-1]  # Last layer
        attention_weights = attentions.mean(dim=1)  # Average over heads
        
        # Get CLS attention to all tokens
        cls_attention = attention_weights[0, 0, :].cpu().numpy()
        
        return cls_attention
    
    def visualize_text_attention(
        self,
        text: str,
        attention_weights: np.ndarray,
        top_k: int = 10
    ) -> Dict:
        """
        Create attention visualization for text.
        
        Args:
            text: Original text
            attention_weights: Attention weights
            top_k: Number of top tokens to highlight
            
        Returns:
            Visualization data dictionary
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Align attention with tokens (skip special tokens)
        token_attention = attention_weights[1:len(tokens)+1]
        
        # Normalize
        token_attention = token_attention / token_attention.max()
        
        # Get top attended tokens
        top_indices = np.argsort(token_attention)[-top_k:][::-1]
        top_tokens = [(tokens[i], float(token_attention[i])) for i in top_indices]
        
        return {
            'tokens': tokens,
            'attention': token_attention.tolist(),
            'top_tokens': top_tokens,
            'highlighted_text': self._highlight_text(tokens, token_attention)
        }
    
    def _highlight_text(
        self,
        tokens: List[str],
        attention: np.ndarray,
        threshold: float = 0.5
    ) -> str:
        """Create highlighted text representation."""
        highlighted = []
        
        for token, att in zip(tokens, attention):
            if att > threshold:
                highlighted.append(f"**{token}**")
            else:
                highlighted.append(token)
                
        return ' '.join(highlighted)


class CounterfactualExplainer:
    """
    Generates counterfactual explanations.
    
    Shows what minimal changes would flip the prediction.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def generate_counterfactual(
        self,
        x: torch.Tensor,
        target_class: int,
        max_iterations: int = 100,
        step_size: float = 0.01
    ) -> Dict:
        """
        Generate counterfactual example.
        
        Args:
            x: Original input
            target_class: Target class for counterfactual
            max_iterations: Maximum optimization steps
            step_size: Step size for optimization
            
        Returns:
            Counterfactual explanation dictionary
        """
        x_cf = x.clone().requires_grad_(True)
        original_pred = None
        
        self.model.eval()
        
        for i in range(max_iterations):
            # Forward pass
            output = self.model(x_cf)
            pred = output.argmax(dim=-1).item()
            
            if original_pred is None:
                original_pred = pred
                
            # Check if we've reached target
            if pred == target_class:
                break
                
            # Compute loss toward target
            loss = nn.functional.cross_entropy(
                output,
                torch.tensor([target_class])
            )
            
            # Backward pass
            loss.backward()
            
            # Update counterfactual
            with torch.no_grad():
                x_cf -= step_size * x_cf.grad
                x_cf.grad.zero_()
                
        # Compute changes
        changes = (x_cf - x).detach().cpu().numpy()
        
        return {
            'original_prediction': original_pred,
            'counterfactual_prediction': pred,
            'counterfactual': x_cf.detach().cpu().numpy(),
            'changes': changes,
            'l2_distance': np.linalg.norm(changes),
            'num_features_changed': np.sum(np.abs(changes) > 0.01)
        }


if __name__ == '__main__':
    print("Explainability Module for CryptoGuard-LLM")
    print("Provides interpretable explanations for fraud detection predictions")
