"""
Heterogeneous Graph Attention Network (HGAT) for Transaction Graph Analysis

This module implements the Graph Neural Network component of CryptoGuard-LLM,
designed to process cryptocurrency transaction graphs with multiple node and edge types.

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
from typing import Optional, Tuple


class GraphAttentionLayer(MessagePassing):
    """
    Graph Attention Layer with multi-head attention mechanism.
    
    Implements attention-based message passing for fraud detection in
    transaction graphs.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        # Linear transformations
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Edge feature transformation
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.att_edge = None
            
        if bias:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)
            
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
            
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, heads * out_channels]
        """
        H, C = self.heads, self.out_channels
        
        # Linear transformation
        x_src = self.lin_src(x).view(-1, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)
        
        # Compute attention scores
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        
        # Propagate
        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            alpha=(alpha_src, alpha_dst),
            edge_attr=edge_attr
        )
        
        out = out.view(-1, H * C)
        
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def message(
        self,
        x_j: torch.Tensor,
        alpha_j: torch.Tensor,
        alpha_i: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int]
    ) -> torch.Tensor:
        """Compute messages with attention weights."""
        
        alpha = alpha_j + alpha_i
        
        if edge_attr is not None and self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha = alpha + (edge_attr * self.att_edge).sum(dim=-1)
            
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)


class HeterogeneousGraphAttentionNetwork(nn.Module):
    """
    Heterogeneous Graph Attention Network for cryptocurrency transaction analysis.
    
    Processes transaction graphs with multiple node types (wallets, contracts, exchanges)
    and edge types (transfers, swaps, contract calls).
    
    Args:
        num_layers: Number of message-passing layers
        hidden_dim: Hidden dimension size
        attention_heads: Number of attention heads
        dropout: Dropout probability
        negative_slope: LeakyReLU negative slope
        node_feature_dim: Input node feature dimension
        edge_feature_dim: Input edge feature dimension
    """
    
    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 256,
        attention_heads: int = 8,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
        node_feature_dim: int = 128,
        edge_feature_dim: int = 64
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        
        # Node feature embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout)
        )
        
        # Edge feature MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Graph attention layers
        self.attention_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim // attention_heads
            
            self.attention_layers.append(
                GraphAttentionLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=attention_heads,
                    negative_slope=negative_slope,
                    dropout=dropout,
                    edge_dim=128
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the HGAT.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph-level embeddings [batch_size, hidden_dim]
        """
        
        # Embed node features
        x = self.node_embedding(x)
        
        # Process edge features
        if edge_attr is not None:
            edge_attr = self.edge_mlp(edge_attr)
        
        # Apply attention layers with residual connections
        for i, (attn_layer, bn) in enumerate(zip(self.attention_layers, self.batch_norms)):
            x_prev = x
            x = attn_layer(x, edge_index, edge_attr)
            x = bn(x)
            x = F.leaky_relu(x, 0.2)
            x = F.dropout(x, p=0.3, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_prev
                
        # Graph-level pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Combine mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        # Output projection
        x = self.output_projection(x)
        
        return x
    
    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get node-level embeddings without pooling."""
        
        x = self.node_embedding(x)
        
        if edge_attr is not None:
            edge_attr = self.edge_mlp(edge_attr)
            
        for i, (attn_layer, bn) in enumerate(zip(self.attention_layers, self.batch_norms)):
            x_prev = x
            x = attn_layer(x, edge_index, edge_attr)
            x = bn(x)
            x = F.leaky_relu(x, 0.2)
            
            if i > 0:
                x = x + x_prev
                
        return x


class TransactionGraphEncoder(nn.Module):
    """
    Encoder for cryptocurrency transaction graphs.
    
    Converts raw transaction data into graph format suitable for the HGAT.
    """
    
    def __init__(
        self,
        wallet_features: int = 32,
        transaction_features: int = 16,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Wallet feature encoder
        self.wallet_encoder = nn.Sequential(
            nn.Linear(wallet_features, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # Transaction feature encoder
        self.transaction_encoder = nn.Sequential(
            nn.Linear(transaction_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
    def forward(
        self,
        wallet_features: torch.Tensor,
        transaction_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode wallet and transaction features."""
        
        node_features = self.wallet_encoder(wallet_features)
        edge_features = self.transaction_encoder(transaction_features)
        
        return node_features, edge_features


if __name__ == '__main__':
    # Example usage
    model = HeterogeneousGraphAttentionNetwork(
        num_layers=4,
        hidden_dim=256,
        attention_heads=8
    )
    
    # Create dummy data
    num_nodes = 100
    num_edges = 300
    
    x = torch.randn(num_nodes, 128)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 64)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Forward pass
    output = model(x, edge_index, edge_attr, batch)
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
