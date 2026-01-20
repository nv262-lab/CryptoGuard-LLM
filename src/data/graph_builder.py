"""
Transaction Graph Builder for CryptoGuard-LLM

Constructs graph representations of cryptocurrency transactions
for Graph Neural Network processing.

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TransactionGraphBuilder:
    """
    Builds transaction graphs from cryptocurrency data.
    
    Creates heterogeneous graphs with:
    - Nodes: Wallets, Contracts, Exchanges
    - Edges: Transactions, Contract calls, Token transfers
    
    Args:
        include_contracts: Whether to include contract nodes
        include_temporal: Whether to add temporal edge features
        max_neighbors: Maximum neighbors per node for sampling
    """
    
    # Node types
    NODE_TYPES = ['wallet', 'contract', 'exchange']
    
    # Edge types
    EDGE_TYPES = ['transfer', 'contract_call', 'token_transfer', 'internal']
    
    def __init__(
        self,
        include_contracts: bool = True,
        include_temporal: bool = True,
        max_neighbors: int = 50
    ):
        self.include_contracts = include_contracts
        self.include_temporal = include_temporal
        self.max_neighbors = max_neighbors
        
        # Node and edge mappings
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.node_types = {}
        
        # Known contract and exchange addresses
        self.known_contracts: Set[str] = set()
        self.known_exchanges: Set[str] = set()
        
    def build_graph(
        self,
        transactions: List[Dict],
        target_addresses: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Build a transaction graph from a list of transactions.
        
        Args:
            transactions: List of transaction dictionaries
            target_addresses: Optional list of addresses to focus on
            
        Returns:
            Dictionary containing graph tensors
        """
        # Reset mappings
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.node_types = {}
        
        # Build node set
        self._build_node_set(transactions, target_addresses)
        
        # Build edges
        edge_index, edge_attr, edge_types = self._build_edges(transactions)
        
        # Build node features
        node_features = self._build_node_features(transactions)
        
        graph = {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(edge_attr, dtype=torch.float32),
            'edge_types': torch.tensor(edge_types, dtype=torch.long),
            'num_nodes': len(self.node_to_idx),
            'num_edges': edge_index.shape[1] if len(edge_index) > 0 else 0
        }
        
        logger.info(
            f"Built graph with {graph['num_nodes']} nodes and {graph['num_edges']} edges"
        )
        
        return graph
    
    def _build_node_set(
        self,
        transactions: List[Dict],
        target_addresses: Optional[List[str]] = None
    ):
        """Build the set of nodes from transactions."""
        addresses = set()
        
        for tx in transactions:
            from_addr = tx.get('from')
            to_addr = tx.get('to')
            
            if from_addr:
                addresses.add(from_addr.lower())
            if to_addr:
                addresses.add(to_addr.lower())
                
        # Filter to target addresses if specified
        if target_addresses:
            target_set = set(addr.lower() for addr in target_addresses)
            # Include targets and their direct neighbors
            relevant = set()
            for tx in transactions:
                from_addr = tx.get('from', '').lower()
                to_addr = tx.get('to', '').lower()
                if from_addr in target_set or to_addr in target_set:
                    relevant.add(from_addr)
                    relevant.add(to_addr)
            addresses = addresses.intersection(relevant)
            
        # Create node mappings
        for idx, addr in enumerate(sorted(addresses)):
            self.node_to_idx[addr] = idx
            self.idx_to_node[idx] = addr
            self.node_types[addr] = self._get_node_type(addr)
            
    def _get_node_type(self, address: str) -> int:
        """Determine the type of a node."""
        address = address.lower()
        
        if address in self.known_exchanges:
            return 2  # exchange
        elif address in self.known_contracts:
            return 1  # contract
        else:
            return 0  # wallet
            
    def _build_edges(
        self,
        transactions: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build edge index and features."""
        edges = []
        edge_features = []
        edge_types = []
        
        for tx in transactions:
            from_addr = tx.get('from', '').lower()
            to_addr = tx.get('to', '').lower()
            
            # Skip if addresses not in node set
            if from_addr not in self.node_to_idx or to_addr not in self.node_to_idx:
                continue
                
            from_idx = self.node_to_idx[from_addr]
            to_idx = self.node_to_idx[to_addr]
            
            # Add edge
            edges.append([from_idx, to_idx])
            
            # Edge features
            features = self._extract_edge_features(tx)
            edge_features.append(features)
            
            # Edge type
            edge_type = self._get_edge_type(tx)
            edge_types.append(edge_type)
            
        if not edges:
            return np.zeros((2, 0)), np.zeros((0, 64)), np.zeros(0)
            
        edge_index = np.array(edges).T
        edge_attr = np.array(edge_features)
        edge_types = np.array(edge_types)
        
        return edge_index, edge_attr, edge_types
    
    def _extract_edge_features(self, tx: Dict) -> List[float]:
        """Extract features for an edge (transaction)."""
        features = []
        
        # Value (normalized)
        value = float(tx.get('value', 0))
        features.append(np.log1p(value))
        
        # Gas price
        gas_price = float(tx.get('gasPrice', 0))
        features.append(np.log1p(gas_price))
        
        # Gas used
        gas_used = float(tx.get('gasUsed', 0))
        features.append(np.log1p(gas_used))
        
        # Input data length
        input_data = tx.get('input', '0x')
        features.append(len(input_data) / 1000)  # Normalized
        
        # Temporal features
        if self.include_temporal:
            timestamp = tx.get('timestamp', 0)
            dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
            features.append(dt.hour / 24)  # Hour normalized
            features.append(dt.weekday() / 7)  # Day normalized
            
        # Pad to fixed size
        while len(features) < 64:
            features.append(0.0)
            
        return features[:64]
    
    def _get_edge_type(self, tx: Dict) -> int:
        """Determine the type of an edge."""
        input_data = tx.get('input', '0x')
        to_addr = tx.get('to', '').lower()
        
        if to_addr in self.known_contracts:
            return 1  # contract_call
        elif len(input_data) > 10:
            # Check for token transfer signature
            if input_data.startswith('0xa9059cbb'):  # ERC20 transfer
                return 2  # token_transfer
            return 1  # contract_call
        else:
            return 0  # transfer
            
    def _build_node_features(self, transactions: List[Dict]) -> np.ndarray:
        """Build node feature matrix."""
        num_nodes = len(self.node_to_idx)
        feature_dim = 128
        
        # Initialize with zeros
        node_features = np.zeros((num_nodes, feature_dim))
        
        # Aggregate transaction statistics per node
        node_stats = defaultdict(lambda: {
            'in_count': 0, 'out_count': 0,
            'in_value': 0, 'out_value': 0,
            'in_addresses': set(), 'out_addresses': set(),
            'timestamps': []
        })
        
        for tx in transactions:
            from_addr = tx.get('from', '').lower()
            to_addr = tx.get('to', '').lower()
            value = float(tx.get('value', 0))
            timestamp = tx.get('timestamp', 0)
            
            if from_addr in self.node_to_idx:
                node_stats[from_addr]['out_count'] += 1
                node_stats[from_addr]['out_value'] += value
                node_stats[from_addr]['out_addresses'].add(to_addr)
                node_stats[from_addr]['timestamps'].append(timestamp)
                
            if to_addr in self.node_to_idx:
                node_stats[to_addr]['in_count'] += 1
                node_stats[to_addr]['in_value'] += value
                node_stats[to_addr]['in_addresses'].add(from_addr)
                node_stats[to_addr]['timestamps'].append(timestamp)
                
        # Convert stats to features
        for addr, idx in self.node_to_idx.items():
            stats = node_stats[addr]
            
            # Basic counts
            node_features[idx, 0] = np.log1p(stats['in_count'])
            node_features[idx, 1] = np.log1p(stats['out_count'])
            node_features[idx, 2] = np.log1p(stats['in_value'])
            node_features[idx, 3] = np.log1p(stats['out_value'])
            
            # Unique counterparties
            node_features[idx, 4] = len(stats['in_addresses'])
            node_features[idx, 5] = len(stats['out_addresses'])
            
            # Ratios
            total_count = stats['in_count'] + stats['out_count']
            if total_count > 0:
                node_features[idx, 6] = stats['in_count'] / total_count
                node_features[idx, 7] = stats['out_count'] / total_count
                
            # Temporal spread
            if stats['timestamps']:
                timestamps = stats['timestamps']
                node_features[idx, 8] = np.std(timestamps) / 86400 if len(timestamps) > 1 else 0
                
            # Node type (one-hot)
            node_type = self.node_types.get(addr, 0)
            node_features[idx, 10 + node_type] = 1.0
            
        return node_features
    
    def add_known_contracts(self, contracts: List[str]):
        """Add known contract addresses."""
        self.known_contracts.update(addr.lower() for addr in contracts)
        
    def add_known_exchanges(self, exchanges: List[str]):
        """Add known exchange addresses."""
        self.known_exchanges.update(addr.lower() for addr in exchanges)
        
    def get_subgraph(
        self,
        center_address: str,
        transactions: List[Dict],
        hops: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Extract a subgraph centered on an address.
        
        Args:
            center_address: Address to center the subgraph on
            transactions: List of all transactions
            hops: Number of hops to include
            
        Returns:
            Subgraph as dictionary of tensors
        """
        center = center_address.lower()
        
        # BFS to find neighbors within hops
        visited = {center}
        frontier = {center}
        
        for _ in range(hops):
            new_frontier = set()
            for tx in transactions:
                from_addr = tx.get('from', '').lower()
                to_addr = tx.get('to', '').lower()
                
                if from_addr in frontier:
                    new_frontier.add(to_addr)
                if to_addr in frontier:
                    new_frontier.add(from_addr)
                    
            frontier = new_frontier - visited
            visited.update(frontier)
            
            if not frontier:
                break
                
        # Filter transactions to subgraph
        subgraph_txs = [
            tx for tx in transactions
            if tx.get('from', '').lower() in visited
            and tx.get('to', '').lower() in visited
        ]
        
        return self.build_graph(subgraph_txs, list(visited))


class BatchGraphBuilder:
    """
    Builds batched graphs for efficient training.
    
    Combines multiple transaction graphs into a single
    batched graph for parallel processing.
    """
    
    def __init__(self, graph_builder: TransactionGraphBuilder):
        self.graph_builder = graph_builder
        
    def build_batch(
        self,
        transaction_groups: List[List[Dict]]
    ) -> Dict[str, torch.Tensor]:
        """
        Build a batched graph from multiple transaction groups.
        
        Args:
            transaction_groups: List of transaction lists
            
        Returns:
            Batched graph dictionary
        """
        all_node_features = []
        all_edge_index = []
        all_edge_attr = []
        batch_assignments = []
        
        node_offset = 0
        
        for batch_idx, transactions in enumerate(transaction_groups):
            graph = self.graph_builder.build_graph(transactions)
            
            all_node_features.append(graph['node_features'])
            
            # Offset edge indices
            edge_index = graph['edge_index'] + node_offset
            all_edge_index.append(edge_index)
            all_edge_attr.append(graph['edge_attr'])
            
            # Batch assignment
            batch_assignments.extend([batch_idx] * graph['num_nodes'])
            
            node_offset += graph['num_nodes']
            
        return {
            'node_features': torch.cat(all_node_features, dim=0),
            'edge_index': torch.cat(all_edge_index, dim=1),
            'edge_attr': torch.cat(all_edge_attr, dim=0),
            'batch': torch.tensor(batch_assignments, dtype=torch.long)
        }


if __name__ == '__main__':
    # Example usage
    print("TransactionGraphBuilder - Graph construction for cryptocurrency data")
    
    builder = TransactionGraphBuilder()
    
    # Sample transactions
    sample_txs = [
        {
            'from': '0x1111111111111111111111111111111111111111',
            'to': '0x2222222222222222222222222222222222222222',
            'value': 1000000000000000000,
            'gasPrice': 20000000000,
            'gasUsed': 21000,
            'input': '0x',
            'timestamp': 1704067200
        },
        {
            'from': '0x2222222222222222222222222222222222222222',
            'to': '0x3333333333333333333333333333333333333333',
            'value': 500000000000000000,
            'gasPrice': 25000000000,
            'gasUsed': 21000,
            'input': '0x',
            'timestamp': 1704067300
        }
    ]
    
    graph = builder.build_graph(sample_txs)
    print(f"Built graph:")
    print(f"  Nodes: {graph['num_nodes']}")
    print(f"  Edges: {graph['num_edges']}")
    print(f"  Node features shape: {graph['node_features'].shape}")
    print(f"  Edge features shape: {graph['edge_attr'].shape}")
