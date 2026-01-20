"""
Dataset Classes for CryptoGuard-LLM

Handles loading and processing of cryptocurrency transaction data
for fraud detection training and evaluation.

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class CryptoFraudDataset(Dataset):
    """
    Main dataset class for cryptocurrency fraud detection.
    
    Handles multi-modal data including:
    - Transaction graph features
    - Textual data (social media, whitepapers, threat intel)
    - Temporal features
    
    Args:
        data_path: Path to dataset directory
        split: One of 'train', 'val', 'test'
        transform: Optional transform to apply to samples
        tokenizer: Tokenizer for text processing
        max_text_length: Maximum length for text sequences
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        transform: Optional[callable] = None,
        tokenizer: Optional[callable] = None,
        max_text_length: int = 512
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # Load data
        self._load_data()
        
        logger.info(f"Loaded {len(self)} samples for {split} split")
        
    def _load_data(self):
        """Load dataset from disk."""
        split_file = self.data_path / f"{self.split}.parquet"
        
        if split_file.exists():
            self.data = pd.read_parquet(split_file)
        else:
            # Try JSON format
            json_file = self.data_path / f"{self.split}.json"
            if json_file.exists():
                self.data = pd.read_json(json_file)
            else:
                raise FileNotFoundError(
                    f"Dataset file not found: {split_file} or {json_file}"
                )
                
        # Validate required columns
        required_cols = ['transaction_hash', 'label']
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self.data.iloc[idx]
        
        # Graph features
        graph_data = self._process_graph_features(row)
        
        # Text features
        text_data = self._process_text_features(row)
        
        # Label
        label = torch.tensor(row['label'], dtype=torch.long)
        
        sample = {
            'graph': graph_data,
            'text': text_data,
            'label': label,
            'transaction_hash': row['transaction_hash']
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def _process_graph_features(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        """Process graph-related features."""
        # Node features
        node_features = row.get('node_features', None)
        if node_features is not None:
            if isinstance(node_features, str):
                node_features = json.loads(node_features)
            node_features = torch.tensor(node_features, dtype=torch.float32)
        else:
            # Default node features
            node_features = torch.zeros(128, dtype=torch.float32)
            
        # Edge index
        edge_index = row.get('edge_index', None)
        if edge_index is not None:
            if isinstance(edge_index, str):
                edge_index = json.loads(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
        # Edge features
        edge_features = row.get('edge_features', None)
        if edge_features is not None:
            if isinstance(edge_features, str):
                edge_features = json.loads(edge_features)
            edge_features = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_features = torch.zeros((0, 64), dtype=torch.float32)
            
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features
        }
    
    def _process_text_features(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        """Process text features using tokenizer."""
        text = row.get('text', '')
        if pd.isna(text):
            text = ''
            
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
        else:
            # Return placeholder if no tokenizer
            return {
                'input_ids': torch.zeros(self.max_text_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_text_length, dtype=torch.long)
            }
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        labels = self.data['label'].values
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        labels = self.data['label'].values
        return {
            'total_samples': len(self),
            'fraud_samples': int(labels.sum()),
            'legitimate_samples': int((1 - labels).sum()),
            'fraud_ratio': float(labels.mean()),
            'class_weights': self.get_class_weights().tolist()
        }


class TransactionDataset(Dataset):
    """
    Dataset for individual cryptocurrency transactions.
    
    Simpler dataset class for transaction-level features
    without graph structure.
    """
    
    def __init__(
        self,
        transactions: pd.DataFrame,
        feature_columns: List[str],
        label_column: str = 'label'
    ):
        self.transactions = transactions
        self.feature_columns = feature_columns
        self.label_column = label_column
        
        # Extract features and labels
        self.features = torch.tensor(
            transactions[feature_columns].values,
            dtype=torch.float32
        )
        self.labels = torch.tensor(
            transactions[label_column].values,
            dtype=torch.long
        )
        
    def __len__(self) -> int:
        return len(self.transactions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class TemporalTransactionDataset(Dataset):
    """
    Dataset with temporal ordering for time-series analysis.
    
    Maintains temporal order and supports sliding window
    approaches for fraud detection.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        window_size: int = 100,
        stride: int = 50,
        timestamp_column: str = 'timestamp'
    ):
        self.data_path = Path(data_path)
        self.window_size = window_size
        self.stride = stride
        self.timestamp_column = timestamp_column
        
        self._load_and_sort_data()
        self._create_windows()
        
    def _load_and_sort_data(self):
        """Load data and sort by timestamp."""
        self.data = pd.read_parquet(self.data_path)
        self.data = self.data.sort_values(self.timestamp_column)
        self.data = self.data.reset_index(drop=True)
        
    def _create_windows(self):
        """Create sliding windows over the data."""
        self.windows = []
        for start in range(0, len(self.data) - self.window_size + 1, self.stride):
            end = start + self.window_size
            self.windows.append((start, end))
            
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start, end = self.windows[idx]
        window_data = self.data.iloc[start:end]
        
        # Process window
        features = torch.tensor(
            window_data.drop(columns=['label', self.timestamp_column]).values,
            dtype=torch.float32
        )
        labels = torch.tensor(window_data['label'].values, dtype=torch.long)
        
        return {
            'features': features,
            'labels': labels,
            'timestamps': window_data[self.timestamp_column].values
        }


def create_data_loaders(
    data_path: Union[str, Path],
    batch_size: int = 1024,
    num_workers: int = 4,
    tokenizer: Optional[callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_path: Path to dataset directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        tokenizer: Tokenizer for text processing
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = CryptoFraudDataset(
        data_path, split='train', tokenizer=tokenizer
    )
    val_dataset = CryptoFraudDataset(
        data_path, split='val', tokenizer=tokenizer
    )
    test_dataset = CryptoFraudDataset(
        data_path, split='test', tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Example usage
    print("CryptoFraudDataset - Dataset classes for cryptocurrency fraud detection")
    print("Usage: dataset = CryptoFraudDataset(data_path='./data', split='train')")
