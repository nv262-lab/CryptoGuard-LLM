#!/usr/bin/env python3
"""
Training Script for CryptoGuard-LLM

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import CryptoGuardLLM, CryptoGuardConfig


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class Trainer:
    """Trainer class for CryptoGuard-LLM."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        logger: logging.Logger
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.get('learning_rate', 0.01),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100)
        )
        
        # Loss function with class weights
        class_weights = torch.tensor(
            config.get('class_weights', [1.0, 47.0])
        ).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Best metrics
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move data to device
            graph_data = {k: v.to(self.device) for k, v in batch['graph'].items()}
            text_data = {k: v.to(self.device) for k, v in batch['text'].items()}
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(graph_data, text_data)
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
            
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            graph_data = {k: v.to(self.device) for k, v in batch['graph'].items()}
            text_data = {k: v.to(self.device) for k, v in batch['text'].items()}
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(graph_data, text_data)
            loss = self.criterion(outputs['logits'], labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    def train(self, num_epochs: int, checkpoint_dir: str):
        """Full training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        patience = self.config.get('early_stopping_patience', 10)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%"
            )
            
            # Validate
            val_metrics = self.validate()
            self.logger.info(
                f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Val F1: {val_metrics['f1']:.2f}%"
            )
            
            # Update scheduler
            self.scheduler.step()
            
            # Check for improvement
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self.model.save_checkpoint(
                    os.path.join(checkpoint_dir, 'cryptoguard_best.pt'),
                    self.optimizer,
                    epoch,
                    val_metrics
                )
                self.logger.info(f"New best model saved with F1: {self.best_val_f1:.2f}%")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.model.save_checkpoint(
                    os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'),
                    self.optimizer,
                    epoch,
                    val_metrics
                )
                
        self.logger.info(f"Training complete. Best F1: {self.best_val_f1:.2f}% at epoch {self.best_epoch}")


def main():
    parser = argparse.ArgumentParser(description='Train CryptoGuard-LLM')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting CryptoGuard-LLM training")
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {}
        logger.warning(f"Config file not found: {args.config}. Using defaults.")
    
    # Override config with command line arguments
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
        
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model_config = CryptoGuardConfig(
        gnn_num_layers=config.get('gnn_num_layers', 4),
        gnn_hidden_dim=config.get('gnn_hidden_dim', 256),
        gnn_attention_heads=config.get('gnn_attention_heads', 8)
    )
    model = CryptoGuardLLM(model_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        model.load_pretrained(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # TODO: Initialize data loaders with actual dataset
    # train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 1024), shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 1024), shuffle=False)
    
    logger.info("To run training, implement data loading for your dataset.")
    logger.info("See src/data/dataset.py for dataset class implementation.")
    

if __name__ == '__main__':
    main()
