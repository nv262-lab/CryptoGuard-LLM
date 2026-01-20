#!/usr/bin/env python3
"""
Inference Script for CryptoGuard-LLM

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import argparse
import json
import logging
from pathlib import Path

import torch

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import CryptoGuardLLM


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_transactions(input_path: str):
    """Load transactions from file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def save_predictions(predictions, output_path: str):
    """Save predictions to file."""
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='CryptoGuard-LLM Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/cryptoguard_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--tx', type=str, default=None,
                        help='Single transaction hash to analyze')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input JSON file with transactions')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Path to output predictions file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--stream', action='store_true',
                        help='Enable streaming mode for real-time monitoring')
    parser.add_argument('--source', type=str, default=None,
                        help='WebSocket source for streaming mode')
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting CryptoGuard-LLM inference")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = CryptoGuardLLM()
    
    if Path(args.checkpoint).exists():
        model.load_pretrained(args.checkpoint)
        logger.info(f"Loaded model from {args.checkpoint}")
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}. Using untrained model.")
    
    model.to(device)
    model.eval()
    
    # Run inference
    if args.tx:
        # Single transaction
        logger.info(f"Analyzing transaction: {args.tx}")
        # TODO: Fetch transaction data and run prediction
        print(f"Transaction analysis for {args.tx} - implement data fetching")
        
    elif args.input:
        # Batch inference
        logger.info(f"Loading transactions from {args.input}")
        transactions = load_transactions(args.input)
        logger.info(f"Loaded {len(transactions)} transactions")
        
        # TODO: Run predictions
        # predictions = model.predict(transactions, threshold=args.threshold)
        # save_predictions(predictions, args.output)
        
        print(f"Batch inference - implement transaction processing")
        
    elif args.stream:
        # Streaming mode
        logger.info(f"Starting streaming mode from {args.source}")
        print("Streaming mode - implement WebSocket connection")
        
    else:
        logger.error("Please specify --tx, --input, or --stream")
        return
    
    logger.info("Inference complete")


if __name__ == '__main__':
    main()
