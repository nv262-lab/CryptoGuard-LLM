#!/usr/bin/env python3
"""
Download Pre-trained Models for CryptoGuard-LLM

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import os
import argparse
import logging
from pathlib import Path
import hashlib

# For downloading
import urllib.request
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs (placeholder - replace with actual URLs after model hosting)
MODEL_URLS = {
    'cryptoguard_best': {
        'url': 'https://example.com/models/cryptoguard_best.pt',
        'md5': 'placeholder_md5_hash',
        'size': '500MB'
    },
    'bert_security': {
        'url': 'https://example.com/models/bert_security.pt', 
        'md5': 'placeholder_md5_hash',
        'size': '400MB'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def verify_md5(file_path: str, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5


def main():
    parser = argparse.ArgumentParser(description='Download CryptoGuard-LLM models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'cryptoguard_best', 'bert_security'],
                        help='Model to download')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for models')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if exists')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_download = list(MODEL_URLS.keys()) if args.model == 'all' else [args.model]
    
    for model_name in models_to_download:
        model_info = MODEL_URLS[model_name]
        output_path = output_dir / f"{model_name}.pt"
        
        if output_path.exists() and not args.force:
            logger.info(f"{model_name} already exists. Use --force to re-download.")
            continue
            
        logger.info(f"Downloading {model_name} ({model_info['size']})...")
        
        try:
            download_file(model_info['url'], str(output_path))
            
            # Verify checksum
            if model_info['md5'] != 'placeholder_md5_hash':
                if verify_md5(str(output_path), model_info['md5']):
                    logger.info(f"Successfully downloaded and verified {model_name}")
                else:
                    logger.error(f"MD5 verification failed for {model_name}")
                    output_path.unlink()
            else:
                logger.info(f"Downloaded {model_name} (checksum verification skipped)")
                
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            logger.info("Note: Pre-trained models will be available after paper acceptance.")
            
    logger.info("Download complete!")
    logger.info("\nNote: Pre-trained model weights will be released upon paper acceptance.")
    logger.info("For now, you can train your own model using: python scripts/train.py")


if __name__ == '__main__':
    main()
