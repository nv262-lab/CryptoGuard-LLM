# CryptoGuard-LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

**A Multi-Modal Deep Learning Framework for Real-Time Cryptocurrency Fraud Detection with Societal Impact Analysis**

## 📋 Overview

CryptoGuard-LLM is a comprehensive multi-modal deep learning framework that integrates Large Language Models (LLMs), Graph Neural Networks (GNNs), and ensemble machine learning techniques for real-time detection of cryptocurrency-related cyber threats.

### Key Features

- 🔍 **Multi-Modal Analysis**: Combines blockchain transaction graphs, exchange data, social media sentiment, and threat intelligence
- 🧠 **LLM Integration**: Fine-tuned BERT for threat classification + GPT-4 for semantic analysis
- 📊 **Graph Neural Networks**: Heterogeneous Graph Attention Networks (HGAT) for transaction pattern detection
- ⚡ **Real-Time Processing**: 2,847 transactions/second with 1.2s latency (95th percentile)
- 🎯 **High Accuracy**: 96.8% detection accuracy, 94.7% recall, 95.2% F1-score

## 📈 Performance

| Fraud Type | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Rug Pulls | 97.2% | 96.8% | 97.0% | 284,521 |
| Phishing | 95.8% | 93.4% | 94.6% | 241,892 |
| Ponzi Schemes | 96.4% | 94.9% | 95.6% | 178,234 |
| Exchange Hacks | 98.1% | 97.2% | 97.6% | 148,762 |
| Ransomware | 94.3% | 91.8% | 93.0% | 98,234 |
| **Overall** | **96.8%** | **94.7%** | **95.2%** | **993,127** |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                        │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│ Blockchain  │  Exchange   │   Social    │   Threat    │ Dark   │
│ Transactions│    APIs     │   Media     │   Intel     │ Web    │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴───┬────┘
       │             │             │             │          │
       ▼             ▼             ▼             ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ML/LLM PROCESSING LAYER                       │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│    GNN      │ Transformer │   GPT-4     │  Ensemble   │Anomaly │
│  (HGAT)     │   Encoder   │  Analysis   │ Classifier  │Detector│
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴───┬────┘
       │             │             │             │          │
       ▼             ▼             ▼             ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│               THREAT DETECTION & RESPONSE LAYER                 │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│    Fraud    │   Money     │  Phishing   │    Ransomware       │
│  Detection  │ Laundering  │   Attacks   │     Payments        │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────────────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│        Alert Dashboard | Regulatory Reports | API Endpoints     │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 2.1.0
- CUDA 12.1 (for GPU acceleration)
- 32GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/[username]/CryptoGuard-LLM.git
cd CryptoGuard-LLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

### Basic Usage

```python
from src.models import CryptoGuardLLM

# Initialize the model
model = CryptoGuardLLM(
    gnn_layers=4,
    hidden_dim=256,
    attention_heads=8,
    bert_model='bert-base-uncased'
)

# Load pre-trained weights
model.load_pretrained('checkpoints/cryptoguard_best.pt')

# Detect fraud in transactions
transactions = load_transactions('data/sample_transactions.json')
predictions = model.predict(transactions)

# Get detailed analysis
for tx, pred in zip(transactions, predictions):
    print(f"Transaction: {tx['hash']}")
    print(f"Fraud Probability: {pred['probability']:.4f}")
    print(f"Fraud Type: {pred['fraud_type']}")
    print(f"Explanation: {pred['explanation']}")
```

## 📁 Repository Structure

```
CryptoGuard-LLM/
├── paper/                      # IEEE TIFS paper
│   ├── CryptoGuard_LLM_TIFS_final.tex
│   └── figures/
│       ├── fig1_architecture.png
│       ├── fig2_performance.png
│       ├── fig3_temporal.png
│       └── fig4_distribution.png
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn.py              # Graph Neural Network
│   │   ├── bert_classifier.py  # BERT fine-tuning
│   │   ├── ensemble.py         # Ensemble classifier
│   │   └── cryptoguard.py      # Main model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset classes
│   │   ├── preprocessing.py    # Data preprocessing
│   │   └── graph_builder.py    # Transaction graph construction
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # Evaluation metrics
│       ├── visualization.py    # Plotting utilities
│       └── explainability.py   # Model explanations
├── configs/
│   ├── model_config.yaml       # Model hyperparameters
│   ├── training_config.yaml    # Training settings
│   └── data_config.yaml        # Data processing settings
├── scripts/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── inference.py            # Inference script
│   └── download_models.py      # Download pre-trained models
├── docs/
│   ├── installation.md
│   ├── usage.md
│   └── api_reference.md
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
gnn:
  num_layers: 4
  hidden_dim: 256
  attention_heads: 8
  dropout: 0.3
  activation: leaky_relu
  negative_slope: 0.2

bert:
  model_name: bert-base-uncased
  max_length: 512
  learning_rate: 2e-5

ensemble:
  mlp_dims: [384, 128, 2]
  class_weights: auto
```

### Training Configuration (`configs/training_config.yaml`)

```yaml
training:
  epochs: 100
  batch_size: 1024
  optimizer: sgd
  momentum: 0.9
  learning_rate: 0.01
  scheduler: cosine_annealing
  weight_decay: 1e-4
  early_stopping_patience: 10

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  temporal_split: true
  stratified_sampling: true
```

## 📊 Dataset

Our experimental dataset comprises **47.3 million transactions** collected between January 2024 and December 2025:

| Blockchain | Transactions (M) | Illicit (%) | Time Period |
|------------|------------------|-------------|-------------|
| Bitcoin | 18.7 | 1.8% | Jan 2024 - Dec 2025 |
| Ethereum | 21.4 | 2.4% | Jan 2024 - Dec 2025 |
| Altcoins (15) | 7.2 | 2.3% | Jan 2024 - Dec 2025 |
| **Total** | **47.3** | **2.1%** | **24 months** |

### Data Access

The anonymized transaction dataset is available upon reasonable request to qualified researchers. Please contact the corresponding author with:
- Research institution affiliation
- Intended use case
- Agreement to data use terms (no deanonymization attempts)

## 🧪 Experiments

### Training

```bash
# Train from scratch
python scripts/train.py --config configs/training_config.yaml

# Resume training
python scripts/train.py --config configs/training_config.yaml --resume checkpoints/last.pt

# Train with custom hyperparameters
python scripts/train.py --lr 0.001 --batch_size 512 --epochs 50
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/cryptoguard_best.pt

# Generate detailed metrics
python scripts/evaluate.py --checkpoint checkpoints/cryptoguard_best.pt --detailed

# Cross-validation
python scripts/evaluate.py --checkpoint checkpoints/cryptoguard_best.pt --cv 5
```

### Inference

```bash
# Single transaction
python scripts/inference.py --tx "0x123..."

# Batch inference
python scripts/inference.py --input data/transactions.json --output results/predictions.json

# Real-time monitoring
python scripts/inference.py --stream --source ws://exchange-api
```

## 📖 Citation

If you use CryptoGuard-LLM in your research, please cite our paper:

```bibtex
@article{vummaneni2026cryptoguard,
  title={CryptoGuard-LLM: A Multi-Modal Deep Learning Framework for Real-Time Cryptocurrency Fraud Detection with Societal Impact Analysis},
  author={Vummaneni, Naga Sujitha and Jammula, Usha Ratnam and Komperla, Ramesh Chandra Aditya},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  publisher={IEEE}
}
```

## 👥 Authors

- **Naga Sujitha Vummaneni** (Senior Member, IEEE) - Cornell University - [nv262@cornell.edu](mailto:nv262@cornell.edu) - [ORCID: 0009-0004-5492-9293](https://orcid.org/0009-0004-5492-9293)
- **Usha Ratnam Jammula** (Member, IEEE) - Independent Researcher - [jammula.usha@gmail.com](mailto:jammula.usha@gmail.com)
- **Ramesh Chandra Aditya Komperla** (Senior Member, IEEE) - Independent Researcher

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📬 Contact

For questions or collaboration inquiries, please contact:
- **Corresponding Author**: Naga Sujitha Vummaneni ([nv262@cornell.edu](mailto:nv262@cornell.edu))

## 🙏 Acknowledgments

- Cornell University for computational resources and research infrastructure
- The blockchain analytics community for publicly available datasets
- The cybersecurity research community for open-source tools and published work

---

**⚠️ Disclaimer**: This framework is intended for research and educational purposes. Users are responsible for ensuring compliance with applicable laws and regulations in their jurisdiction.
