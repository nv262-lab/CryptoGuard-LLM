"""
BERT-based Fraud Classifier for Threat Intelligence Processing

This module implements the NLP component of CryptoGuard-LLM,
using fine-tuned BERT for analyzing security documents, social media,
and threat intelligence feeds.

Authors: Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from typing import Dict, List, Optional, Tuple, Union


class BERTFraudClassifier(nn.Module):
    """
    Fine-tuned BERT model for cryptocurrency fraud detection in text.
    
    Processes:
    - Project whitepapers and documentation
    - Social media posts about cryptocurrency projects
    - Threat intelligence reports
    - Dark web forum discussions
    
    Args:
        model_name: Pre-trained BERT model name
        max_length: Maximum sequence length
        hidden_dim: BERT hidden dimension
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        hidden_dim: int = 768,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Embedding projection (for combining with GNN)
        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Threat indicator extraction head
        self.indicator_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # IOC embedding dimension
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized input [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length]
            
        Returns:
            Dictionary containing logits, probabilities, and embeddings
        """
        
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        probabilities = F.softmax(logits, dim=-1)
        
        # Projected embeddings
        embeddings = self.embedding_projection(cls_output)
        
        # Threat indicators
        indicators = self.indicator_extractor(cls_output)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'embeddings': embeddings,
            'indicators': indicators,
            'cls_output': cls_output
        }
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get text embeddings for ensemble model."""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.embedding_projection(cls_output)
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input text(s)."""
        
        if isinstance(texts, str):
            texts = [texts]
            
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
    
    def classify_text(
        self,
        text: Union[str, List[str]],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Classify text for fraud indicators.
        
        Args:
            text: Input text or list of texts
            threshold: Classification threshold
            
        Returns:
            List of classification results
        """
        self.eval()
        
        if isinstance(text, str):
            text = [text]
            
        # Tokenize
        inputs = self.tokenize(text)
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
        results = []
        for i, t in enumerate(text):
            prob = outputs['probabilities'][i, 1].item()
            results.append({
                'text': t[:100] + '...' if len(t) > 100 else t,
                'is_suspicious': prob >= threshold,
                'fraud_probability': prob,
                'confidence': abs(prob - 0.5) * 2
            })
            
        return results


class ThreatIntelligenceProcessor(nn.Module):
    """
    Specialized processor for threat intelligence feeds.
    
    Extracts and classifies:
    - Indicators of Compromise (IOCs)
    - Threat actor TTPs
    - Malicious wallet addresses
    - Suspicious contract patterns
    """
    
    def __init__(
        self,
        bert_model: BERTFraudClassifier,
        ioc_embedding_dim: int = 128,
        num_ioc_types: int = 10
    ):
        super().__init__()
        
        self.bert_model = bert_model
        
        # IOC type classifier
        self.ioc_classifier = nn.Sequential(
            nn.Linear(bert_model.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_ioc_types)
        )
        
        # Severity predictor
        self.severity_predictor = nn.Sequential(
            nn.Linear(bert_model.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # IOC types
        self.ioc_types = [
            'malicious_wallet',
            'phishing_url',
            'suspicious_contract',
            'scam_token',
            'mixer_service',
            'ransomware_address',
            'ponzi_scheme',
            'rug_pull',
            'wash_trading',
            'other'
        ]
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Process threat intelligence text."""
        
        # Get BERT outputs
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = bert_outputs['cls_output']
        
        # IOC classification
        ioc_logits = self.ioc_classifier(cls_output)
        ioc_probs = F.softmax(ioc_logits, dim=-1)
        
        # Severity prediction
        severity = self.severity_predictor(cls_output)
        
        return {
            **bert_outputs,
            'ioc_logits': ioc_logits,
            'ioc_probabilities': ioc_probs,
            'ioc_types': torch.argmax(ioc_probs, dim=-1),
            'severity': severity.squeeze(-1)
        }
    
    def extract_iocs(
        self,
        text: str
    ) -> List[Dict]:
        """Extract IOCs from threat intelligence text."""
        
        self.eval()
        inputs = self.bert_model.tokenize(text)
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
        ioc_type_idx = outputs['ioc_types'].item()
        
        return {
            'text': text,
            'ioc_type': self.ioc_types[ioc_type_idx],
            'ioc_probability': outputs['ioc_probabilities'][0, ioc_type_idx].item(),
            'severity': outputs['severity'].item(),
            'fraud_probability': outputs['probabilities'][0, 1].item()
        }


class SocialMediaAnalyzer(nn.Module):
    """
    Analyzer for cryptocurrency-related social media content.
    
    Detects:
    - Pump and dump schemes
    - Coordinated shilling
    - FUD campaigns
    - Scam promotions
    """
    
    SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
    MANIPULATION_TYPES = ['organic', 'coordinated_pump', 'fud_campaign', 'scam_promotion']
    
    def __init__(
        self,
        bert_model: BERTFraudClassifier
    ):
        super().__init__()
        
        self.bert_model = bert_model
        hidden_dim = bert_model.hidden_dim
        
        # Sentiment classifier
        self.sentiment_classifier = nn.Linear(hidden_dim, 3)
        
        # Manipulation detector
        self.manipulation_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.MANIPULATION_TYPES))
        )
        
        # Bot detection
        self.bot_detector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze social media content."""
        
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = bert_outputs['cls_output']
        
        # Sentiment
        sentiment_logits = self.sentiment_classifier(cls_output)
        sentiment_probs = F.softmax(sentiment_logits, dim=-1)
        
        # Manipulation detection
        manipulation_logits = self.manipulation_detector(cls_output)
        manipulation_probs = F.softmax(manipulation_logits, dim=-1)
        
        # Bot probability
        bot_prob = self.bot_detector(cls_output)
        
        return {
            **bert_outputs,
            'sentiment_logits': sentiment_logits,
            'sentiment_probs': sentiment_probs,
            'sentiment': torch.argmax(sentiment_probs, dim=-1),
            'manipulation_logits': manipulation_logits,
            'manipulation_probs': manipulation_probs,
            'manipulation_type': torch.argmax(manipulation_probs, dim=-1),
            'bot_probability': bot_prob.squeeze(-1)
        }
    
    def analyze_post(self, text: str) -> Dict:
        """Analyze a single social media post."""
        
        self.eval()
        inputs = self.bert_model.tokenize(text)
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
        sentiment_idx = outputs['sentiment'].item()
        manipulation_idx = outputs['manipulation_type'].item()
        
        return {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'sentiment': self.SENTIMENT_LABELS[sentiment_idx],
            'sentiment_confidence': outputs['sentiment_probs'][0, sentiment_idx].item(),
            'manipulation_type': self.MANIPULATION_TYPES[manipulation_idx],
            'manipulation_confidence': outputs['manipulation_probs'][0, manipulation_idx].item(),
            'bot_probability': outputs['bot_probability'].item(),
            'fraud_probability': outputs['probabilities'][0, 1].item()
        }


if __name__ == '__main__':
    # Example usage
    classifier = BERTFraudClassifier(
        model_name='bert-base-uncased',
        max_length=512,
        num_classes=2
    )
    
    # Test classification
    test_texts = [
        "URGENT: New token launching 1000x gains guaranteed! Don't miss out!",
        "Bitcoin's hash rate reached a new all-time high today.",
        "Send 1 ETH to receive 10 ETH back - limited time offer from Vitalik!"
    ]
    
    results = classifier.classify_text(test_texts)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Suspicious: {result['is_suspicious']}")
        print(f"Probability: {result['fraud_probability']:.4f}")
        print()
