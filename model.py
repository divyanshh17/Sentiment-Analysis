"""
Model architecture for sentiment classification with interpretable components.
Supports BERT/RoBERTa fine-tuning with optional interpretable head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple
import numpy as np


class InterpretableSentimentClassifier(nn.Module):
    """
    Sentiment classifier with interpretable components.
    Base architecture: BERT/RoBERTa with classification head.
    """
    
    def __init__(self, model_name: str = "roberta-base", num_labels: int = 3,
                 dropout: float = 0.1, use_interpretable_head: bool = False):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_interpretable_head = use_interpretable_head
        
        # Load pre-trained model
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        if use_interpretable_head:
            # Interpretable head: linear layer with attention-like mechanism
            self.attention = nn.Linear(self.hidden_size, 1)
            self.classifier = nn.Linear(self.hidden_size, num_labels)
        else:
            # Standard classification head
            self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        if self.use_interpretable_head:
            nn.init.xavier_uniform_(self.attention.weight)
            nn.init.zeros_(self.attention.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            return_embeddings: Whether to return token embeddings for attribution
        
        Returns:
            Dictionary with logits, probabilities, and optionally embeddings
        """
        # Get BERT/RoBERTa outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # RoBERTa doesn't have pooler_output, so we use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        else:
            # Use CLS token (first token) for pooling
            pooled_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        if self.use_interpretable_head:
            # Compute attention weights over tokens
            attention_weights = self.attention(sequence_output)  # [batch_size, seq_len, 1]
            attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]
            
            # Mask out padding tokens
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0, float('-inf')
            )
            attention_weights = torch.softmax(attention_weights, dim=-1)
            
            # Weighted sum of token embeddings
            weighted_embeddings = torch.sum(
                attention_weights.unsqueeze(-1) * sequence_output, dim=1
            )  # [batch_size, hidden_size]
            
            # Classification
            logits = self.classifier(weighted_embeddings)
            
            result = {
                'logits': logits,
                'probabilities': torch.softmax(logits, dim=-1),
                'attention_weights': attention_weights
            }
        else:
            # Standard classification using pooled output
            logits = self.classifier(pooled_output)
            
            result = {
                'logits': logits,
                'probabilities': torch.softmax(logits, dim=-1)
            }
        
        # Return embeddings for attribution if requested
        if return_embeddings:
            result['embeddings'] = sequence_output
            result['pooled_output'] = pooled_output
        
        return result
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        """
        Make prediction with confidence scores.
        
        Returns:
            Dictionary with label, confidence, and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = outputs['probabilities']
            confidence, predicted = torch.max(probs, dim=-1)
            
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            
            return {
                'label': label_map[predicted.item()],
                'confidence': confidence.item(),
                'probabilities': {
                    label_map[i]: probs[0][i].item() 
                    for i in range(self.num_labels)
                }
            }
    
    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'use_interpretable_head': self.use_interpretable_head
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            model_name=checkpoint['model_name'],
            num_labels=checkpoint['num_labels'],
            use_interpretable_head=checkpoint.get('use_interpretable_head', False)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


class SentimentModelWrapper:
    """
    Wrapper class for model that handles tokenization and prediction.
    """
    
    def __init__(self, model: InterpretableSentimentClassifier, tokenizer, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict_text(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.
        
        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        result = self.model.predict(input_ids, attention_mask)
        
        # Add token information
        result['tokens'] = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        result['input_ids'] = input_ids
        result['attention_mask'] = attention_mask
        
        return result


if __name__ == "__main__":
    # Test model initialization
    model = InterpretableSentimentClassifier(
        model_name="roberta-base",
        num_labels=3,
        use_interpretable_head=False
    )
    
    print(f"Model initialized: {model.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    test_text = "I love this product!"
    encoding = tokenizer(test_text, return_tensors='pt', padding='max_length', 
                        max_length=128, truncation=True)
    
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
        print(f"\nLogits shape: {outputs['logits'].shape}")
        print(f"Probabilities: {outputs['probabilities']}")
        
        # Test prediction
        wrapper = SentimentModelWrapper(model, tokenizer)
        prediction = wrapper.predict_text(test_text)
        print(f"\nPrediction: {prediction}")

