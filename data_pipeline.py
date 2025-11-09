"""
Data pipeline for text preprocessing, tokenization, and dataset preparation.
Preserves character offsets for token-level attribution visualization.
"""

import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Preprocesses text while preserving character offsets."""
    
    def __init__(self, lowercase: bool = True, remove_urls: bool = True, 
                 remove_mentions: bool = False, remove_hashtags: bool = False):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        
    def preprocess(self, text: str) -> Tuple[str, Dict[int, int]]]:
        """
        Preprocess text and return cleaned text with offset mapping.
        
        Returns:
            cleaned_text: Preprocessed text
            offset_map: Mapping from original to cleaned character positions
        """
        original_text = text
        offset_map = {}
        cleaned = text
        
        # Remove URLs
        if self.remove_urls:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            cleaned = re.sub(url_pattern, '', cleaned)
        
        # Remove mentions
        if self.remove_mentions:
            cleaned = re.sub(r'@\w+', '', cleaned)
        
        # Remove hashtags (but keep the word)
        if self.remove_hashtags:
            cleaned = re.sub(r'#(\w+)', r'\1', cleaned)
        
        # Lowercase
        if self.lowercase:
            cleaned = cleaned.lower()
        
        # Build offset mapping (simplified - maps cleaned positions to original)
        # For simplicity, we'll use character-level alignment
        orig_idx = 0
        clean_idx = 0
        while orig_idx < len(original_text) and clean_idx < len(cleaned):
            if original_text[orig_idx].lower() == cleaned[clean_idx]:
                offset_map[clean_idx] = orig_idx
                orig_idx += 1
                clean_idx += 1
            else:
                orig_idx += 1
        
        return cleaned, offset_map


class SentimentDataset:
    """Dataset class for sentiment analysis with offset preservation."""
    
    def __init__(self, tokenizer_name: str = "roberta-base", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.preprocessor = TextPreprocessor()
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def tokenize_with_offsets(self, text: str) -> Dict:
        """
        Tokenize text and preserve character offsets for each token.
        
        Returns:
            Dictionary with tokens, input_ids, attention_mask, and token_offsets
        """
        # Preprocess text
        cleaned_text, offset_map = self.preprocessor.preprocess(text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Get token offsets
        token_offsets = encoding['offset_mapping'][0].tolist()
        
        # Map offsets back to original text
        original_offsets = []
        for start, end in token_offsets:
            if start == 0 and end == 0:  # Special tokens or padding
                original_offsets.append((0, 0))
            else:
                # Map cleaned offsets to original offsets
                orig_start = offset_map.get(start, start)
                orig_end = offset_map.get(end, end) if end < len(offset_map) else len(text)
                original_offsets.append((orig_start, orig_end))
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_offsets': original_offsets,
            'tokens': self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]),
            'original_text': text
        }
    
    def prepare_dataset(self, texts: List[str], labels: List[str], 
                       test_size: float = 0.2, val_size: float = 0.1,
                       balance_classes: bool = True, random_state: int = 42) -> Dict:
        """
        Prepare train/val/test splits with optional class balancing.
        
        Returns:
            Dictionary with train/val/test splits and class weights
        """
        # Convert labels to numeric
        numeric_labels = [self.label_map.get(label.lower(), 1) for label in labels]
        
        # Create DataFrame
        df = pd.DataFrame({'text': texts, 'label': numeric_labels})
        
        # Balance classes if requested
        if balance_classes:
            # Get class distribution
            class_counts = df['label'].value_counts()
            min_class_size = class_counts.min()
            
            # Sample equally from each class
            balanced_dfs = []
            for label in df['label'].unique():
                label_df = df[df['label'] == label]
                if len(label_df) > min_class_size:
                    label_df = label_df.sample(n=min_class_size, random_state=random_state)
                balanced_dfs.append(label_df)
            
            df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
        
        # Split into train/val/test
        train_df, temp_df = train_test_split(
            df, test_size=(test_size + val_size), random_state=random_state, stratify=df['label']
        )
        
        val_size_adjusted = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size_adjusted), random_state=random_state, stratify=temp_df['label']
        )
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label']
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Tokenize all splits
        train_data = [self.tokenize_with_offsets(text) for text in train_df['text'].tolist()]
        val_data = [self.tokenize_with_offsets(text) for text in val_df['text'].tolist()]
        test_data = [self.tokenize_with_offsets(text) for text in test_df['text'].tolist()]
        
        return {
            'train': {
                'data': train_data,
                'labels': train_df['label'].tolist(),
                'texts': train_df['text'].tolist()
            },
            'val': {
                'data': val_data,
                'labels': val_df['label'].tolist(),
                'texts': val_df['text'].tolist()
            },
            'test': {
                'data': test_data,
                'labels': test_df['label'].tolist(),
                'texts': test_df['text'].tolist()
            },
            'class_weights': class_weight_dict
        }
    
    def create_attribution_eval_set(self, texts: List[str], labels: List[str], 
                                   n_examples: int = 20) -> List[Dict]:
        """
        Create a small set of annotated examples for attribution evaluation.
        Selects diverse examples across classes and lengths.
        """
        df = pd.DataFrame({'text': texts, 'label': labels})
        df['length'] = df['text'].str.len()
        
        # Sample diverse examples
        eval_examples = []
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            n_per_class = n_examples // len(df['label'].unique())
            
            # Sample from different length quartiles
            quartiles = np.percentile(label_df['length'], [25, 50, 75])
            for q in [0, 25, 50, 75, 100]:
                if q == 0:
                    subset = label_df[label_df['length'] <= quartiles[0]]
                elif q == 25:
                    subset = label_df[(label_df['length'] > quartiles[0]) & (label_df['length'] <= quartiles[1])]
                elif q == 50:
                    subset = label_df[(label_df['length'] > quartiles[1]) & (label_df['length'] <= quartiles[2])]
                else:
                    subset = label_df[label_df['length'] > quartiles[2]]
                
                if len(subset) > 0:
                    sample = subset.sample(n=min(1, len(subset)), random_state=42)
                    eval_examples.append({
                        'text': sample.iloc[0]['text'],
                        'label': sample.iloc[0]['label'],
                        'expected_tokens': []  # Can be manually annotated
                    })
        
        return eval_examples[:n_examples]


def load_sample_data() -> Tuple[List[str], List[str]]:
    """
    Load sample sentiment data. In production, replace with actual dataset loading.
    Creates a synthetic dataset for demonstration.
    """
    # Sample texts with labels
    sample_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the worst service I've ever experienced. Terrible!",
        "The movie was okay, nothing special but not bad either.",
        "Fantastic quality and great customer support. Highly recommended!",
        "Poor quality, broke after one day. Very disappointed.",
        "It's fine, does what it's supposed to do.",
        "Outstanding performance! Exceeded all my expectations.",
        "Waste of money. Don't buy this product.",
        "Good value for money, satisfied with the purchase.",
        "Horrible experience, would not recommend to anyone.",
        "The food was delicious and the service was excellent!",
        "Terrible food, cold and tasteless. Very disappointed.",
        "Average meal, nothing to write home about.",
        "Best restaurant in town! Amazing atmosphere and food.",
        "Overpriced and underwhelming. Not worth it.",
        "Decent place, could be better but acceptable.",
        "Incredible experience! Will definitely come back.",
        "Awful service, rude staff, and poor quality.",
        "Nice place, friendly staff, good food.",
        "The worst experience ever. Avoid at all costs.",
    ]
    
    sample_labels = [
        "positive", "negative", "neutral", "positive", "negative",
        "neutral", "positive", "negative", "positive", "negative",
        "positive", "negative", "neutral", "positive", "negative",
        "neutral", "positive", "negative", "positive", "negative"
    ]
    
    return sample_texts, sample_labels


if __name__ == "__main__":
    # Example usage
    dataset = SentimentDataset()
    texts, labels = load_sample_data()
    
    splits = dataset.prepare_dataset(texts, labels, balance_classes=True)
    print(f"Train: {len(splits['train']['labels'])} examples")
    print(f"Val: {len(splits['val']['labels'])} examples")
    print(f"Test: {len(splits['test']['labels'])} examples")
    print(f"Class weights: {splits['class_weights']}")
    
    # Save processed data
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    with open('data/processed/splits.json', 'w') as f:
        json.dump({
            'train_texts': splits['train']['texts'],
            'train_labels': splits['train']['labels'],
            'val_texts': splits['val']['texts'],
            'val_labels': splits['val']['labels'],
            'test_texts': splits['test']['texts'],
            'test_labels': splits['test']['labels'],
        }, f, indent=2)
    
    # Create attribution eval set
    eval_set = dataset.create_attribution_eval_set(texts, labels, n_examples=10)
    with open('data/annotations/attribution_eval.json', 'w') as f:
        json.dump(eval_set, f, indent=2)
    
    print("\nAttribution evaluation set created!")

