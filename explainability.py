"""
Explainability module for sentiment analysis.
Implements Integrated Gradients, SHAP-like approximations, rationale generation, and counterfactuals.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from captum.attr import IntegratedGradients, ShapleyValueSampling
try:
    from captum.attr._utils.common import _construct_default_reference
except ImportError:
    # Fallback for older captum versions
    _construct_default_reference = None
from captum.attr import visualization as viz
import random
import re
from collections import Counter


class AttributionExplainer:
    """Base class for attribution methods."""
    
    def __init__(self, model, tokenizer, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def get_attributions(self, text: str, target_class: Optional[int] = None) -> Dict:
        """Get token-level attributions. To be implemented by subclasses."""
        raise NotImplementedError


class IntegratedGradientsExplainer(AttributionExplainer):
    """Integrated Gradients for token-level attribution."""
    
    def __init__(self, model, tokenizer, device: str = 'cpu', n_steps: int = 50):
        super().__init__(model, tokenizer, device)
        self.n_steps = n_steps
        self.ig = IntegratedGradients(self._forward_func)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
    
    def _forward_func(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Wrapper for model forward pass for Captum."""
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        attention_mask = (input_ids != pad_id).long()
        outputs = self.model(input_ids, attention_mask)
        return outputs['logits']
    
    def get_attributions(self, text: str, target_class: Optional[int] = None) -> Dict:
        """
        Compute Integrated Gradients attributions.
        
        Returns:
            Dictionary with tokens, attributions, and scores
        """
        try:
            # Validate input
            if not text or not isinstance(text, str):
                return {'tokens': [], 'attributions': [], 'offsets': [], 'method': 'integrated_gradients'}
            
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
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = outputs['probabilities']
                if target_class is None:
                    target_class = torch.argmax(probs, dim=-1).item()
            
            # Get reference (baseline) - use padding tokens
            ref_input_ids = torch.full_like(input_ids, self.pad_token_id)
            
            # Clone input_ids for attribution (IG will handle gradients internally)
            input_ids_for_attr = input_ids.clone()
            
            # Compute attributions
            try:
                # Temporarily set model to allow gradients
                was_training = self.model.training
                self.model.eval()
                
                attributions, delta = self.ig.attribute(
                    input_ids_for_attr,
                    ref_input_ids,
                    target=target_class,
                    n_steps=self.n_steps,
                    return_convergence_delta=True,
                    internal_batch_size=1  # Process one at a time to avoid memory issues
                )
                
                # Restore model state
                if was_training:
                    self.model.train()
                else:
                    self.model.eval()
            except Exception as e:
                print(f"IG attribution computation failed: {str(e)}")
                # Return empty attributions rather than failing
                return {'tokens': [], 'attributions': [], 'offsets': [], 'method': 'integrated_gradients'}
            
            # Convert to numpy
            if attributions is not None and len(attributions) > 0:
                attributions = attributions[0].cpu().detach().numpy()
            else:
                return {'tokens': [], 'attributions': [], 'offsets': [], 'method': 'integrated_gradients'}
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Get token offsets in original text
            token_offsets = self._get_token_offsets(text, tokens)
            
            # Filter out special tokens and padding
            valid_attributions = []
            valid_tokens = []
            valid_offsets = []
            
            pad_token = self.tokenizer.pad_token if self.tokenizer.pad_token else None
            cls_token = self.tokenizer.cls_token if self.tokenizer.cls_token else None
            sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token else None
            unk_token = self.tokenizer.unk_token if self.tokenizer.unk_token else None
            
            special_tokens = [t for t in [pad_token, cls_token, sep_token, unk_token] if t is not None]
            
            for i, (token, attr, offset) in enumerate(zip(tokens, attributions, token_offsets)):
                if token not in special_tokens and len(token.strip()) > 0:
                    valid_attributions.append(float(attr))
                    valid_tokens.append(token)
                    valid_offsets.append(offset)
            
            # Normalize attributions
            if len(valid_attributions) > 0:
                max_attr = max(abs(a) for a in valid_attributions)
                if max_attr > 0:
                    valid_attributions = [a / max_attr for a in valid_attributions]
            
            return {
                'tokens': valid_tokens,
                'attributions': valid_attributions,
                'offsets': valid_offsets,
                'method': 'integrated_gradients',
                'convergence_delta': float(delta.item()) if delta is not None and hasattr(delta, 'item') else None
            }
        except Exception as e:
            print(f"Error in get_attributions: {str(e)}")
            # Return empty attributions rather than raising exception
            return {'tokens': [], 'attributions': [], 'offsets': [], 'method': 'integrated_gradients'}
    
    def _get_token_offsets(self, text: str, tokens: List[str]) -> List[Tuple[int, int]]:
        """Estimate character offsets for tokens in original text."""
        offsets = []
        char_idx = 0
        
        for token in tokens:
            if token in [self.tokenizer.pad_token, self.tokenizer.cls_token, 
                        self.tokenizer.sep_token]:
                offsets.append((0, 0))
                continue
            
            # Clean token (remove ## for BERT-style tokenizers)
            clean_token = token.replace('Ġ', '').replace('▁', '')
            
            if clean_token in text[char_idx:]:
                start = text.find(clean_token, char_idx)
                end = start + len(clean_token)
                offsets.append((start, end))
                char_idx = end
            else:
                offsets.append((char_idx, char_idx))
        
        return offsets


class SHAPLikeExplainer(AttributionExplainer):
    """SHAP-like approximation using sampling over token masks."""
    
    def __init__(self, model, tokenizer, device: str = 'cpu', n_samples: int = 100):
        super().__init__(model, tokenizer, device)
        self.n_samples = n_samples
    
    def get_attributions(self, text: str, target_class: Optional[int] = None) -> Dict:
        """
        Compute SHAP-like attributions using sampling.
        
        Returns:
            Dictionary with tokens, attributions, and scores
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
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
            baseline_mask = torch.zeros_like(attention_mask)
            baseline_outputs = self.model(baseline_ids, baseline_mask)
            baseline_probs = baseline_outputs['probabilities']
            if target_class is None:
                target_class = torch.argmax(baseline_probs, dim=-1).item()
            baseline_score = baseline_probs[0, target_class].item()
        
        # Get full prediction
        with torch.no_grad():
            full_outputs = self.model(input_ids, attention_mask)
            full_probs = full_outputs['probabilities']
            full_score = full_probs[0, target_class].item()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        n_tokens = len([t for t in tokens if t != self.tokenizer.pad_token])
        
        # Sample coalitions and compute marginal contributions
        attributions = np.zeros(len(tokens))
        
        for _ in range(self.n_samples):
            # Random coalition
            coalition = torch.rand(len(tokens)) > 0.5
            
            # Create masked input
            masked_ids = input_ids.clone()
            masked_ids[0][~coalition] = self.tokenizer.pad_token_id
            masked_mask = (masked_ids != self.tokenizer.pad_token_id).long()
            
            # Get prediction with coalition
            with torch.no_grad():
                outputs = self.model(masked_ids, masked_mask)
                probs = outputs['probabilities']
                score = probs[0, target_class].item()
            
            # Compute marginal contribution for each token
            for i in range(len(tokens)):
                if tokens[i] == self.tokenizer.pad_token:
                    continue
                
                # Token in coalition
                if coalition[i]:
                    # Remove token and recompute
                    temp_coalition = coalition.clone()
                    temp_coalition[i] = False
                    temp_ids = input_ids.clone()
                    temp_ids[0][~temp_coalition] = self.tokenizer.pad_token_id
                    temp_mask = (temp_ids != self.tokenizer.pad_token_id).long()
                    
                    with torch.no_grad():
                        temp_outputs = self.model(temp_ids, temp_mask)
                        temp_probs = temp_outputs['probabilities']
                        temp_score = temp_probs[0, target_class].item()
                    
                    marginal = score - temp_score
                else:
                    # Token not in coalition
                    # Add token and recompute
                    temp_coalition = coalition.clone()
                    temp_coalition[i] = True
                    temp_ids = input_ids.clone()
                    temp_ids[0][~temp_coalition] = self.tokenizer.pad_token_id
                    temp_mask = (temp_ids != self.tokenizer.pad_token_id).long()
                    
                    with torch.no_grad():
                        temp_outputs = self.model(temp_ids, temp_mask)
                        temp_probs = temp_outputs['probabilities']
                        temp_score = temp_probs[0, target_class].item()
                    
                    marginal = temp_score - score
                
                attributions[i] += marginal / self.n_samples
        
        # Normalize
        max_attr = max(abs(a) for a in attributions)
        if max_attr > 0:
            attributions = attributions / max_attr
        
        # Get token offsets
        token_offsets = self._get_token_offsets(text, tokens)
        
        # Filter valid tokens
        valid_tokens = []
        valid_attributions = []
        valid_offsets = []
        
        for token, attr, offset in zip(tokens, attributions, token_offsets):
            if token not in [self.tokenizer.pad_token, self.tokenizer.cls_token,
                           self.tokenizer.sep_token, self.tokenizer.unk_token]:
                valid_tokens.append(token)
                valid_attributions.append(float(attr))
                valid_offsets.append(offset)
        
        return {
            'tokens': valid_tokens,
            'attributions': valid_attributions,
            'offsets': valid_offsets,
            'method': 'shap_like',
            'n_samples': self.n_samples
        }
    
    def _get_token_offsets(self, text: str, tokens: List[str]) -> List[Tuple[int, int]]:
        """Estimate character offsets for tokens in original text."""
        offsets = []
        char_idx = 0
        
        for token in tokens:
            if token in [self.tokenizer.pad_token, self.tokenizer.cls_token,
                        self.tokenizer.sep_token]:
                offsets.append((0, 0))
                continue
            
            clean_token = token.replace('Ġ', '').replace('▁', '')
            
            if clean_token in text[char_idx:]:
                start = text.find(clean_token, char_idx)
                end = start + len(clean_token)
                offsets.append((start, end))
                char_idx = end
            else:
                offsets.append((char_idx, char_idx))
        
        return offsets


class RationaleGenerator:
    """Generate human-readable rationales from model predictions and attributions."""
    
    def __init__(self):
        # Sentiment word lists (can be expanded)
        self.positive_words = ['love', 'great', 'excellent', 'amazing', 'fantastic', 
                              'wonderful', 'perfect', 'good', 'best', 'awesome']
        self.negative_words = ['hate', 'terrible', 'awful', 'bad', 'worst', 
                              'horrible', 'disappointed', 'poor', 'waste']
    
    def generate(self, label: str, confidence: float, attributions: Dict,
                probabilities: Dict, top_k: int = 5) -> str:
        """
        Generate natural language rationale.
        
        Args:
            label: Predicted label
            confidence: Confidence score
            attributions: Attribution dictionary with tokens and scores
            probabilities: Probability distribution
            top_k: Number of top tokens to highlight
        
        Returns:
            Human-readable rationale string
        """
        tokens = attributions['tokens']
        scores = attributions['attributions']
        
        # Get top positive and negative contributing tokens
        token_scores = list(zip(tokens, scores))
        token_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_positive = [t for t, s in token_scores[:top_k] if s > 0]
        top_negative = [t for t, s in token_scores[:top_k] if s < 0]
        
        # Build rationale
        rationale_parts = []
        
        # Main summary
        conf_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        rationale_parts.append(
            f"The sentiment is classified as {label} with {conf_level} confidence ({confidence:.2%})."
        )
        
        # Contributing factors
        if top_positive:
            pos_tokens_str = ", ".join([f"'{t}'" for t in top_positive[:3]])
            rationale_parts.append(
                f"Positive contributions come from words like {pos_tokens_str}."
            )
        
        if top_negative:
            neg_tokens_str = ", ".join([f"'{t}'" for t in top_negative[:3]])
            rationale_parts.append(
                f"Negative contributions come from words like {neg_tokens_str}."
            )
        
        # Probability distribution
        other_probs = {k: v for k, v in probabilities.items() if k != label}
        if other_probs:
            max_other = max(other_probs.items(), key=lambda x: x[1])
            rationale_parts.append(
                f"The model also considered {max_other[0]} ({max_other[1]:.2%} probability)."
            )
        
        return " ".join(rationale_parts)


class CounterfactualGenerator:
    """Generate counterfactual examples that would change the prediction."""
    
    def __init__(self, model, tokenizer, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Word replacement dictionaries
        self.positive_replacements = {
            'hate': 'love', 'terrible': 'great', 'awful': 'wonderful',
            'bad': 'good', 'worst': 'best', 'horrible': 'excellent',
            'disappointed': 'satisfied', 'poor': 'excellent', 'waste': 'valuable'
        }
        self.negative_replacements = {v: k for k, v in self.positive_replacements.items()}
    
    def generate(self, text: str, current_label: str, attributions: Dict,
                n_candidates: int = 3) -> List[Dict]:
        """
        Generate counterfactual edits.
        
        Returns:
            List of counterfactual dictionaries with edits, new labels, and confidence deltas
        """
        counterfactuals = []
        
        try:
            # Validate inputs
            if not text or not isinstance(text, str):
                return counterfactuals
            
            if not attributions or 'tokens' not in attributions or 'attributions' not in attributions:
                return counterfactuals
            
            tokens = attributions.get('tokens', [])
            scores = attributions.get('attributions', [])
            
            if not tokens or not scores:
                return counterfactuals
            
            # Get top contributing tokens
            token_scores = list(zip(tokens, scores))
            token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get original prediction for comparison
            try:
                original_pred = self._predict_text(text)
                original_conf = original_pred.get('confidence', 0.5)
            except Exception as e:
                print(f"Error getting original prediction: {str(e)}")
                original_conf = 0.5
            
            # Strategy 1: Replace top contributing tokens
            for token, score in token_scores[:n_candidates]:
                try:
                    if current_label == "positive" and score > 0:
                        # Try replacing with negative word
                        replacements = self.negative_replacements
                    elif current_label == "negative" and score < 0:
                        # Try replacing with positive word
                        replacements = self.positive_replacements
                    else:
                        continue
                    
                    # Find replacement
                    clean_token = token.lower().replace('Ġ', '').replace('▁', '').strip()
                    if not clean_token or clean_token not in replacements:
                        continue
                    
                    new_token = replacements[clean_token]
                    edited_text = text.replace(clean_token, new_token, 1)
                    
                    # Skip if no change
                    if edited_text == text:
                        continue
                    
                    # Get new prediction
                    try:
                        new_pred = self._predict_text(edited_text)
                        new_label = new_pred.get('label', current_label)
                        new_conf = new_pred.get('confidence', original_conf)
                        delta_conf = new_conf - original_conf
                        
                        if new_label != current_label:
                            counterfactuals.append({
                                'edit': edited_text,
                                'original_text': text,
                                'changed_token': clean_token,
                                'new_token': new_token,
                                'new_label': new_label,
                                'delta_confidence': delta_conf,
                                'new_confidence': new_conf
                            })
                    except Exception as e:
                        print(f"Error predicting counterfactual: {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"Error processing token {token}: {str(e)}")
                    continue
            
            # Strategy 2: Add negation or intensifiers
            if len(counterfactuals) < n_candidates:
                try:
                    if current_label == "positive":
                        # Try adding negation
                        edited_text = f"not {text}"
                        new_pred = self._predict_text(edited_text)
                        new_label = new_pred.get('label', current_label)
                        new_conf = new_pred.get('confidence', original_conf)
                        
                        if new_label != current_label:
                            counterfactuals.append({
                                'edit': edited_text,
                                'original_text': text,
                                'changed_token': None,
                                'new_token': 'not',
                                'new_label': new_label,
                                'delta_confidence': new_conf - original_conf,
                                'new_confidence': new_conf
                            })
                except Exception as e:
                    print(f"Error in negation strategy: {str(e)}")
                    pass
            
        except Exception as e:
            print(f"Error in counterfactual generation: {str(e)}")
            # Return empty list rather than raising exception
        
        return counterfactuals[:n_candidates]
    
    def _predict_text(self, text: str) -> Dict:
        """Helper to get prediction for text."""
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = outputs['probabilities']
            confidence, predicted = torch.max(probs, dim=-1)
            
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            
            return {
                'label': label_map[predicted.item()],
                'confidence': confidence.item()
            }


def compute_explainability_quality(attributions: Dict, model, tokenizer, 
                                   text: str, device: str = 'cpu') -> str:
    """
    Compute explainability quality score based on perturbation tests.
    
    Returns:
        Quality score: 'high', 'medium', or 'low'
    """
    tokens = attributions['tokens']
    scores = attributions['attributions']
    
    # Get top positive tokens
    token_scores = list(zip(tokens, scores))
    top_positive = sorted([(t, s) for t, s in token_scores if s > 0], 
                         key=lambda x: x[1], reverse=True)[:3]
    
    if not top_positive:
        return "low"
    
    # Perturbation test: remove top positive tokens and check if confidence decreases
    original_pred = _predict_text_helper(model, tokenizer, text, device)
    original_conf = original_pred['confidence']
    
    # Create perturbed text (remove top token)
    top_token = top_positive[0][0].replace('Ġ', '').replace('▁', '')
    perturbed_text = text.replace(top_token, '', 1)
    
    perturbed_pred = _predict_text_helper(model, tokenizer, perturbed_text, device)
    perturbed_conf = perturbed_pred['confidence']
    
    # Check if confidence decreased (for positive sentiment)
    if original_pred['label'] == 'positive':
        conf_delta = original_conf - perturbed_conf
        if conf_delta > 0.1:
            return "high"
        elif conf_delta > 0.05:
            return "medium"
        else:
            return "low"
    else:
        # For negative sentiment, removing negative tokens should increase confidence
        conf_delta = perturbed_conf - original_conf
        if conf_delta > 0.1:
            return "high"
        elif conf_delta > 0.05:
            return "medium"
        else:
            return "low"


def _predict_text_helper(model, tokenizer, text: str, device: str) -> Dict:
    """Helper function for prediction."""
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = outputs['probabilities']
        confidence, predicted = torch.max(probs, dim=-1)
        
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
        return {
            'label': label_map[predicted.item()],
            'confidence': confidence.item()
        }

