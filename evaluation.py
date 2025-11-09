"""
Evaluation scripts for explainability metrics.
Computes fidelity, plausibility, and consistency metrics.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer
import argparse

from model import InterpretableSentimentClassifier, SentimentModelWrapper
from explainability import (
    IntegratedGradientsExplainer, SHAPLikeExplainer,
    compute_explainability_quality
)


class ExplainabilityEvaluator:
    """Evaluates explainability methods on various metrics."""
    
    def __init__(self, model, tokenizer, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ig_explainer = IntegratedGradientsExplainer(model, tokenizer, device)
        self.shap_explainer = SHAPLikeExplainer(model, tokenizer, device, n_samples=50)
    
    def evaluate_fidelity(self, texts: List[str], labels: List[str],
                         method: str = 'ig') -> Dict:
        """
        Evaluate fidelity using perturbation tests.
        
        Fidelity measures how well attributions reflect actual model behavior.
        We test: removing top positive tokens should decrease positive probability.
        """
        print("Evaluating fidelity...")
        
        explainer = self.ig_explainer if method == 'ig' else self.shap_explainer
        
        fidelity_scores = []
        perturbation_results = []
        
        for text, true_label in zip(texts, labels):
            # Get attributions
            attributions = explainer.get_attributions(text)
            
            if not attributions['tokens']:
                continue
            
            # Get original prediction
            original_pred = self._predict_text(text)
            original_label = original_pred['label']
            original_conf = original_pred['confidence']
            
            # Get top contributing tokens
            token_scores = list(zip(attributions['tokens'], attributions['attributions']))
            token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Perturbation test: remove top token
            if token_scores:
                top_token = token_scores[0][0].replace('Ġ', '').replace('▁', '')
                top_score = token_scores[0][1]
                
                # Remove token from text
                perturbed_text = text.replace(top_token, '', 1).strip()
                
                if perturbed_text != text:
                    perturbed_pred = self._predict_text(perturbed_text)
                    perturbed_conf = perturbed_pred['confidence']
                    
                    # Check if perturbation had expected effect
                    if original_label == 'positive' and top_score > 0:
                        # Removing positive token should decrease confidence
                        expected_delta = original_conf - perturbed_conf
                        fidelity = 1.0 if expected_delta > 0 else 0.0
                    elif original_label == 'negative' and top_score < 0:
                        # Removing negative token should decrease negative confidence
                        expected_delta = original_conf - perturbed_conf
                        fidelity = 1.0 if expected_delta > 0 else 0.0
                    else:
                        fidelity = 0.5  # Neutral case
                    
                    fidelity_scores.append(fidelity)
                    perturbation_results.append({
                        'text': text,
                        'top_token': top_token,
                        'top_score': float(top_score),
                        'original_conf': float(original_conf),
                        'perturbed_conf': float(perturbed_conf),
                        'fidelity': fidelity
                    })
        
        avg_fidelity = np.mean(fidelity_scores) if fidelity_scores else 0.0
        
        return {
            'average_fidelity': float(avg_fidelity),
            'n_tests': len(fidelity_scores),
            'perturbation_results': perturbation_results[:10]  # Sample
        }
    
    def evaluate_consistency(self, texts: List[str], n_seeds: int = 3) -> Dict:
        """
        Evaluate consistency across different model initializations.
        Note: This requires training multiple models with different seeds.
        For now, we evaluate consistency of attributions for the same model.
        """
        print("Evaluating consistency...")
        
        # Run attributions multiple times and check variance
        consistency_scores = []
        
        for text in texts[:20]:  # Sample for speed
            attributions_list = []
            
            # Get attributions multiple times (IG should be deterministic, but check)
            for _ in range(3):
                attributions = self.ig_explainer.get_attributions(text)
                if attributions['tokens']:
                    # Get top 5 tokens
                    token_scores = list(zip(attributions['tokens'], attributions['attributions']))
                    token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_tokens = [t for t, _ in token_scores[:5]]
                    attributions_list.append(set(top_tokens))
            
            # Compute overlap
            if len(attributions_list) > 1:
                overlap = len(set.intersection(*attributions_list)) / 5.0
                consistency_scores.append(overlap)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        return {
            'average_consistency': float(avg_consistency),
            'n_tests': len(consistency_scores)
        }
    
    def evaluate_plausibility(self, texts: List[str], labels: List[str]) -> Dict:
        """
        Evaluate plausibility (human evaluation stub).
        In production, this would interface with human raters.
        """
        print("Evaluating plausibility (stub)...")
        
        # For now, use heuristics:
        # - Check if top tokens are sentiment-bearing words
        sentiment_words = {
            'positive': ['love', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'perfect'],
            'negative': ['hate', 'terrible', 'awful', 'bad', 'worst', 'horrible', 'disappointed']
        }
        
        plausibility_scores = []
        
        for text, label in zip(texts, labels):
            attributions = self.ig_explainer.get_attributions(text)
            
            if not attributions['tokens']:
                continue
            
            # Get top tokens
            token_scores = list(zip(attributions['tokens'], attributions['attributions']))
            token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            top_tokens = [t.lower().replace('Ġ', '').replace('▁', '') for t, _ in token_scores[:3]]
            
            # Check if top tokens are sentiment words
            relevant_words = sentiment_words.get(label, [])
            matches = sum(1 for token in top_tokens if any(word in token for word in relevant_words))
            plausibility = matches / len(top_tokens) if top_tokens else 0.0
            
            plausibility_scores.append(plausibility)
        
        avg_plausibility = np.mean(plausibility_scores) if plausibility_scores else 0.0
        
        return {
            'average_plausibility': float(avg_plausibility),
            'n_tests': len(plausibility_scores),
            'note': 'Heuristic-based evaluation. For production, use human raters.'
        }
    
    def evaluate_accuracy(self, texts: List[str], labels: List[str]) -> Dict:
        """Evaluate model accuracy on test set."""
        print("Evaluating accuracy...")
        
        predictions = []
        true_labels = []
        
        for text, label in zip(texts, labels):
            pred = self._predict_text(text)
            predictions.append(pred['label'])
            true_labels.append(label)
        
        accuracy = accuracy_score(true_labels, predictions)
        
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return {
            'accuracy': float(accuracy),
            'classification_report': report
        }
    
    def _predict_text(self, text: str) -> Dict:
        """Helper to get prediction."""
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate explainability metrics')
    parser.add_argument('--test_data', type=str, default='data/processed/splits.json',
                       help='Path to test data JSON file')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (optional)')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                       help='Model name if not loading from path')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                       help='Output path for results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, 'r') as f:
        data = json.load(f)
    
    test_texts = data.get('test_texts', [])
    test_labels = data.get('test_labels', [])
    
    # Map numeric labels to strings
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    test_labels = [label_map.get(l, "neutral") for l in test_labels]
    
    print(f"Loaded {len(test_texts)} test examples")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.model_path and os.path.exists(args.model_path):
        model = InterpretableSentimentClassifier.load(args.model_path, device=args.device)
    else:
        model = InterpretableSentimentClassifier(
            model_name=args.model_name,
            num_labels=3
        )
        model.to(args.device)
    
    # Initialize evaluator
    evaluator = ExplainabilityEvaluator(model, tokenizer, device=args.device)
    
    # Run evaluations
    results = {}
    
    # Accuracy
    results['accuracy'] = evaluator.evaluate_accuracy(test_texts, test_labels)
    
    # Fidelity
    results['fidelity'] = evaluator.evaluate_fidelity(test_texts, test_labels, method='ig')
    
    # Consistency
    results['consistency'] = evaluator.evaluate_consistency(test_texts)
    
    # Plausibility
    results['plausibility'] = evaluator.evaluate_plausibility(test_texts, test_labels)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {args.output}")
    print(f"\nSummary:")
    print(f"  Accuracy: {results['accuracy']['accuracy']:.3f}")
    print(f"  Fidelity: {results['fidelity']['average_fidelity']:.3f}")
    print(f"  Consistency: {results['consistency']['average_consistency']:.3f}")
    print(f"  Plausibility: {results['plausibility']['average_plausibility']:.3f}")


if __name__ == '__main__':
    main()

