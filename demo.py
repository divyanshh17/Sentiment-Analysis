"""
Demo script showing example flows for finance and healthcare scenarios.
"""

import argparse
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import InterpretableSentimentClassifier, SentimentModelWrapper
from src.explainability import (
    IntegratedGradientsExplainer, RationaleGenerator, CounterfactualGenerator
)
from transformers import AutoTokenizer
import torch


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_finance():
    """Demo with finance-related examples."""
    print_section("FINANCE SCENARIO DEMO")
    
    finance_examples = [
        "The stock market crashed today, causing massive losses for investors.",
        "Our quarterly earnings exceeded expectations, showing strong growth potential.",
        "The company's financial performance was stable, meeting analyst predictions.",
        "Bankruptcy rumors are spreading, causing panic among shareholders.",
        "The merger announcement boosted investor confidence significantly."
    ]
    
    run_demo_examples(finance_examples, "Finance")


def demo_healthcare():
    """Demo with healthcare-related examples."""
    print_section("HEALTHCARE SCENARIO DEMO")
    
    healthcare_examples = [
        "The new treatment showed remarkable results in clinical trials.",
        "Patient safety concerns have been raised about this medication.",
        "The hospital's response time was adequate but could be improved.",
        "The medical breakthrough offers hope for millions of patients worldwide.",
        "The side effects of this drug are severe and concerning."
    ]
    
    run_demo_examples(healthcare_examples, "Healthcare")


def run_demo_examples(examples, scenario_name):
    """Run demo on a list of examples."""
    # Load model
    print(f"Loading model for {scenario_name} scenario...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'roberta-base'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = InterpretableSentimentClassifier(
        model_name=model_name,
        num_labels=3
    )
    model.to(device)
    model.eval()
    
    wrapper = SentimentModelWrapper(model, tokenizer, device=device)
    ig_explainer = IntegratedGradientsExplainer(model, tokenizer, device=device)
    rationale_gen = RationaleGenerator()
    counterfactual_gen = CounterfactualGenerator(model, tokenizer, device=device)
    
    print(f"Model loaded on {device}\n")
    
    # Process each example
    for i, text in enumerate(examples, 1):
        print(f"\n{'─' * 80}")
        print(f"Example {i}: {scenario_name}")
        print(f"{'─' * 80}")
        print(f"Text: \"{text}\"\n")
        
        # Get prediction
        prediction = wrapper.predict_text(text)
        print(f"Predicted Label: {prediction['label'].upper()}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Probabilities:")
        for label, prob in prediction['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        
        # Get attributions
        print("\nComputing attributions...")
        attributions = ig_explainer.get_attributions(text)
        
        # Show top contributing tokens
        token_scores = list(zip(attributions['tokens'], attributions['attributions']))
        token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\nTop Contributing Tokens:")
        for token, score in token_scores[:5]:
            direction = "↑" if score > 0 else "↓"
            print(f"  {direction} '{token}': {score:+.3f}")
        
        # Generate rationale
        rationale = rationale_gen.generate(
            prediction['label'],
            prediction['confidence'],
            attributions,
            prediction['probabilities']
        )
        print(f"\nRationale: {rationale}")
        
        # Generate counterfactuals
        print("\nGenerating counterfactuals...")
        counterfactuals = counterfactual_gen.generate(
            text, prediction['label'], attributions, n_candidates=2
        )
        
        if counterfactuals:
            print("Counterfactual Examples:")
            for j, cf in enumerate(counterfactuals, 1):
                print(f"  {j}. \"{cf['edit']}\"")
                print(f"     → New Label: {cf['new_label']}")
                print(f"     → Confidence Change: {cf['delta_confidence']:+.2%}")
        else:
            print("  No counterfactuals generated.")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='Demo script for XSA')
    parser.add_argument('--scenario', type=str, choices=['finance', 'healthcare', 'all'],
                       default='all', help='Scenario to demo')
    
    args = parser.parse_args()
    
    if args.scenario in ['finance', 'all']:
        demo_finance()
    
    if args.scenario in ['healthcare', 'all']:
        demo_healthcare()
    
    print_section("DEMO COMPLETE")
    print("For interactive use, start the web UI with: python src/api.py")
    print("Then visit http://localhost:5000 in your browser.")


if __name__ == '__main__':
    main()

