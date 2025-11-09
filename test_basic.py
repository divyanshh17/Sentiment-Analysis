"""
Basic tests for XSA system.
Run with: python -m pytest tests/test_basic.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer

from src.model import InterpretableSentimentClassifier, SentimentModelWrapper
from src.explainability import (
    IntegratedGradientsExplainer, RationaleGenerator, CounterfactualGenerator
)


def test_model_initialization():
    """Test that model can be initialized."""
    model = InterpretableSentimentClassifier(
        model_name="roberta-base",
        num_labels=3
    )
    assert model is not None
    assert model.num_labels == 3


def test_prediction():
    """Test basic prediction functionality."""
    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = InterpretableSentimentClassifier(
        model_name="roberta-base",
        num_labels=3
    )
    model.to(device)
    model.eval()
    
    wrapper = SentimentModelWrapper(model, tokenizer, device=device)
    
    text = "I love this product!"
    result = wrapper.predict_text(text)
    
    assert 'label' in result
    assert 'confidence' in result
    assert 'probabilities' in result
    assert result['label'] in ['positive', 'negative', 'neutral']
    assert 0 <= result['confidence'] <= 1


def test_attributions():
    """Test attribution computation."""
    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = InterpretableSentimentClassifier(
        model_name="roberta-base",
        num_labels=3
    )
    model.to(device)
    model.eval()
    
    explainer = IntegratedGradientsExplainer(model, tokenizer, device=device)
    
    text = "I love this product!"
    attributions = explainer.get_attributions(text)
    
    assert 'tokens' in attributions
    assert 'attributions' in attributions
    assert 'offsets' in attributions
    assert len(attributions['tokens']) > 0
    assert len(attributions['attributions']) == len(attributions['tokens'])


def test_rationale_generation():
    """Test rationale generation."""
    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = InterpretableSentimentClassifier(
        model_name="roberta-base",
        num_labels=3
    )
    model.to(device)
    model.eval()
    
    wrapper = SentimentModelWrapper(model, tokenizer, device=device)
    explainer = IntegratedGradientsExplainer(model, tokenizer, device=device)
    rationale_gen = RationaleGenerator()
    
    text = "I love this product!"
    prediction = wrapper.predict_text(text)
    attributions = explainer.get_attributions(text)
    
    rationale = rationale_gen.generate(
        prediction['label'],
        prediction['confidence'],
        attributions,
        prediction['probabilities']
    )
    
    assert isinstance(rationale, str)
    assert len(rationale) > 0


if __name__ == '__main__':
    print("Running basic tests...")
    
    try:
        test_model_initialization()
        print("✓ Model initialization test passed")
        
        test_prediction()
        print("✓ Prediction test passed")
        
        test_attributions()
        print("✓ Attributions test passed")
        
        test_rationale_generation()
        print("✓ Rationale generation test passed")
        
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

