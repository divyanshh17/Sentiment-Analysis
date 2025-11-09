"""
Flask API for Explainable Sentiment Analysis.
Provides endpoints for prediction, explanation, and feedback.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import AutoTokenizer
import json
from datetime import datetime
from typing import Dict, Optional
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import InterpretableSentimentClassifier, SentimentModelWrapper
from src.explainability import (
    IntegratedGradientsExplainer, SHAPLikeExplainer,
    RationaleGenerator, CounterfactualGenerator,
    compute_explainability_quality
)

# Get the directory of this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'frontend', 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'frontend', 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

# Global variables
MODEL = None
TOKENIZER = None
MODEL_WRAPPER = None
IG_EXPLAINER = None
SHAP_EXPLAINER = None
RATIONALE_GENERATOR = None
COUNTERFACTUAL_GENERATOR = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path: Optional[str] = None, model_name: str = "roberta-base"):
    """Load the sentiment analysis model."""
    global MODEL, TOKENIZER, MODEL_WRAPPER, IG_EXPLAINER, SHAP_EXPLAINER
    global RATIONALE_GENERATOR, COUNTERFACTUAL_GENERATOR
    
    print(f"Loading model: {model_name} on device: {DEVICE}")
    
    # Load tokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    
    # Load or initialize model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        MODEL = InterpretableSentimentClassifier.load(model_path, device=DEVICE)
    else:
        print(f"Initializing new model: {model_name}")
        MODEL = InterpretableSentimentClassifier(
            model_name=model_name,
            num_labels=3,
            use_interpretable_head=False
        )
        MODEL.to(DEVICE)
    
    # Initialize wrappers and explainers
    MODEL_WRAPPER = SentimentModelWrapper(MODEL, TOKENIZER, device=DEVICE)
    IG_EXPLAINER = IntegratedGradientsExplainer(MODEL, TOKENIZER, device=DEVICE)
    SHAP_EXPLAINER = SHAPLikeExplainer(MODEL, TOKENIZER, device=DEVICE, n_samples=50)
    RATIONALE_GENERATOR = RationaleGenerator()
    COUNTERFACTUAL_GENERATOR = CounterfactualGenerator(MODEL, TOKENIZER, device=DEVICE)
    
    print("Model loaded successfully!")


@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': DEVICE
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint with explanations.
    
    Request body:
    {
        "text": "I love this product!",
        "explanation_method": "ig",  # "ig" or "shap"
        "include_counterfactuals": true,
        "target_class": null  # Optional: specify class for attribution
    }
    
    Response:
    {
        "label": "positive",
        "confidence": 0.95,
        "attributions": [
            {"token": "love", "score": 0.45, "span": [2, 6]}
        ],
        "rationale": "...",
        "explanation_method": "ig",
        "counterfactuals": [...],
        "provenance": {...},
        "explainability_quality": "high"
    }
    """
    if MODEL is None or TOKENIZER is None:
        return jsonify({'error': 'Model not loaded. Please wait for initialization.'}), 503
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON in request body'}), 400
        
        text = data.get('text', '').strip()
        
        # Validate text input
        if not text:
            return jsonify({'error': 'Text is required and cannot be empty'}), 400
        
        if len(text) > 2000:
            return jsonify({'error': 'Text is too long. Maximum 2000 characters allowed.'}), 400
        
        # Get prediction with error handling
        try:
            prediction = MODEL_WRAPPER.predict_text(text)
            label = prediction.get('label', 'neutral')
            confidence = prediction.get('confidence', 0.5)
            probabilities = prediction.get('probabilities', {})
        except Exception as e:
            return jsonify({'error': f'Failed to generate prediction: {str(e)}'}), 500
        
        # Validate explanation method
        explanation_method = data.get('explanation_method', 'ig').lower()
        if explanation_method not in ['ig', 'shap']:
            explanation_method = 'ig'  # Default fallback
        
        target_class = data.get('target_class', None)
        
        # Map label to class index
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        if target_class is None:
            target_class = label_map.get(label, 1)
        elif isinstance(target_class, str):
            target_class = label_map.get(target_class.lower(), 1)
        
        # Get attributions with error handling
        try:
            if explanation_method == 'shap':
                attributions = SHAP_EXPLAINER.get_attributions(text, target_class=target_class)
            else:  # Default to IG
                attributions = IG_EXPLAINER.get_attributions(text, target_class=target_class)
            
            # Validate attributions
            if not attributions or 'tokens' not in attributions:
                attributions = {'tokens': [], 'attributions': [], 'offsets': []}
            
            # Ensure all lists have same length
            min_len = min(len(attributions.get('tokens', [])), 
                         len(attributions.get('attributions', [])),
                         len(attributions.get('offsets', [])))
            
            if min_len == 0:
                # Fallback: create empty attributions
                formatted_attributions = []
            else:
                formatted_attributions = [
                    {
                        'token': str(token),
                        'score': float(score),
                        'span': list(offset) if isinstance(offset, (tuple, list)) and len(offset) == 2 else [0, 0]
                    }
                    for token, score, offset in zip(
                        attributions['tokens'][:min_len],
                        attributions['attributions'][:min_len],
                        attributions['offsets'][:min_len]
                    )
                ]
        except Exception as e:
            print(f"Error computing attributions: {str(e)}")
            # Continue with empty attributions rather than failing
            formatted_attributions = []
            attributions = {'tokens': [], 'attributions': [], 'offsets': []}
        
        # Generate rationale with error handling
        try:
            rationale = RATIONALE_GENERATOR.generate(
                label, confidence, attributions, probabilities
            )
        except Exception as e:
            print(f"Error generating rationale: {str(e)}")
            rationale = f"The sentiment is classified as {label} with {confidence:.2%} confidence."
        
        # Generate counterfactuals if requested (with error handling)
        counterfactuals = []
        if data.get('include_counterfactuals', False):
            try:
                counterfactuals_raw = COUNTERFACTUAL_GENERATOR.generate(
                    text, label, attributions, n_candidates=3
                )
                if counterfactuals_raw:
                    counterfactuals = [
                        {
                            'edit': str(cf.get('edit', '')),
                            'new_label': str(cf.get('new_label', label)),
                            'delta_confidence': float(cf.get('delta_confidence', 0.0))
                        }
                        for cf in counterfactuals_raw
                        if isinstance(cf, dict) and 'edit' in cf
                    ]
            except Exception as e:
                print(f"Error generating counterfactuals: {str(e)}")
                # Continue without counterfactuals rather than failing
                counterfactuals = []
        
        # Compute explainability quality with error handling
        try:
            explainability_quality = compute_explainability_quality(
                attributions, MODEL, TOKENIZER, text, DEVICE
            )
        except Exception as e:
            print(f"Error computing explainability quality: {str(e)}")
            explainability_quality = "medium"  # Default fallback
        
        # Build response
        response = {
            'label': str(label),
            'confidence': float(confidence),
            'attributions': formatted_attributions,
            'rationale': str(rationale),
            'explanation_method': explanation_method,
            'counterfactuals': counterfactuals,
            'provenance': {
                'model': str(MODEL.model_name),
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'device': str(DEVICE)
            },
            'explainability_quality': str(explainability_quality)
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = str(e)
        print(f"Error in /api/predict: {error_msg}")
        print(f"Traceback: {error_trace}")
        
        # Return user-friendly error message
        return jsonify({
            'error': 'Unable to analyze text. Please try again later.',
            'details': error_msg if len(error_msg) < 100 else 'Internal server error'
        }), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """
    Collect user feedback on explanations.
    
    Request body:
    {
        "text": "...",
        "prediction": {...},
        "feedback_type": "correct" | "incorrect" | "explanation_helpful" | "explanation_confusing",
        "comments": "..."
    }
    """
    try:
        data = request.get_json()
        
        # In production, save feedback to database
        # For now, just log it
        feedback_data = {
            'text': data.get('text', ''),
            'prediction': data.get('prediction', {}),
            'feedback_type': data.get('feedback_type', ''),
            'comments': data.get('comments', ''),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Save to file (in production, use database)
        feedback_file = 'data/feedback.jsonl'
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')
        
        return jsonify({'status': 'feedback_received'})
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /api/predict: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({'error': str(e), 'traceback': error_trace}), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """
    Get explanations for a given text with different methods.
    
    Request body:
    {
        "text": "...",
        "methods": ["ig", "shap"]  # List of methods to compute
    }
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        methods = data.get('methods', ['ig'])
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Get prediction
        prediction = MODEL_WRAPPER.predict_text(text)
        label = prediction['label']
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        target_class = label_map.get(label, 1)
        
        explanations = {}
        
        for method in methods:
            if method == 'ig':
                attributions = IG_EXPLAINER.get_attributions(text, target_class=target_class)
            elif method == 'shap':
                attributions = SHAP_EXPLAINER.get_attributions(text, target_class=target_class)
            else:
                continue
            
            explanations[method] = {
                'tokens': attributions['tokens'],
                'attributions': attributions['attributions'],
                'offsets': attributions['offsets']
            }
        
        return jsonify({
            'label': label,
            'confidence': prediction['confidence'],
            'explanations': explanations
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /api/predict: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({'error': str(e), 'traceback': error_trace}), 500


if __name__ == '__main__':
    # Load model on startup
    model_path = os.environ.get('MODEL_PATH', None)
    model_name = os.environ.get('MODEL_NAME', 'roberta-base')
    
    load_model(model_path, model_name)
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

