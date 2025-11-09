# Explainable Sentiment Analysis (XSA)

A comprehensive sentiment analysis system with integrated explainability features, following the principles from "Explainable Sentiment Analysis: Bridging the Gap Between Performance and Transparency" (Maroo & Bhardwaj).

## Features

- **High-Performance Classifier**: Fine-tuned BERT/RoBERTa for sentiment classification
- **Multiple Explanation Methods**: Integrated Gradients (IG), SHAP-like approximations, LIME-style explanations
- **Natural Language Rationales**: Human-readable explanations of model predictions
- **Counterfactual Generation**: Propose minimal edits that would change the sentiment label
- **Interactive UI**: Token-level highlighting, confidence visualization, method selection
- **Comprehensive Evaluation**: Fidelity, plausibility, and consistency metrics

## Project Structure

```
.
├── data/                   # Data storage
│   ├── raw/               # Raw datasets
│   ├── processed/         # Processed datasets
│   └── annotations/       # Attribution evaluation examples
├── src/
│   ├── data_pipeline.py   # Data preprocessing and loading
│   ├── model.py           # Model architecture
│   ├── explainability.py  # IG, SHAP, rationale, counterfactuals
│   ├── api.py             # Flask API endpoints
│   └── evaluation.py      # Evaluation metrics and scripts
├── frontend/              # Web UI
│   ├── static/           # CSS, JS, assets
│   └── templates/        # HTML templates
├── notebooks/             # Jupyter notebooks for demos and analysis
├── scripts/               # Utility scripts
└── tests/                 # Unit tests

```

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

2. **Download/prepare data**:
```bash
python scripts/prepare_data.py
```

3. **Train the model**:
```bash
python scripts/train_model.py --model_name roberta-base --epochs 3
```

4. **Start the API server**:
```bash
python src/api.py
```

5. **Access the UI**:
Open `http://localhost:5000` in your browser

## API Endpoints

### POST `/api/predict`
Predict sentiment with explanations.

**Request**:
```json
{
  "text": "I love this product!",
  "explanation_method": "ig",
  "include_counterfactuals": true
}
```

**Response**:
```json
{
  "label": "positive",
  "confidence": 0.95,
  "attributions": [
    {"token": "love", "score": 0.45, "span": [2, 6]},
    {"token": "product", "score": 0.12, "span": [11, 18]}
  ],
  "rationale": "The sentiment is positive primarily due to the word 'love'...",
  "explanation_method": "ig",
  "counterfactuals": [
    {
      "edit": "I hate this product!",
      "new_label": "negative",
      "delta_confidence": -0.85
    }
  ],
  "provenance": {
    "model": "roberta-base",
    "timestamp": "2024-01-01T12:00:00Z"
  },
  "explainability_quality": "high"
}
```

### GET `/api/health`
Health check endpoint.

## Evaluation

Run evaluation scripts:
```bash
python src/evaluation.py --test_data data/processed/test.json --output results/
```

This computes:
- **Fidelity**: Perturbation tests and surrogate model checks
- **Plausibility**: Human rating interface (stub)
- **Consistency**: Cross-seed consistency checks

## Demo Scripts

See `notebooks/demo.ipynb` for interactive examples, or run:
```bash
python scripts/demo.py --scenario finance
python scripts/demo.py --scenario healthcare
```

## Performance

- **Inference**: <2s for label + IG on CPU for texts <512 tokens
- **SHAP**: Rate-limited, runs asynchronously for longer texts
- **Accuracy**: Competitive performance on standard sentiment datasets

## License

MIT License

