# Quick Start Guide

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

2. **Prepare data**:
```bash
python scripts/prepare_data.py
```

## Running the System

### Option 1: Web UI (Recommended)

1. **Start the API server**:
```bash
python src/api.py
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. **Enter text** in the input box and click "Analyze" to see:
   - Sentiment prediction with confidence
   - Token-level attributions (highlighted in the text)
   - Natural language rationale
   - Counterfactual examples
   - Explanation quality score

### Option 2: Command Line Demo

Run the demo script with finance or healthcare examples:
```bash
python scripts/demo.py --scenario finance
python scripts/demo.py --scenario healthcare
python scripts/demo.py --scenario all
```

### Option 3: Jupyter Notebook

Open and run the interactive notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

## Training a Model

To train your own model on custom data:

1. **Prepare your data** in JSON format:
```json
{
  "train_texts": ["text1", "text2", ...],
  "train_labels": ["positive", "negative", ...],
  "val_texts": [...],
  "val_labels": [...]
}
```

2. **Train the model**:
```bash
python scripts/train_model.py --model_name roberta-base --epochs 3 --batch_size 16
```

3. **Use the trained model**:
```bash
python src/api.py --model_path models/best_model.pt
```

## API Usage

### Predict with Explanations

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love this product!",
    "explanation_method": "ig",
    "include_counterfactuals": true
  }'
```

### Response Format

```json
{
  "label": "positive",
  "confidence": 0.95,
  "attributions": [
    {"token": "love", "score": 0.45, "span": [2, 6]}
  ],
  "rationale": "The sentiment is positive...",
  "explanation_method": "ig",
  "counterfactuals": [...],
  "explainability_quality": "high"
}
```

## Evaluation

Run evaluation metrics:
```bash
python src/evaluation.py --test_data data/processed/splits.json --output results/evaluation_results.json
```

This computes:
- **Accuracy**: Model classification accuracy
- **Fidelity**: How well attributions reflect model behavior (perturbation tests)
- **Consistency**: Attribution stability across runs
- **Plausibility**: Human-aligned explanations (heuristic-based)

## Troubleshooting

### Model Download Issues
If you encounter issues downloading models, ensure you have internet access. Models are downloaded from HuggingFace on first use.

### CUDA/GPU Issues
The system automatically uses CPU if CUDA is not available. To force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
python src/api.py
```

### Port Already in Use
If port 5000 is in use, specify a different port:
```bash
export PORT=5001
python src/api.py
```

## Next Steps

- Customize the rationale generator in `src/explainability.py`
- Add your own datasets in `data/raw/`
- Extend counterfactual generation strategies
- Integrate with your own model training pipeline

