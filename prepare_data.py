"""
Script to prepare and preprocess data for training.
"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_pipeline import SentimentDataset, load_sample_data


def main():
    print("Preparing data for training...")
    
    # Load sample data (in production, load from actual dataset)
    texts, labels = load_sample_data()
    
    print(f"Loaded {len(texts)} examples")
    
    # Prepare dataset
    dataset = SentimentDataset()
    splits = dataset.prepare_dataset(
        texts, labels,
        test_size=0.2,
        val_size=0.1,
        balance_classes=True,
        random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train']['labels'])} examples")
    print(f"  Val: {len(splits['val']['labels'])} examples")
    print(f"  Test: {len(splits['test']['labels'])} examples")
    
    # Save processed data
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
    
    print("\nSaved processed data to data/processed/splits.json")
    
    # Create attribution eval set
    os.makedirs('data/annotations', exist_ok=True)
    eval_set = dataset.create_attribution_eval_set(texts, labels, n_examples=10)
    
    with open('data/annotations/attribution_eval.json', 'w') as f:
        json.dump(eval_set, f, indent=2)
    
    print("Saved attribution evaluation set to data/annotations/attribution_eval.json")
    print("\nData preparation complete!")


if __name__ == '__main__':
    main()

