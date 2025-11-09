"""
Training script for sentiment analysis model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from src.model import InterpretableSentimentClassifier
from src.data_pipeline import SentimentDataset, load_sample_data


class SentimentDatasetWrapper(Dataset):
    """PyTorch Dataset wrapper."""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'].squeeze(0),
            'attention_mask': self.data[idx]['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs['logits'], labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs['logits'], labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs['logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                       help='Pre-trained model name')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for saved model')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to training data (optional)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    if args.data_path and os.path.exists(args.data_path):
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        texts = data.get('train_texts', [])
        labels = data.get('train_labels', [])
        val_texts = data.get('val_texts', [])
        val_labels = data.get('val_labels', [])
    else:
        # Use sample data
        texts, label_strings = load_sample_data()
        dataset = SentimentDataset()
        splits = dataset.prepare_dataset(texts, label_strings)
        texts = splits['train']['texts']
        labels = splits['train']['labels']
        val_texts = splits['val']['texts']
        val_labels = splits['val']['labels']
    
    print(f"Training examples: {len(texts)}")
    print(f"Validation examples: {len(val_texts)}")
    
    # Prepare datasets
    dataset = SentimentDataset(model_name=args.model_name)
    train_data = [dataset.tokenize_with_offsets(text) for text in texts]
    val_data = [dataset.tokenize_with_offsets(text) for text in val_texts]
    
    train_dataset = SentimentDatasetWrapper(train_data, labels)
    val_dataset = SentimentDatasetWrapper(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print(f"Initializing model: {args.model_name}")
    model = InterpretableSentimentClassifier(
        model_name=args.model_name,
        num_labels=3,
        use_interpretable_head=False
    )
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            model.save(model_path)
            print(f"Saved best model to {model_path}")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()

