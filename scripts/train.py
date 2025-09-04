#!/usr/bin/env python3
"""
Training script for the LLM model.

Usage:
    python scripts/train.py --config configs/default_config.json
    python scripts/train.py --config configs/small_config.json --epochs 5
"""

import argparse
import os
import sys
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.transformer import create_small_gpt
from data.tokenizer import SimpleTokenizer
from data.dataset import download_openwebtext_sample, create_dataloaders, load_custom_dataset
from training.trainer import LLMTrainer
from utils.config import Config


def main():
    parser = argparse.ArgumentParser(description='Train LLM model')
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data file (txt or json)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Output directory for models and logs')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.set('training.epochs', args.epochs)
    if args.batch_size is not None:
        config.set('training.batch_size', args.batch_size)
    if args.learning_rate is not None:
        config.set('training.learning_rate', args.learning_rate)
    
    # Update output directories
    config.set('training.checkpoint_dir', os.path.join(args.output_dir, 'checkpoints'))
    config.set('training.log_dir', os.path.join(args.output_dir, 'logs'))
    
    # Validate configuration
    config.validate_config()
    
    print("\nTraining Configuration:")
    config.print_config()
    
    # Load or create tokenizer
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.pkl')
    tokenizer = SimpleTokenizer(vocab_size=config.get('data.tokenizer_vocab_size'))
    
    # Load training data
    if args.data:
        print(f"Loading custom dataset from {args.data}")
        texts = load_custom_dataset(args.data)
    else:
        print("Using sample dataset for demonstration")
        texts = download_openwebtext_sample()
    
    print(f"Dataset size: {len(texts)} texts")
    
    # Train tokenizer if it doesn't exist
    if not os.path.exists(tokenizer_path):
        print("Training tokenizer...")
        tokenizer.train(texts)
        tokenizer.save(tokenizer_path)
    else:
        print("Loading existing tokenizer...")
        tokenizer.load(tokenizer_path)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        texts, 
        tokenizer,
        batch_size=config.get('training.batch_size'),
        max_length=config.get('data.max_length'),
        train_split=config.get('data.train_split'),
        val_split=config.get('data.val_split')
    )
    
    # Create model
    model_config = config.get_model_config()
    model_config['vocab_size'] = len(tokenizer.word_to_id)
    
    print(f"Creating model with {model_config}")
    model = create_small_gpt(**model_config)
    
    # Create trainer
    training_config = config.get_training_config()
    trainer = LLMTrainer(model, train_loader, val_loader, training_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best model saved at: {os.path.join(training_config['checkpoint_dir'], 'best_model.pt')}")
    print(f"Tokenizer saved at: {tokenizer_path}")
    
    # Save final configuration
    final_config_path = os.path.join(args.output_dir, 'final_config.json')
    config.save_config(final_config_path)
    print(f"Final configuration saved at: {final_config_path}")


if __name__ == "__main__":
    main()