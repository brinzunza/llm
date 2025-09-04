#!/usr/bin/env python3
"""
Quick start example for training and using the LLM.

This example demonstrates the complete workflow:
1. Data preparation
2. Model training (minimal example)
3. Model evaluation
4. Text generation

Run this script to see the entire pipeline in action.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.transformer import create_small_gpt
from data.tokenizer import SimpleTokenizer
from data.dataset import download_openwebtext_sample, create_dataloaders
from training.trainer import LLMTrainer
from training.evaluator import LLMEvaluator
from inference import LLMInference
from utils.config import Config


def main():
    print("=== LLM Quick Start Example ===\n")
    
    # 1. Data Preparation
    print("1. Preparing data...")
    texts = download_openwebtext_sample()
    print(f"   Loaded {len(texts)} training texts")
    
    # 2. Tokenizer Setup
    print("\n2. Setting up tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=1000)  # Small vocab for demo
    tokenizer.train(texts)
    print(f"   Tokenizer vocabulary size: {len(tokenizer.word_to_id)}")
    
    # 3. Model Creation
    print("\n3. Creating model...")
    model = create_small_gpt(vocab_size=len(tokenizer.word_to_id))
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {param_count:,}")
    
    # 4. Data Loaders
    print("\n4. Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        texts, tokenizer, batch_size=2, max_length=128  # Small batches for demo
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # 5. Training Configuration
    print("\n5. Setting up training...")
    config = {
        'learning_rate': 5e-4,
        'weight_decay': 0.01,
        'epochs': 2,  # Very short training for demo
        'batch_size': 2,
        'max_seq_length': 128,
        'log_interval': 5,
        'save_interval': 1,
        'checkpoint_dir': './demo_models/checkpoints',
        'log_dir': './demo_models/logs',
        'gradient_accumulation_steps': 1,
        'warmup_steps': 10,
    }
    
    # 6. Training
    print("\n6. Training model (this may take a few minutes)...")
    trainer = LLMTrainer(model, train_loader, val_loader, config)
    trainer.train()
    print("   Training completed!")
    
    # 7. Evaluation
    print("\n7. Evaluating model...")
    evaluator = LLMEvaluator(model, tokenizer)
    perplexity, loss = evaluator.calculate_perplexity(test_loader)
    print(f"   Test Perplexity: {perplexity:.2f}")
    print(f"   Test Loss: {loss:.4f}")
    
    # 8. Text Generation
    print("\n8. Generating text...")
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "Programming languages"
    ]
    
    for prompt in test_prompts:
        generated = evaluator.generate_text(
            prompt, 
            max_length=30, 
            temperature=0.8,
            top_k=20
        )
        print(f"   Prompt: '{prompt}'")
        print(f"   Generated: '{generated}'")
        print()
    
    # 9. Save Results
    print("9. Saving model and tokenizer...")
    os.makedirs('./demo_models', exist_ok=True)
    
    # Save tokenizer
    tokenizer.save('./demo_models/tokenizer.pkl')
    
    # Model is already saved by trainer
    print("   Model saved at: ./demo_models/checkpoints/best_model.pt")
    print("   Tokenizer saved at: ./demo_models/tokenizer.pkl")
    
    print("\n=== Quick Start Complete! ===")
    print("\nNext steps:")
    print("1. Try training on your own text data")
    print("2. Experiment with different model sizes")
    print("3. Adjust generation parameters")
    print("4. Use the interactive chat mode:")
    print("   python src/inference.py --model demo_models/checkpoints/best_model.pt --tokenizer demo_models/tokenizer.pkl --mode chat")


if __name__ == "__main__":
    main()