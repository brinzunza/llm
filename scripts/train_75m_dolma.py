#!/usr/bin/env python3
"""
Training script for 75M parameter model on Dolma dataset
"""

import sys
import os
import argparse
import json
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.transformer import create_75m_gpt, count_parameters
from data.tokenizer import SimpleTokenizer
from data.dataset import create_common_crawl_dataloaders
from training.trainer import LLMTrainer
from utils.config import Config


def verify_common_crawl_dataset():
    """Test Common Crawl dataset connectivity"""
    print("\n=== Testing Common Crawl Dataset Connection ===")
    try:
        from datasets import load_dataset
        print("🔄 Testing connection to allenai/c4 (Common Crawl)...")
        
        # Try to load just the first few samples to test connectivity
        test_dataset = load_dataset(
            "allenai/c4", 
            "en",
            streaming=True, 
            split="train"
        )
        
        # Try to get a few samples
        sample_count = 0
        for sample in test_dataset:
            sample_count += 1
            if sample_count >= 3:  # Just test the first few samples
                break
                
        print(f"✅ Successfully connected to Common Crawl! Retrieved {sample_count} test samples")
        return True, "Common Crawl dataset is accessible"
        
    except Exception as e:
        print(f"❌ Common Crawl connection failed: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Train 75M LLM on Common Crawl dataset')
    parser.add_argument('--config', type=str, 
                       default='configs/75m_dolma_config.json',
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--dry_run', action='store_true',
                       help='Test setup without training')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = Config(args.config)
    config.validate_config()
    
    # Display configuration
    print("\n=== Training Configuration ===")
    print(f"Model: 75M parameters")
    print(f"Dataset: Common Crawl (target: {config.get('data.target_tokens'):,} tokens)")
    print(f"Batch size: {config.get('training.batch_size')}")
    print(f"Sequence length: {config.get('data.max_length')}")
    print(f"Learning rate: {config.get('training.learning_rate')}")
    print(f"Mixed precision: {config.get('training.mixed_precision')}")
    print(f"Gradient accumulation: {config.get('training.gradient_accumulation_steps')}")
    
    # Create model
    print("\n=== Creating Model ===")
    model = create_75m_gpt(vocab_size=config.get('model.vocab_size'))
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test Common Crawl dataset connectivity early
    cc_available, cc_status = verify_common_crawl_dataset()
    
    # Create tokenizer
    print("\n=== Setting up Tokenizer ===")
    tokenizer = SimpleTokenizer(
        vocab_size=config.get('data.tokenizer_vocab_size')
    )
    
    # Train tokenizer on small sample first (this is a limitation of our simple tokenizer)
    print("Training tokenizer on sample data...")
    sample_texts = [
        "The future of artificial intelligence",
        "Machine learning transforms data into insights",
        "Deep learning networks process information",
        "Natural language processing enables communication",
        "Computer vision interprets visual data"
    ]
    tokenizer.train(sample_texts)
    
    # Create dataloaders
    print("\n=== Setting up Data ===")
    try:
        train_loader, val_loader, _ = create_common_crawl_dataloaders(
            tokenizer, 
            config.to_dict()
        )
        
        # Display dataset verification information
        print("\n=== Dataset Verification ===")
        print("✅ Common Crawl dataloaders created successfully!")
        
        # Get dataset info from the underlying dataset
        train_dataset_info = train_loader.dataset.get_dataset_info()
        print(f"📊 Dataset source: {train_dataset_info['source']}")
        print(f"🎯 Target training tokens: {train_dataset_info['target_tokens']:,}")
        print(f"📏 Sequence length: {train_dataset_info['max_length']}")
        
        # Verify Common Crawl dataset is being used correctly
        if "common crawl" in train_dataset_info['source'].lower() or "c4" in train_dataset_info['source'].lower():
            print("✅ Common Crawl dataset is correctly configured and connected!")
            print("🎉 You are training on the official Common Crawl dataset!")
        elif "fallback" in train_dataset_info['source'].lower():
            print("⚠️  Using fallback dataset - Common Crawl may not be accessible")
            print("💡 This will still work for testing, but consider checking your internet connection")
        else:
            print("❓ Dataset source unclear - may be using fallback data")
            
        # Show expected training size
        total_examples = train_dataset_info['target_tokens'] // train_dataset_info['max_length']
        print(f"📋 Expected training examples: ~{total_examples:,}")
        
        # Estimate training time
        batch_size = config.get('training.batch_size')
        estimated_steps = total_examples // batch_size
        print(f"⏱️  Estimated training steps: ~{estimated_steps:,}")
            
    except Exception as e:
        print(f"❌ Error loading Common Crawl dataset: {e}")
        print("This may be due to network issues or dataset access.")
        return 1
    
    if args.dry_run:
        print("\n=== Dry Run Complete ===")
        print("Setup successful! Use --dry_run=false to start training.")
        return 0
    
    # Create trainer
    print("\n=== Initializing Trainer ===")
    trainer = LLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.get_training_config()
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("\n=== Starting Training ===")
    print("This will train a 75M parameter model on 5B tokens from Common Crawl")
    print("Estimated training time: 3-6 days on RTX 3060")
    print("Press Ctrl+C to stop training and save checkpoint\n")
    
    try:
        trainer.train()
        print("\n=== Training Complete ===")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Model saved to: {config.get('training.checkpoint_dir')}/best_model.pt")
        
    except KeyboardInterrupt:
        print("\n=== Training Interrupted ===")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        print(f"Checkpoint saved to: {config.get('training.checkpoint_dir')}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)