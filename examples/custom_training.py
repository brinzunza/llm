#!/usr/bin/env python3
"""
Example of custom training with your own data and configurations.

This example shows how to:
1. Load custom text data
2. Create custom model configurations
3. Train with advanced settings
4. Monitor training progress
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.transformer import GPTModel
from data.tokenizer import SimpleTokenizer
from data.dataset import create_dataloaders
from training.trainer import LLMTrainer
from utils.config import Config


def create_custom_dataset():
    """Create a custom dataset for demonstration"""
    
    # Example: Technology-focused dataset
    tech_texts = [
        "Artificial intelligence is transforming industries across the globe.",
        "Machine learning algorithms can identify patterns in large datasets.",
        "Deep learning neural networks consist of multiple hidden layers.",
        "Natural language processing enables computers to understand human speech.",
        "Computer vision allows machines to interpret visual information.",
        "Data science combines statistics, programming, and domain expertise.",
        "Cloud computing provides scalable infrastructure for applications.",
        "Cybersecurity protects digital assets from malicious attacks.",
        "Software engineering follows systematic approaches to build applications.",
        "Database systems efficiently store and retrieve structured information.",
        "Web development creates interactive online experiences for users.",
        "Mobile applications provide convenient access to digital services.",
        "Internet of Things connects everyday devices to the internet.",
        "Blockchain technology enables secure decentralized transactions.",
        "Quantum computing promises exponential speedups for certain problems.",
        "Robotics integrates mechanical engineering with artificial intelligence.",
        "Virtual reality creates immersive digital environments for users.",
        "Augmented reality overlays digital information onto physical world.",
        "DevOps practices streamline software development and deployment processes.",
        "Agile methodologies emphasize iterative development and team collaboration."
    ]
    
    # Expand dataset with variations
    expanded_texts = []
    for text in tech_texts:
        expanded_texts.append(text)
        # Add variations
        expanded_texts.append(f"In modern technology, {text.lower()}")
        expanded_texts.append(f"Researchers found that {text.lower()}")
        expanded_texts.append(f"The future of {text.lower()}")
        
        # Add related sentences
        if "artificial intelligence" in text.lower():
            expanded_texts.append("AI systems require careful training and validation.")
        elif "machine learning" in text.lower():
            expanded_texts.append("ML models learn from historical data patterns.")
        elif "programming" in text.lower():
            expanded_texts.append("Code quality depends on design principles and testing.")
    
    return expanded_texts


def create_custom_model_config():
    """Create a custom model configuration"""
    
    # Medium-sized model configuration
    custom_config = {
        "model": {
            "vocab_size": 15000,    # Will be updated based on tokenizer
            "d_model": 512,         # Smaller than default for faster training
            "n_heads": 8,           # Fewer heads
            "n_layers": 10,         # Moderate depth
            "max_seq_len": 256,     # Shorter sequences
            "dropout": 0.15         # Slightly higher dropout
        },
        "training": {
            "learning_rate": 4e-4,
            "weight_decay": 0.02,
            "epochs": 8,
            "batch_size": 6,
            "max_seq_length": 256,
            "log_interval": 8,
            "save_interval": 2,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 80,
            "checkpoint_dir": "./custom_models/checkpoints",
            "log_dir": "./custom_models/logs"
        },
        "data": {
            "train_split": 0.75,
            "val_split": 0.15,
            "test_split": 0.10,
            "max_length": 256,
            "tokenizer_vocab_size": 15000
        }
    }
    
    return Config.from_dict(custom_config)


def monitor_training_callback(trainer, epoch, train_loss, val_loss):
    """Custom callback to monitor training progress"""
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
    
    # Custom early stopping logic
    if val_loss < 2.0:  # Example threshold
        print("  Good validation loss achieved!")
    
    # Generate sample text every few epochs
    if epoch % 3 == 0:
        print("  Sample generation:")
        sample_text = trainer.model.generate(
            torch.tensor([[1, 2, 3]], device=trainer.device),  # Simple input
            max_new_tokens=20,
            temperature=0.8
        )
        print(f"    Generated: {sample_text}")


def main():
    print("=== Custom Training Example ===\n")
    
    # 1. Create custom dataset
    print("1. Creating custom dataset...")
    custom_texts = create_custom_dataset()
    print(f"   Dataset size: {len(custom_texts)} texts")
    print(f"   Sample text: {custom_texts[0]}")
    
    # 2. Create custom configuration
    print("\n2. Setting up custom configuration...")
    config = create_custom_model_config()
    config.validate_config()
    print("   Configuration validated successfully")
    
    # 3. Setup tokenizer
    print("\n3. Training custom tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=config.get('data.tokenizer_vocab_size'))
    tokenizer.train(custom_texts)
    print(f"   Vocabulary size: {len(tokenizer.word_to_id)}")
    
    # Update model config with actual vocab size
    config.set('model.vocab_size', len(tokenizer.word_to_id))
    
    # 4. Create model with custom architecture
    print("\n4. Creating custom model...")
    model_config = config.get_model_config()
    
    model = GPTModel(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        max_seq_len=model_config['max_seq_len'],
        dropout=model_config['dropout']
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {param_count:,}")
    
    # 5. Create data loaders
    print("\n5. Preparing data loaders...")
    data_config = config.get_data_config()
    train_loader, val_loader, test_loader = create_dataloaders(
        custom_texts,
        tokenizer,
        batch_size=config.get('training.batch_size'),
        max_length=data_config['max_length'],
        train_split=data_config['train_split'],
        val_split=data_config['val_split']
    )
    
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # 6. Setup custom trainer
    print("\n6. Initializing custom trainer...")
    training_config = config.get_training_config()
    trainer = LLMTrainer(model, train_loader, val_loader, training_config)
    
    # Add custom training hooks (if implemented)
    # trainer.add_callback('epoch_end', monitor_training_callback)
    
    # 7. Start training
    print("\n7. Starting custom training...")
    print("   This will train for several epochs...")
    trainer.train()
    
    # 8. Save everything
    print("\n8. Saving results...")
    os.makedirs('./custom_models', exist_ok=True)
    
    # Save tokenizer
    tokenizer.save('./custom_models/tokenizer.pkl')
    
    # Save final configuration
    config.save_config('./custom_models/training_config.json')
    
    print("   Custom training completed!")
    print("   Model: ./custom_models/checkpoints/best_model.pt")
    print("   Tokenizer: ./custom_models/tokenizer.pkl")
    print("   Config: ./custom_models/training_config.json")
    
    # 9. Quick evaluation
    print("\n9. Quick evaluation...")
    from training.evaluator import LLMEvaluator
    
    evaluator = LLMEvaluator(model, tokenizer)
    perplexity, loss = evaluator.calculate_perplexity(test_loader)
    
    print(f"   Final test perplexity: {perplexity:.2f}")
    print(f"   Final test loss: {loss:.4f}")
    
    # Test generation
    test_prompts = [
        "Artificial intelligence will",
        "The future of programming",
        "Machine learning helps"
    ]
    
    print("\n   Sample generations:")
    for prompt in test_prompts:
        generated = evaluator.generate_text(prompt, max_length=25, temperature=0.7)
        print(f"     '{prompt}' -> '{generated}'")
    
    print("\n=== Custom Training Example Complete! ===")


if __name__ == "__main__":
    main()