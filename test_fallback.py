#!/usr/bin/env python3
"""
Test script to verify shared dataset functionality works
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset import create_common_crawl_dataloaders, get_shared_dataset
from data.tokenizer import SimpleTokenizer

def test_shared_dataset():
    print("=== Testing Shared Dataset Between Train/Val ===")

    # Create tokenizer
    print("Setting up tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=100)
    sample_texts = [
        'Machine learning is fascinating.',
        'Python is a great programming language.',
        'Deep learning uses neural networks.',
        'Artificial intelligence is the future.',
        'Data science combines statistics and programming.'
    ]
    tokenizer.train(sample_texts)
    print(f"✅ Tokenizer created with vocab size: {tokenizer.vocab_size}")

    # Create config for small test
    config = {
        'training': {'batch_size': 2},
        'data': {
            'max_length': 128,
            'target_tokens': 1000,  # Small for quick test
            'val_split': 0.2
        }
    }

    try:
        print("\n🧪 Testing dataloader creation with shared dataset...")
        train_loader, val_loader, _ = create_common_crawl_dataloaders(tokenizer, config)

        print("✅ Both dataloaders created successfully!")

        # Test that we can get data from both
        print("\n🧪 Testing training dataloader...")
        train_batch = next(iter(train_loader))
        print(f"✅ Training batch: shape={train_batch.shape}")

        print("\n🧪 Testing validation dataloader...")
        val_batch = next(iter(val_loader))
        print(f"✅ Validation batch: shape={val_batch.shape}")

        print("\n🎉 Success! Both train and validation datasets work with shared data source!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_reuse():
    print("\n=== Testing Dataset Cache Reuse ===")

    try:
        # Clear any existing cache to test fresh
        from data.dataset import _dataset_cache
        _dataset_cache.clear()

        print("🧪 Testing first dataset access...")
        dataset1 = get_shared_dataset("common_crawl")
        print("✅ First access completed")

        print("🧪 Testing second dataset access (should be cached)...")
        dataset2 = get_shared_dataset("common_crawl")
        print("✅ Second access completed")

        if dataset1 is dataset2:
            print("✅ Cache working: Same dataset object returned")
            return True
        else:
            print("❌ Cache not working: Different objects returned")
            return False

    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_cache_reuse()
    success2 = test_shared_dataset()

    if success1 and success2:
        print("\n✅ All shared dataset tests PASSED!")
        print("🎉 Single download now works for both training and validation!")
        exit(0)
    else:
        print("\n❌ Some tests FAILED!")
        exit(1)