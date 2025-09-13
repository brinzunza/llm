#!/usr/bin/env python3
"""
Test script to simulate network failure and verify fallback mechanism
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset import CommonCrawlStreamingDataset
from data.tokenizer import SimpleTokenizer
import datasets

# Monkey patch the load_dataset function to simulate failure
import data.dataset as dataset_module
original_load_dataset = dataset_module.load_dataset

def failing_load_dataset(*args, **kwargs):
    """Simulate network failure"""
    if 'allenai/c4' in args or ('wikitext' in args) or ('openwebtext' in args) or ('bookcorpus' in args):
        raise ConnectionError("Simulated network timeout for testing fallbacks")
    return original_load_dataset(*args, **kwargs)

def test_network_failure_fallback():
    print("=== Testing Network Failure Fallback ===")
    
    # Patch load_dataset to simulate failures
    dataset_module.load_dataset = failing_load_dataset
    
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
    
    try:
        # Create dataset - this should trigger fallback to sample data
        print("\nTesting dataset loading with simulated network failure...")
        dataset = CommonCrawlStreamingDataset(tokenizer, max_length=128, target_tokens=1000)
        
        print("Starting dataset iteration (should use local sample data)...")
        count = 0
        tokens_processed = 0
        
        for batch in dataset:
            count += 1
            tokens_processed += batch.numel()
            print(f"✅ Batch {count}: shape={batch.shape}, tokens_so_far={tokens_processed}")
            
            if count >= 5:  # Get a few batches to test
                break
        
        print(f"\n🎉 Success! Processed {count} batches with {tokens_processed} tokens")
        print(f"📊 Dataset source used: {dataset.dataset_source}")
        print(f"🔄 Samples processed: {dataset.samples_processed}")
        
        # Verify we're using fallback data
        if "Local sample data" in dataset.dataset_source:
            print("✅ Successfully fell back to local sample data!")
            return True
        else:
            print("❌ Expected to use local sample data but didn't")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Restore original function
        dataset_module.load_dataset = original_load_dataset

if __name__ == "__main__":
    success = test_network_failure_fallback()
    if success:
        print("\n✅ Network failure fallback test PASSED!")
        exit(0)
    else:
        print("\n❌ Network failure fallback test FAILED!")
        exit(1)