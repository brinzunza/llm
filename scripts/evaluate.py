#!/usr/bin/env python3
"""
Evaluation script for the LLM model.

Usage:
    python scripts/evaluate.py --model models/checkpoints/best_model.pt --tokenizer models/tokenizer.pkl
    python scripts/evaluate.py --model models/checkpoints/best_model.pt --tokenizer models/tokenizer.pkl --data test_data.txt
"""

import argparse
import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.tokenizer import SimpleTokenizer
from data.dataset import download_openwebtext_sample, create_dataloaders, load_custom_dataset
from training.evaluator import load_model_for_evaluation, LLMEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to tokenizer file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to evaluation data file (txt or json)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Path to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--prompts', type=str, nargs='*',
                       default=["The future of artificial intelligence",
                               "Machine learning is",
                               "Programming languages are"],
                       help='Prompts for generation evaluation')
    
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.load(args.tokenizer)
    
    print("Loading model...")
    model = load_model_for_evaluation(args.model, tokenizer)
    
    # Load evaluation data
    if args.data:
        print(f"Loading evaluation data from {args.data}")
        texts = load_custom_dataset(args.data)
    else:
        print("Using sample dataset for evaluation")
        texts = download_openwebtext_sample()
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        texts, tokenizer, batch_size=args.batch_size, max_length=512
    )
    
    # Create evaluator
    evaluator = LLMEvaluator(model, tokenizer)
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(test_loader, args.prompts)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation Results:")
    print(f"Test Perplexity: {results['perplexity']:.2f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print(f"Inference Speed: {results['inference_speed']['tokens_per_second']:.0f} tokens/sec")
    
    print(f"\nGeneration Examples:")
    if 'generation_examples' in results:
        for example in results['generation_examples'][:3]:  # Show first 3
            print(f"Prompt: {example['prompt']}")
            for i, gen in enumerate(example['generations'][:2]):  # Show first 2 generations
                print(f"  Gen {i+1}: {gen}")
            print()
    
    print(f"Full results saved to: {args.output}")


if __name__ == "__main__":
    main()