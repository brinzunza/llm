#!/usr/bin/env python3
"""
Text generation script for the LLM model.

Usage:
    python scripts/generate.py --model models/checkpoints/best_model.pt --tokenizer models/tokenizer.pkl --prompt "The future of AI"
    python scripts/generate.py --model models/checkpoints/best_model.pt --tokenizer models/tokenizer.pkl --interactive
"""

import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import LLMInference


def main():
    parser = argparse.ArgumentParser(description='Generate text with LLM model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to tokenizer file')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt for generation')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive chat mode')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='Repetition penalty')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    print("Loading model and tokenizer...")
    llm = LLMInference(args.model, args.tokenizer)
    
    if args.interactive:
        # Start interactive chat
        llm.chat()
    elif args.prompt:
        # Generate from prompt
        print(f"\nPrompt: {args.prompt}")
        print("=" * 50)
        
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"\nSample {i+1}:")
            
            generated = llm.generate(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
            
            print(generated)
    else:
        # Default prompts
        default_prompts = [
            "The future of artificial intelligence",
            "Machine learning is revolutionizing",
            "In the world of programming",
            "Data science helps us understand"
        ]
        
        print("No prompt provided. Generating from default prompts:")
        
        for prompt in default_prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 30)
            
            generated = llm.generate(
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
            
            print(generated)
            print()


if __name__ == "__main__":
    main()