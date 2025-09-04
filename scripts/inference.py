#!/usr/bin/env python3
"""
Inference script for trained LLM model.

Usage:
    python scripts/inference.py --model models/checkpoint_final.pt --prompt "Hello world"
    python scripts/inference.py --model models/best_model.pt --prompt "The future of AI is" --max_tokens 100
"""

import argparse
import os
import sys
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.transformer import create_small_gpt
from data.tokenizer import SimpleTokenizer


def load_model_and_tokenizer(checkpoint_path):
    """Load trained model and tokenizer from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Create model with same config as training
    model = create_small_gpt(
        vocab_size=model_config.get('vocab_size', 1000),
        d_model=model_config.get('d_model', 512),
        n_heads=model_config.get('n_heads', 8),
        n_layers=model_config.get('n_layers', 6),
        max_seq_len=model_config.get('max_seq_len', 1024)
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer
    tokenizer = SimpleTokenizer(vocab_size=model_config.get('vocab_size', 1000))
    if 'tokenizer_state' in checkpoint:
        tokenizer.load_state(checkpoint['tokenizer_state'])
    else:
        print("Warning: No tokenizer state found in checkpoint. Using default tokenizer.")
    
    return model, tokenizer, config


def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8, top_k=50):
    """Generate text using the trained model"""
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], device=device)
    
    print(f"Prompt: {prompt}")
    print(f"Generating {max_tokens} tokens...")
    print("-" * 50)
    
    generated_tokens = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model predictions
            outputs = model(input_tensor)
            logits = outputs[:, -1, :]  # Last token logits
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_token_id = next_token.item()
            
            # Check for end of sequence
            if next_token_id == tokenizer.pad_token_id:
                break
            
            # Add to generated sequence
            generated_tokens.append(next_token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=-1)
            
            # Decode and print token
            try:
                token_text = tokenizer.decode([next_token_id])
                print(token_text, end='', flush=True)
            except:
                print(f"[{next_token_id}]", end='', flush=True)
    
    print("\n" + "-" * 50)
    
    # Decode full text
    try:
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text
    except:
        print("Warning: Could not decode full text")
        return prompt + " [generation failed]"


def interactive_mode(model, tokenizer):
    """Interactive text generation mode"""
    print("Interactive mode. Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            max_tokens = input("Max tokens (default 50): ").strip()
            max_tokens = int(max_tokens) if max_tokens else 50
            
            temperature = input("Temperature (default 0.8): ").strip()
            temperature = float(temperature) if temperature else 0.8
            
            print()
            generate_text(model, tokenizer, prompt, max_tokens, temperature)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained LLM')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=50,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.0 = greedy, higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling (0 = disabled)')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive mode')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Available model files:")
        models_dir = os.path.dirname(args.model) or './models'
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pt'):
                    print(f"  {os.path.join(models_dir, f)}")
        return
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    try:
        model, tokenizer, config = load_model_and_tokenizer(args.model)
        model = model.to(device)
        
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate text
    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        prompt = args.prompt or "The future of artificial intelligence is"
        generated_text = generate_text(
            model, tokenizer, prompt, 
            args.max_tokens, args.temperature, args.top_k
        )
        print(f"\nFull generated text:\n{generated_text}")


if __name__ == "__main__":
    main()