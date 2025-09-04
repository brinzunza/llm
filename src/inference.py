import torch
import torch.nn.functional as F
import argparse
import json
import os
from typing import Optional, List

from model.transformer import create_small_gpt
from data.tokenizer import SimpleTokenizer
from training.evaluator import LLMEvaluator


class LLMInference:
    def __init__(self, model_path: str, tokenizer_path: str, device: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            tokenizer_path: Path to tokenizer file
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Inference engine ready on {self.device}")
        print(f"Vocabulary size: {len(self.tokenizer.word_to_id)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config
        vocab_size = len(self.tokenizer.word_to_id)
        model = create_small_gpt(vocab_size=vocab_size)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print training info
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch']} epochs")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        
        return model
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0,
                top_k: Optional[int] = 50, top_p: float = 0.9, 
                repetition_penalty: float = 1.0) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Track generated tokens for repetition penalty
        generated_tokens = input_ids.clone()
        
        with torch.no_grad():
            for step in range(max_length):
                # Get model predictions
                logits, _ = self.model(generated_tokens)
                next_token_logits = logits[0, -1, :].clone()
                
                # Apply repetition penalty
                if repetition_penalty != 1.0 and generated_tokens.size(1) > 1:
                    for token_id in set(generated_tokens[0].tolist()):
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] *= repetition_penalty
                        else:
                            next_token_logits[token_id] /= repetition_penalty
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if torch.all(torch.isinf(next_token_logits)):
                    # If all tokens are filtered out, pick the original top token
                    next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0)
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to sequence
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                
                # Crop if sequence gets too long
                if generated_tokens.size(1) > self.model.max_seq_len:
                    generated_tokens = generated_tokens[:, -self.model.max_seq_len:]
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens[0].tolist())
        return generated_text
    
    def chat(self):
        """Interactive chat mode"""
        print("\n=== LLM Chat Mode ===")
        print("Type 'quit' to exit, 'clear' to clear history")
        print("Use /settings to adjust generation parameters")
        print("-" * 40)
        
        # Default settings
        settings = {
            'max_length': 50,
            'temperature': 0.8,
            'top_k': 40,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }
        
        conversation_history = ""
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    conversation_history = ""
                    print("Conversation history cleared.")
                    continue
                elif user_input.lower() == '/settings':
                    self._adjust_settings(settings)
                    continue
                elif not user_input:
                    continue
                
                # Add user input to conversation
                conversation_history += f"User: {user_input}\nAssistant: "
                
                # Generate response
                response = self.generate(
                    conversation_history,
                    max_length=settings['max_length'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p'],
                    repetition_penalty=settings['repetition_penalty']
                )
                
                # Extract only the assistant's response
                if "Assistant:" in response:
                    assistant_response = response.split("Assistant:")[-1].strip()
                    # Clean up response
                    assistant_response = assistant_response.split("User:")[0].strip()
                else:
                    assistant_response = response.strip()
                
                print(f"\nAssistant: {assistant_response}")
                
                # Update conversation history
                conversation_history += assistant_response + "\n"
                
                # Keep conversation history manageable
                if len(conversation_history) > 1000:
                    conversation_history = conversation_history[-800:]
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _adjust_settings(self, settings):
        """Adjust generation settings interactively"""
        print("\nCurrent settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        print("\nEnter new values (press Enter to keep current):")
        
        for key in settings:
            try:
                new_value = input(f"{key} [{settings[key]}]: ").strip()
                if new_value:
                    if key in ['max_length', 'top_k']:
                        settings[key] = int(new_value)
                    else:
                        settings[key] = float(new_value)
                    print(f"Updated {key} to {settings[key]}")
            except ValueError:
                print(f"Invalid value for {key}, keeping current value")
    
    def batch_generate(self, prompts: List[str], output_file: Optional[str] = None, **kwargs):
        """Generate text for multiple prompts"""
        results = []
        
        print(f"Generating text for {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            generated = self.generate(prompt, **kwargs)
            
            result = {
                'prompt': prompt,
                'generated': generated,
                'settings': kwargs
            }
            results.append(result)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='LLM Inference Engine')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer file')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--mode', type=str, default='chat', choices=['chat', 'generate', 'batch'],
                       help='Inference mode')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling threshold')
    parser.add_argument('--output', type=str, help='Output file for batch mode')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    llm = LLMInference(args.model, args.tokenizer, args.device)
    
    if args.mode == 'chat':
        llm.chat()
    elif args.mode == 'generate':
        if not args.prompt:
            print("Error: --prompt required for generate mode")
            return
        
        result = llm.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"\nGenerated text:\n{result}")
    elif args.mode == 'batch':
        if not args.output:
            print("Error: --output required for batch mode")
            return
        
        # Example prompts for batch generation
        prompts = [
            "The future of artificial intelligence",
            "Machine learning helps us",
            "Programming is the art of",
            "Data science involves"
        ]
        
        llm.batch_generate(
            prompts,
            args.output,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )


if __name__ == "__main__":
    main()