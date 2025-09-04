import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple
import time


class LLMEvaluator:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def calculate_perplexity(self, data_loader):
        """Calculate perplexity on a dataset"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                # Calculate loss
                logits, loss = self.model(batch, batch)
                
                # Count non-padding tokens
                mask = (batch != self.tokenizer.pad_token_id).float()
                num_tokens = mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity, avg_loss
    
    def generate_text(self, prompt: str, max_length=100, temperature=1.0, 
                     top_k=50, top_p=0.9):
        """Generate text from a prompt"""
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True), 
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits, _ = self.model(generated)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
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
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Crop if too long
                if generated.size(1) > self.model.max_seq_len:
                    generated = generated[:, -self.model.max_seq_len:]
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].tolist())
        return generated_text
    
    def evaluate_generation_quality(self, prompts: List[str], num_samples=3):
        """Evaluate generation quality with multiple prompts"""
        results = []
        
        for prompt in prompts:
            prompt_results = {
                'prompt': prompt,
                'generations': []
            }
            
            print(f"Evaluating prompt: '{prompt}'")
            
            for i in range(num_samples):
                generated = self.generate_text(
                    prompt, 
                    max_length=50, 
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9
                )
                prompt_results['generations'].append(generated)
                print(f"  Generation {i+1}: {generated}")
            
            results.append(prompt_results)
            print()
        
        return results
    
    def calculate_token_accuracy(self, data_loader, top_k=1):
        """Calculate top-k token prediction accuracy"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                # Get predictions
                logits, _ = self.model(batch[:, :-1])  # Input without last token
                targets = batch[:, 1:]  # Targets without first token
                
                # Get top-k predictions
                _, top_k_preds = torch.topk(logits, k=top_k, dim=-1)
                
                # Check if target is in top-k
                mask = (targets != self.tokenizer.pad_token_id)
                targets_expanded = targets.unsqueeze(-1).expand(-1, -1, top_k)
                correct = (top_k_preds == targets_expanded).any(dim=-1)
                
                correct_predictions += (correct & mask).sum().item()
                total_predictions += mask.sum().item()
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy
    
    def benchmark_inference_speed(self, sequence_length=512, batch_size=1, num_runs=10):
        """Benchmark inference speed"""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randint(
            0, len(self.tokenizer.word_to_id), 
            (batch_size, sequence_length)
        ).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        tokens_per_second = (batch_size * sequence_length) / avg_time
        
        return {
            'avg_inference_time': avg_time,
            'tokens_per_second': tokens_per_second,
            'batch_size': batch_size,
            'sequence_length': sequence_length
        }
    
    def comprehensive_evaluation(self, test_loader, prompts=None):
        """Run comprehensive evaluation"""
        print("Running comprehensive evaluation...")
        
        results = {}
        
        # 1. Perplexity
        print("Calculating perplexity...")
        perplexity, avg_loss = self.calculate_perplexity(test_loader)
        results['perplexity'] = perplexity
        results['test_loss'] = avg_loss
        print(f"Test Perplexity: {perplexity:.2f}")
        print(f"Test Loss: {avg_loss:.4f}")
        
        # 2. Token accuracy
        print("\nCalculating token accuracy...")
        top1_acc = self.calculate_token_accuracy(test_loader, top_k=1)
        top5_acc = self.calculate_token_accuracy(test_loader, top_k=5)
        results['top1_accuracy'] = top1_acc
        results['top5_accuracy'] = top5_acc
        print(f"Top-1 Accuracy: {top1_acc:.4f}")
        print(f"Top-5 Accuracy: {top5_acc:.4f}")
        
        # 3. Generation quality
        if prompts:
            print("\nEvaluating generation quality...")
            generation_results = self.evaluate_generation_quality(prompts)
            results['generation_examples'] = generation_results
        
        # 4. Inference speed
        print("\nBenchmarking inference speed...")
        speed_results = self.benchmark_inference_speed()
        results['inference_speed'] = speed_results
        print(f"Inference speed: {speed_results['tokens_per_second']:.0f} tokens/second")
        print(f"Average inference time: {speed_results['avg_inference_time']:.4f} seconds")
        
        return results


def load_model_for_evaluation(checkpoint_path, tokenizer):
    """Load a trained model for evaluation"""
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from model.transformer import create_small_gpt
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with same config
    config = checkpoint.get('config', {})
    vocab_size = len(tokenizer.word_to_id)
    model = create_small_gpt(vocab_size=vocab_size)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model


if __name__ == "__main__":
    # Test evaluation setup
    import sys
    sys.path.append('..')
    
    from model.transformer import create_small_gpt
    from data.tokenizer import SimpleTokenizer
    from data.dataset import download_openwebtext_sample, create_dataloaders
    
    # Create model and data
    tokenizer = SimpleTokenizer(vocab_size=1000)
    texts = download_openwebtext_sample()
    tokenizer.train(texts)
    
    model = create_small_gpt(vocab_size=1000)
    
    # Create test data
    _, _, test_loader = create_dataloaders(
        texts, tokenizer, batch_size=4, max_length=128
    )
    
    # Create evaluator
    evaluator = LLMEvaluator(model, tokenizer)
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "Programming languages are"
    ]
    
    print("Evaluator created successfully!")
    print("Run evaluator.comprehensive_evaluation(test_loader, test_prompts) to start evaluation.")