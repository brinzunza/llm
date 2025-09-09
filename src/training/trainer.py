import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import time
import os
from typing import Dict, Any
import json


class LLMTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Multi-GPU setup
        self.use_multi_gpu = torch.cuda.device_count() > 1
        self.world_size = torch.cuda.device_count() if self.use_multi_gpu else 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup model for multi-GPU or single GPU
        if self.use_multi_gpu:
            print(f"Using {self.world_size} GPUs for training")
            # Use DataParallel for simpler multi-GPU setup (no distributed training setup required)
            self.model = nn.DataParallel(model)
            self.model.to(self.device)
        else:
            print(f"Using single device: {self.device}")
            self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        # For streaming datasets, estimate steps based on target tokens
        if hasattr(train_loader.dataset, 'target_tokens'):
            estimated_samples = train_loader.dataset.target_tokens // config.get('data', {}).get('max_length', 1024)
            estimated_steps = estimated_samples // config['batch_size']
            total_steps = estimated_steps
            print(f"Estimated training steps: {total_steps:,}")
        else:
            total_steps = len(train_loader) * config['epochs']
            
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Mixed precision training
        self.use_mixed_precision = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.use_multi_gpu:
            print(f"Effective batch size with {self.world_size} GPUs: {config['batch_size'] * self.world_size}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        self._log_training_setup()
    
    def _log_training_setup(self):
        """Log training setup information"""
        setup_info = {
            'device': str(self.device),
            'multi_gpu': self.use_multi_gpu,
            'world_size': self.world_size,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count()
        }
        
        print("\n=== Training Setup ===")
        if torch.cuda.is_available():
            if self.use_multi_gpu:
                print(f"✅ Multi-GPU training enabled ({self.world_size} GPUs)")
                print("   This will automatically distribute batches across GPUs")
            else:
                print("✅ Single GPU training enabled")
        else:
            print("ℹ️  CPU training (no CUDA GPUs detected)")
            print("   Consider using a GPU for faster training")
        
        print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print("=" * 25)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulation_loss = 0
        
        # For streaming datasets, we don't have len()
        try:
            total_batches = len(self.train_loader)
        except:
            total_batches = "unknown"
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    logits, loss = self.model(batch, batch)
                    loss = loss / self.gradient_accumulation_steps
            else:
                logits, loss = self.model(batch, batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                accumulation_loss += loss.item()
                
                # Update weights after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.get('gradient_clip_norm', 1.0)
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += accumulation_loss
                    accumulation_loss = 0
                    self.global_step += 1
            else:
                loss.backward()
                accumulation_loss += loss.item()
                
                # Update weights after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.get('gradient_clip_norm', 1.0)
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += accumulation_loss
                    accumulation_loss = 0
                    self.global_step += 1
            
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{total_batches}, "
                      f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, "
                      f"LR: {current_lr:.6f}, Step: {self.global_step}")
        
        if num_batches > 0:
            avg_loss = total_loss / (num_batches // self.gradient_accumulation_steps)
        else:
            avg_loss = 0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                logits, loss = self.model(batch, batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        # Handle DataParallel models - save the underlying model state
        model_state_dict = self.model.module.state_dict() if self.use_multi_gpu else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'use_multi_gpu': self.use_multi_gpu,
            'world_size': self.world_size
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DataParallel models - load into the underlying model
        if self.use_multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        # Handle backward compatibility for checkpoints without multi-GPU info
        checkpoint_multi_gpu = checkpoint.get('use_multi_gpu', False)
        checkpoint_world_size = checkpoint.get('world_size', 1)
        
        if checkpoint_multi_gpu != self.use_multi_gpu:
            print(f"Warning: Checkpoint was saved with multi_gpu={checkpoint_multi_gpu}, "
                  f"but current setup is multi_gpu={self.use_multi_gpu}")
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Loaded model was trained on {checkpoint_world_size} GPU(s), "
              f"current setup uses {self.world_size} GPU(s)")
    
    def calculate_perplexity(self, loss):
        """Calculate perplexity from loss"""
        return math.exp(loss)
    
    def _print_device_info(self):
        """Print comprehensive device information"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print(f"💪 TRAINING WITH {gpu_count} GPUs")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"   • GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                print(f"   • Effective batch size: {self.config['batch_size'] * gpu_count}")
            else:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🎯 TRAINING WITH 1 GPU")
                print(f"   • {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("🖥️  TRAINING WITH CPU")
            print("   • No CUDA GPUs detected - using CPU")
            print("   • Consider using GPU for faster training")
        
        print(f"   • Mixed precision: {'✅ Enabled' if self.use_mixed_precision else '❌ Disabled'}")
        print(f"   • Gradient accumulation: {self.gradient_accumulation_steps} steps")
        print("="*60)
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        self._print_device_info()
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Calculate metrics
            train_perplexity = self.calculate_perplexity(train_loss)
            val_perplexity = self.calculate_perplexity(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
            print("-" * 50)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(is_best)
            
            # Save training log
            self.save_training_log()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint()
    
    def save_training_log(self):
        """Save training metrics to log file"""
        log_data = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        log_path = os.path.join(self.config['log_dir'], 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)


def get_default_config():
    """Get default training configuration"""
    return {
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'epochs': 10,
        'batch_size': 8,
        'max_seq_length': 512,
        'log_interval': 10,
        'save_interval': 1,
        'checkpoint_dir': './models/checkpoints',
        'log_dir': './models/logs',
        'gradient_accumulation_steps': 1,
        'warmup_steps': 100,
    }


if __name__ == "__main__":
    # Test training setup
    import sys
    sys.path.append('..')
    
    from model.transformer import create_small_gpt
    from data.tokenizer import SimpleTokenizer
    from data.dataset import download_openwebtext_sample, create_dataloaders
    
    # Create model and data
    model = create_small_gpt(vocab_size=1000)
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # Load and prepare data
    texts = download_openwebtext_sample()
    tokenizer.train(texts)
    
    train_loader, val_loader, _ = create_dataloaders(
        texts, tokenizer, batch_size=4, max_length=128
    )
    
    # Training config
    config = get_default_config()
    config['epochs'] = 2
    config['batch_size'] = 4
    
    # Create trainer
    trainer = LLMTrainer(model, train_loader, val_loader, config)
    
    print("Trainer created successfully!")
    print("Run trainer.train() to start training.")