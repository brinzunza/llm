# My LLM - A Complete Language Model Training Framework

A comprehensive framework for training and using your own Large Language Model (LLM), now optimized for a 75M parameter model trained on 5B tokens from the Dolma dataset, designed to run efficiently on consumer hardware like RTX 3060.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Understanding the Code](#understanding-the-code)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project implements a complete pipeline for training a GPT-style transformer language model from scratch. The model is designed to be:
- **Lightweight**: ~150M parameters, trainable on consumer GPUs
- **Educational**: Well-documented code for learning purposes
- **Practical**: Includes training, evaluation, and inference capabilities
- **Modular**: Easy to modify and experiment with

### Model Specifications
- **Architecture**: GPT-style decoder-only transformer
- **Parameters**: ~75 million (optimized for efficiency)
- **Dataset**: Dolma (5 billion tokens)
- **Vocabulary**: 32,000 tokens (customizable)
- **Context Length**: 1024 tokens (extended)
- **Layers**: 10 transformer blocks
- **Hidden Size**: 640
- **Attention Heads**: 10

### Training Specifications
- **Training Time**: 3-6 days on RTX 3060
- **Memory Usage**: ~8-10GB VRAM with mixed precision
- **Effective Batch Size**: 48 (12 batch size × 4 gradient accumulation)
- **Token-to-Parameter Ratio**: ~67 tokens per parameter (well optimized)

## Architecture

The model implements a standard transformer decoder architecture with:

### Core Components
1. **Token Embeddings**: Convert input tokens to dense vectors
2. **Position Embeddings**: Add positional information to token embeddings
3. **Transformer Blocks**: Stack of self-attention and feed-forward layers
4. **Layer Normalization**: Stabilize training with pre-layer normalization
5. **Causal Attention**: Ensure the model only attends to previous tokens
6. **Output Projection**: Project hidden states to vocabulary logits

### Key Features
- **Weight Tying**: Token embedding and output projection share weights
- **GELU Activation**: More sophisticated activation than ReLU
- **Dropout**: Regularization to prevent overfitting
- **Gradient Clipping**: Stabilize training dynamics
- **Cosine Learning Rate Schedule**: Smooth learning rate decay

## Features

### Training
- Configurable model architectures
- Automatic mixed precision support
- Gradient accumulation for large effective batch sizes
- Learning rate scheduling
- Comprehensive logging and checkpointing
- Validation-based early stopping

### Evaluation
- Perplexity calculation
- Token-level accuracy metrics
- Text generation quality assessment
- Inference speed benchmarking

### Inference
- Interactive chat mode
- Batch text generation
- Configurable sampling strategies (temperature, top-k, top-p)
- Repetition penalty
- Multiple output formats

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended but not required)

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd my_llm

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models/checkpoints models/logs
```

## Quick Start

### 1. Install Dependencies
```bash
# Clone the repository
git clone <your-repo-url>
cd my_llm

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models/checkpoints models/logs
```

### 2. Test Setup (Recommended)
```bash
# Test the configuration without actually training
python scripts/train_75m_dolma.py --dry_run
```

### 3. Start Training
```bash
# Train the 75M parameter model on Dolma dataset (5B tokens)
python scripts/train_75m_dolma.py

# Or train in the background and save logs
nohup python scripts/train_75m_dolma.py > training.log 2>&1 &
```

### 4. Monitor Progress
```bash
# Watch training progress
tail -f training.log

# Check GPU usage
nvidia-smi -l 1
```

### 5. Use Your Trained Model
```bash
python src/inference.py --model models/checkpoints/best_model.pt --tokenizer models/tokenizer.pkl --mode chat
```

## Detailed Usage

### Training a Model

1. **Prepare Your Dataset**:
   ```python
   # For custom text data
   from src.data.dataset import load_custom_dataset
   texts = load_custom_dataset('path/to/your/data.txt')
   ```

2. **Configure Training**:
   Edit `configs/default_config.json` or create a custom config:
   ```json
   {
     "model": {
       "d_model": 768,
       "n_heads": 12,
       "n_layers": 12
     },
     "training": {
       "learning_rate": 3e-4,
       "batch_size": 8,
       "epochs": 10
     }
   }
   ```

3. **Start Training**:
   ```bash
   python scripts/train.py --config your_config.json
   ```

### Model Evaluation

Evaluate your trained model:
```python
from src.training.evaluator import LLMEvaluator

evaluator = LLMEvaluator(model, tokenizer)
results = evaluator.comprehensive_evaluation(test_loader, prompts)
```

### Text Generation

Generate text using various strategies:
```python
from src.inference import LLMInference

llm = LLMInference('path/to/model.pt', 'path/to/tokenizer.pkl')

# Simple generation
text = llm.generate("The future of AI is", max_length=50)

# Interactive chat
llm.chat()
```

## Configuration

### Model Configurations

The framework includes predefined configurations:

- **Small Model** (`configs/small_config.json`): ~50M parameters
  - 8 layers, 512 hidden size, 8 attention heads
  - Good for experimentation and fast training

- **Default Model** (`configs/default_config.json`): ~150M parameters
  - 12 layers, 768 hidden size, 12 attention heads
  - Balanced performance and resource usage

### Custom Configuration

Create custom configurations by modifying:
```json
{
  "model": {
    "vocab_size": 32000,      // Vocabulary size
    "d_model": 768,           // Hidden dimension
    "n_heads": 12,            // Number of attention heads
    "n_layers": 12,           // Number of transformer layers
    "max_seq_len": 512,       // Maximum sequence length
    "dropout": 0.1            // Dropout rate
  },
  "training": {
    "learning_rate": 3e-4,    // Initial learning rate
    "weight_decay": 0.01,     // L2 regularization
    "epochs": 10,             // Number of training epochs
    "batch_size": 8,          // Batch size
    "gradient_accumulation_steps": 1
  }
}
```

## Understanding the Code

### Core Components

#### 1. Model Architecture (`src/model/transformer.py`)
The transformer implementation follows the GPT architecture:

- **MultiHeadAttention**: Implements scaled dot-product attention
- **FeedForward**: Position-wise feed-forward networks
- **TransformerBlock**: Combines attention and feed-forward with residual connections
- **GPTModel**: The complete model with embeddings and output projection

#### 2. Tokenization (`src/data/tokenizer.py`)
Simple word-based tokenizer:
- Vocabulary building from training corpus
- Text preprocessing and normalization
- Encoding/decoding between text and token IDs
- Special token handling (PAD, UNK, BOS, EOS)

#### 3. Dataset Handling (`src/data/dataset.py`)
- Text preprocessing and chunking
- Dataset splitting (train/validation/test)
- DataLoader creation with proper batching
- Support for custom datasets

#### 4. Training Loop (`src/training/trainer.py`)
Comprehensive training implementation:
- Forward and backward passes
- Gradient clipping and accumulation
- Learning rate scheduling
- Checkpointing and logging
- Validation monitoring

#### 5. Evaluation (`src/training/evaluator.py`)
Multiple evaluation metrics:
- Perplexity calculation
- Token-level accuracy
- Generation quality assessment
- Speed benchmarking

### Key Concepts

#### Attention Mechanism
The model uses causal self-attention to process sequences:
```python
# Scaled dot-product attention
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attention_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attention_weights, V)
```

#### Positional Encoding
Learned positional embeddings are added to token embeddings:
```python
token_emb = self.token_embedding(input_ids)
pos_emb = self.position_embedding(position_ids)
x = token_emb + pos_emb
```

#### Causal Masking
The model only attends to previous tokens:
```python
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, -1e9)
```

## Training Process

### Data Preparation
1. **Text Collection**: Gather text data relevant to your domain
2. **Preprocessing**: Clean and normalize text
3. **Tokenization**: Convert text to token sequences
4. **Dataset Creation**: Split into train/validation/test sets

### Model Training
1. **Initialization**: Random weight initialization with proper scaling
2. **Forward Pass**: Process input through transformer layers
3. **Loss Calculation**: Cross-entropy loss for next-token prediction
4. **Backpropagation**: Compute gradients and update weights
5. **Validation**: Monitor performance on held-out data

### Training Tips
- Start with a small model for experimentation
- Use gradient accumulation if memory is limited
- Monitor validation loss to prevent overfitting
- Adjust learning rate based on loss curves
- Save checkpoints regularly

## Evaluation

### Automatic Metrics
- **Perplexity**: Measures how well the model predicts the next token
- **Accuracy**: Token-level prediction accuracy
- **Speed**: Inference tokens per second

### Manual Evaluation
- **Generation Quality**: Coherence and relevance of generated text
- **Domain Adaptation**: Performance on domain-specific tasks
- **Prompt Following**: Ability to follow instructions

### Example Evaluation
```python
# Load trained model
evaluator = LLMEvaluator(model, tokenizer)

# Calculate perplexity
perplexity, loss = evaluator.calculate_perplexity(test_loader)
print(f"Perplexity: {perplexity:.2f}")

# Test generation
prompts = ["The future of technology", "Machine learning helps"]
results = evaluator.evaluate_generation_quality(prompts)
```

## Inference

### Interactive Chat
```bash
python src/inference.py --model best_model.pt --tokenizer tokenizer.pkl --mode chat
```

### Batch Generation
```python
llm = LLMInference('model.pt', 'tokenizer.pkl')
prompts = ["Prompt 1", "Prompt 2"]
results = llm.batch_generate(prompts, output_file='results.json')
```

### Generation Parameters
- **Temperature**: Controls randomness (0.1 = conservative, 1.0+ = creative)
- **Top-k**: Consider only top k tokens
- **Top-p**: Nucleus sampling threshold
- **Repetition Penalty**: Reduce repetitive text

## File Structure

```
my_llm/
├── src/
│   ├── model/
│   │   └── transformer.py      # Model architecture
│   ├── data/
│   │   ├── tokenizer.py        # Tokenization
│   │   └── dataset.py          # Dataset handling
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   └── evaluator.py        # Evaluation metrics
│   ├── utils/
│   │   └── config.py           # Configuration management
│   └── inference.py            # Inference engine
├── configs/
│   ├── default_config.json     # Default configuration
│   └── small_config.json       # Small model configuration
├── scripts/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── generate.py            # Generation script
├── data/                      # Training data
├── models/                    # Saved models
│   ├── checkpoints/          # Training checkpoints
│   └── logs/                 # Training logs
├── examples/                  # Usage examples
├── tests/                     # Unit tests
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## Memory and Compute Requirements

### Optimal Configuration (75M Model)
- **RAM**: 16GB
- **GPU**: RTX 3060 (12GB VRAM) or better
- **Storage**: 50GB for dataset and checkpoints
- **Training Time**: 3-6 days continuous

### Minimum Requirements
- **RAM**: 12GB
- **GPU**: 8GB+ VRAM
- **Storage**: 30GB for dataset and model
- **Training Time**: 7-10 days (with smaller batch size)

### Not Recommended
- **Less than 8GB VRAM**: Training will be too slow or fail
- **CPU Only**: Would take 2-3 months to complete

### Expected Results
After training, you should expect:
- **Perplexity**: 15-25 on validation set
- **Generation Quality**: Coherent 50-100 token responses
- **Capabilities**: Basic conversation, simple reasoning, factual recall

## Troubleshooting

### Common Issues

#### Dolma Dataset Loading Fails
```bash
# Test your internet connection
ping huggingface.co

# Try with smaller dataset first
# Edit config and reduce target_tokens to 100000000 (100M) for testing

# If Dolma fails, the script automatically falls back to C4 dataset
```

#### Out of Memory During Training
```bash
# Reduce batch size in configs/75m_dolma_config.json
"batch_size": 8,  # down from 12
"gradient_accumulation_steps": 6,  # up from 4

# Or reduce sequence length
"max_length": 512,  # down from 1024
```

#### Training Appears Stuck
```bash
# Check GPU utilization
nvidia-smi

# If GPU usage is low, dataset may be loading slowly
# This is normal for the first few hours while Dolma streams
```

#### Loss Not Decreasing
```bash
# Monitor training for at least 1000 steps
# Initial loss should be ~10-11, should drop to ~3-4 after several hours

# If loss stays high after 5000 steps:
# 1. Check learning rate (should be around 5e-4)
# 2. Verify tokenizer is working
# 3. Check gradient accumulation is functioning
```

#### Training Interrupted
```bash
# Resume from the last checkpoint
python scripts/train_75m_dolma.py --resume models/checkpoints/checkpoint_epoch_X.pt

# Check what checkpoints are available
ls models/checkpoints/
```

### Debug Commands
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test model creation
python src/model/transformer.py

# Test tokenizer
python src/data/tokenizer.py

# Test training setup
python src/training/trainer.py
```

## Advanced Usage

### Custom Architectures
Modify `src/model/transformer.py` to experiment with:
- Different attention mechanisms
- Architectural variations (MLP size, layer count)
- Novel activation functions
- Custom normalization schemes

### Custom Loss Functions
Implement domain-specific losses in the training loop:
```python
def custom_loss(logits, targets, model):
    # Standard cross-entropy
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    # Add custom regularization
    reg_loss = sum(p.pow(2.0).sum() for p in model.parameters())
    
    return loss + 0.01 * reg_loss
```

### Multi-GPU Training
For larger models, implement distributed training:
```python
# Use DataParallel for simple multi-GPU
model = nn.DataParallel(model)

# Or DistributedDataParallel for better performance
model = nn.parallel.DistributedDataParallel(model)
```

## Performance Optimization

### Training Optimization
- Use automatic mixed precision (AMP)
- Implement gradient checkpointing for memory
- Optimize DataLoader with `num_workers` and `pin_memory`
- Use efficient optimizers (AdamW with proper weight decay)

### Inference Optimization
- Use `torch.jit.script` for model compilation
- Implement key-value caching for generation
- Use quantization for deployment
- Consider ONNX export for production

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Check code style
flake8 src/
black src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- PyTorch team for the excellent deep learning framework
- Hugging Face for inspiration on model implementations

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{my_llm_2024,
  title={My LLM: A Complete Language Model Training Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/my_llm}
}
```