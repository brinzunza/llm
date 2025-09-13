import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import json
import random
from typing import List, Dict, Any, Iterator
import requests
import os
import datasets
from datasets import load_dataset
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > max_length:
                # Split long texts into chunks
                for i in range(0, len(tokens), max_length):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) > 10:  # Only keep chunks with reasonable length
                        self.tokenized_texts.append(chunk)
            else:
                self.tokenized_texts.append(tokens)
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)


def load_sample_data():
    """Load sample text data for training"""
    sample_texts = [
        "The art of programming is the art of organizing complexity.",
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Artificial intelligence refers to the simulation of human intelligence in machines.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "Natural language processing helps computers understand and interpret human language.",
        "Computer vision enables machines to interpret and make decisions based on visual input.",
        "Data science combines domain expertise, programming skills, and knowledge of mathematics and statistics.",
        "Software engineering is the systematic approach to the design, development, and maintenance of software.",
        "Algorithms are step-by-step procedures for calculations, data processing, and automated reasoning.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "JavaScript is a versatile programming language primarily used for web development.",
        "Databases are organized collections of structured information or data stored electronically.",
        "Cloud computing delivers computing services over the internet on a pay-as-you-go basis.",
        "Cybersecurity protects digital information, networks, and systems from digital attacks.",
        "Web development involves building and maintaining websites and web applications.",
        "Mobile app development creates software applications that run on mobile devices.",
        "Version control systems track changes to files and coordinate work among multiple people.",
        "Agile methodology emphasizes iterative development and collaboration between teams.",
        "User experience design focuses on creating products that provide meaningful experiences.",
        "DevOps combines software development and IT operations to shorten development cycles.",
    ]
    
    # Expand the dataset by creating variations
    expanded_texts = []
    for text in sample_texts:
        expanded_texts.append(text)
        # Add some variations
        expanded_texts.append(text.replace(".", " and continues to evolve rapidly."))
        expanded_texts.append(f"In today's world, {text.lower()}")
        expanded_texts.append(f"Understanding {text.lower()}")
    
    return expanded_texts


# Global dataset cache to share loaded datasets between train/val
_dataset_cache = {}
_cache_lock = threading.Lock()

def get_shared_dataset(dataset_key):
    """Get a shared dataset from cache or load it"""
    with _cache_lock:
        if dataset_key not in _dataset_cache:
            print(f"🔄 Loading shared dataset: {dataset_key}")

            # Try to load the dataset with robust error handling
            dataset = None

            # Setup requests with retry logic
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Try primary dataset with retries
            for attempt in range(3):
                try:
                    print(f"🔄 Connecting to dataset - Attempt {attempt + 1}/3...")

                    download_config = datasets.DownloadConfig(
                        resume_download=True,
                        max_retries=3,
                    )

                    if dataset_key == "common_crawl":
                        dataset = load_dataset(
                            "allenai/c4",
                            "en",
                            streaming=True,
                            split="train",
                            download_config=download_config
                        )
                        print("✅ Successfully loaded Common Crawl dataset!")
                        break

                except Exception as e:
                    print(f"❌ Attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        delay = 5 * (2 ** attempt)
                        print(f"⏳ Retrying in {delay} seconds...")
                        time.sleep(delay)

            # Try fallback datasets
            if dataset is None:
                print("\n=== Trying fallback datasets ===")
                fallback_options = [
                    ("wikitext", "wikitext-103-raw-v1", "train"),
                    ("openwebtext", None, "train"),
                    ("bookcorpus", None, "train"),
                ]

                for dataset_name, config_name, split in fallback_options:
                    try:
                        print(f"🔄 Trying fallback dataset: {dataset_name}...")
                        if config_name:
                            dataset = load_dataset(dataset_name, config_name, streaming=True, split=split)
                        else:
                            dataset = load_dataset(dataset_name, streaming=True, split=split)

                        print(f"✅ Successfully connected to fallback dataset: {dataset_name}")
                        break

                    except Exception as e:
                        print(f"❌ Fallback {dataset_name} failed: {e}")
                        continue

            # Final fallback to local data
            if dataset is None:
                print("\n=== Using local sample data as final fallback ===")
                sample_texts = load_sample_data()

                def sample_generator():
                    while True:
                        for text in sample_texts:
                            yield {"text": text}

                dataset = sample_generator()
                print("✅ Using local sample data")

            _dataset_cache[dataset_key] = dataset
            print(f"💾 Dataset cached for reuse: {dataset_key}")

        else:
            print(f"♻️ Reusing cached dataset: {dataset_key}")

        return _dataset_cache[dataset_key]


class CommonCrawlStreamingDataset(IterableDataset):
    """Streaming dataset for Common Crawl dataset"""
    
    def __init__(self, tokenizer, max_length=1024, target_tokens=5_000_000_000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_tokens = target_tokens
        self.tokens_processed = 0
        self.samples_processed = 0
        self.dataset_source = None
    
    def get_dataset_info(self):
        """Get information about the dataset"""
        return {
            'source': self.dataset_source or 'Unknown',
            'target_tokens': self.target_tokens,
            'tokens_processed': self.tokens_processed,
            'samples_processed': self.samples_processed,
            'max_length': self.max_length,
            'progress_percentage': (self.tokens_processed / self.target_tokens) * 100 if self.target_tokens > 0 else 0
        }
        

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Use shared dataset to avoid multiple downloads
        print(f"\n=== Getting dataset for {self.target_tokens:,} tokens ===")

        dataset = get_shared_dataset("common_crawl")

        # Determine dataset source from the cached dataset
        if hasattr(dataset, '__module__') and 'datasets' in str(type(dataset)):
            self.dataset_source = "Common Crawl (allenai/c4) - Shared"
        elif callable(dataset):
            self.dataset_source = "Local sample data - Shared"
        else:
            self.dataset_source = "Fallback dataset - Shared"

        print(f"📊 Dataset source: {self.dataset_source}")
        print(f"🎯 Target tokens: {self.target_tokens:,}")
        print(f"📏 Sequence length: {self.max_length}")
            
        # Progress tracking
        last_progress_report = 0
        progress_interval = max(1000, self.target_tokens // 100)  # Report every 1% or 1000 tokens
        
        print(f"\n🚀 Starting data streaming from {self.dataset_source}")
        print(f"📈 Progress will be reported every {progress_interval:,} tokens")
        
        for sample in dataset:
            if self.tokens_processed >= self.target_tokens:
                print(f"\n🎯 Target reached: {self.tokens_processed:,} tokens processed")
                break
                
            # Extract text from sample (Common Crawl c4 uses 'text' field)
            text = sample.get('text', '') or sample.get('content', '')
            if not text:
                continue
                
            self.samples_processed += 1
            
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Split into chunks of max_length
            for i in range(0, len(tokens), self.max_length):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) < 50:  # Skip very short chunks
                    continue
                    
                # Pad to max_length
                if len(chunk) < self.max_length:
                    chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))
                
                self.tokens_processed += len(chunk)
                
                # Progress reporting
                if self.tokens_processed - last_progress_report >= progress_interval:
                    progress_pct = (self.tokens_processed / self.target_tokens) * 100
                    print(f"📊 Progress: {self.tokens_processed:,}/{self.target_tokens:,} tokens ({progress_pct:.1f}%) | {self.samples_processed:,} samples processed")
                    last_progress_report = self.tokens_processed
                
                yield torch.tensor(chunk, dtype=torch.long)
                
                if self.tokens_processed >= self.target_tokens:
                    break


def download_openwebtext_sample():
    """Download a small sample from OpenWebText for training"""
    print("For a real training dataset, you should download larger corpora like:")
    print("- OpenWebText: https://github.com/jcpeterson/openwebtext")
    print("- WikiText: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/")
    print("- Common Crawl: https://commoncrawl.org/")
    print("\nFor this demo, we'll use generated sample data.")
    
    return load_sample_data()


def load_common_crawl_dataset(tokenizer, max_length=1024, target_tokens=5_000_000_000):
    """Load the Common Crawl dataset for training"""
    print(f"Loading Common Crawl dataset with target of {target_tokens:,} tokens...")
    return CommonCrawlStreamingDataset(tokenizer, max_length, target_tokens)


def create_dataloaders(texts: List[str], tokenizer, batch_size=8, max_length=512, 
                      train_split=0.8, val_split=0.1):
    """Create train, validation, and test dataloaders for traditional datasets"""
    
    # Shuffle texts
    random.shuffle(texts)
    
    # Split data
    n_train = int(len(texts) * train_split)
    n_val = int(len(texts) * val_split)
    
    train_texts = texts[:n_train]
    val_texts = texts[n_train:n_train + n_val]
    test_texts = texts[n_train + n_val:]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)
    test_dataset = TextDataset(test_texts, tokenizer, max_length)
    
    print(f"Train tokens: {len(train_dataset)}")
    print(f"Validation tokens: {len(val_dataset)}")
    print(f"Test tokens: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_common_crawl_dataloaders(tokenizer, config):
    """Create streaming dataloaders for Common Crawl dataset"""
    
    # Extract config values
    batch_size = config.get('training', {}).get('batch_size', 12)
    max_length = config.get('data', {}).get('max_length', 1024)
    target_tokens = config.get('data', {}).get('target_tokens', 5_000_000_000)
    val_split = config.get('data', {}).get('val_split', 0.01)
    
    # Calculate validation tokens
    val_tokens = int(target_tokens * val_split)
    train_tokens = target_tokens - val_tokens
    
    print(f"\n=== Setting up streaming dataloaders ===")
    print(f"🎯 Total target tokens: {target_tokens:,}")
    print(f"📚 Training tokens: {train_tokens:,} ({100*(1-val_split):.1f}%)")
    print(f"🔍 Validation tokens: {val_tokens:,} ({100*val_split:.1f}%)")
    print(f"📦 Batch size: {batch_size}")
    print(f"📏 Sequence length: {max_length}")
    
    # Calculate estimated dataset size metrics
    estimated_batches_per_epoch = train_tokens // (batch_size * max_length)
    estimated_data_size_gb = (target_tokens * 4) / (1024**3)  # Rough estimate: 4 bytes per token
    
    print(f"📊 Estimated batches per epoch: {estimated_batches_per_epoch:,}")
    print(f"💾 Estimated data size: {estimated_data_size_gb:.2f} GB")
    
    # Create streaming datasets that share the same underlying data source
    print(f"\n🔧 Creating training dataset...")
    train_dataset = CommonCrawlStreamingDataset(tokenizer, max_length, train_tokens)
    print(f"🔧 Creating validation dataset (will reuse same data source)...")
    val_dataset = CommonCrawlStreamingDataset(tokenizer, max_length, val_tokens)
    
    # Create dataloaders
    print(f"🔧 Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,  # Keep 0 for streaming datasets
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ Dataloaders created successfully!")
    print(f"🚀 Ready to begin training on dataset: {train_dataset.dataset_source or 'Common Crawl (pending connection)'}")
    
    return train_loader, val_loader, None  # No test loader for streaming


# Backward compatibility alias
create_dolma_dataloaders = create_common_crawl_dataloaders


def load_custom_dataset(file_path: str):
    """Load custom dataset from file"""
    texts = []
    
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by paragraphs or sentences
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = [item['text'] if isinstance(item, dict) else str(item) for item in data]
            elif isinstance(data, dict) and 'texts' in data:
                texts = data['texts']
    
    return texts


if __name__ == "__main__":
    # Test dataset creation
    from tokenizer import SimpleTokenizer
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # Load sample data
    texts = download_openwebtext_sample()
    
    # Train tokenizer
    tokenizer.train(texts)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        texts, tokenizer, batch_size=4, max_length=128
    )
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")
        print(f"Sample tokens: {batch[0][:20]}")
        break