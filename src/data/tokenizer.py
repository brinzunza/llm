import re
import json
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


class SimpleTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_freq = Counter()
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Add spaces around punctuation
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple word-based tokenization"""
        text = self._preprocess_text(text)
        
        # Split by whitespace
        tokens = text.split()
        
        return tokens
    
    def train(self, texts: List[str]):
        """Train tokenizer on a list of texts"""
        print("Training tokenizer...")
        
        # Count word frequencies
        for text in texts:
            tokens = self._tokenize_text(text)
            self.word_freq.update(tokens)
        
        # Create vocabulary
        # Start with special tokens
        vocab = self.special_tokens.copy()
        
        # Add most frequent words
        most_common = self.word_freq.most_common(self.vocab_size - len(self.special_tokens))
        vocab.extend([word for word, _ in most_common])
        
        # Create mappings
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        
        print(f"Vocabulary size: {len(self.word_to_id)}")
        print(f"Most common words: {list(self.word_freq.most_common(10))}")
        
    def encode(self, text: str, add_special_tokens=True) -> List[int]:
        """Encode text to token IDs"""
        tokens = self._tokenize_text(text)
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.word_to_id:
                token_ids.append(self.word_to_id[token])
            else:
                token_ids.append(self.word_to_id[self.unk_token])
                
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens=True) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'word_freq': dict(self.word_freq),
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.vocab_size = tokenizer_data['vocab_size']
        self.word_to_id = tokenizer_data['word_to_id']
        self.id_to_word = tokenizer_data['id_to_word']
        self.word_freq = Counter(tokenizer_data['word_freq'])
        self.special_tokens = tokenizer_data['special_tokens']
        
        print(f"Tokenizer loaded from {filepath}")
    
    @property
    def pad_token_id(self):
        return self.word_to_id[self.pad_token]
    
    @property
    def unk_token_id(self):
        return self.word_to_id[self.unk_token]
    
    @property
    def bos_token_id(self):
        return self.word_to_id[self.bos_token]
    
    @property
    def eos_token_id(self):
        return self.word_to_id[self.eos_token]


if __name__ == "__main__":
    # Test tokenizer
    sample_texts = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating and powerful.",
        "Natural language processing enables computers to understand human language."
    ]
    
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.train(sample_texts)
    
    # Test encoding/decoding
    test_text = "Hello world! This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")