import json
import os
from typing import Dict, Any, Optional


class Config:
    """Configuration management for LLM training and inference"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self.load_default_config()
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"Configuration loaded from {config_path}")
    
    def load_default_config(self):
        """Load default configuration"""
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'configs',
            'default_config.json'
        )
        
        if os.path.exists(default_config_path):
            self.load_config(default_config_path)
        else:
            # Fallback to hardcoded defaults
            self.config = self._get_hardcoded_defaults()
            print("Using hardcoded default configuration")
    
    def _get_hardcoded_defaults(self):
        """Get hardcoded default configuration"""
        return {
            "model": {
                "vocab_size": 32000,
                "d_model": 768,
                "n_heads": 12,
                "n_layers": 12,
                "max_seq_len": 512,
                "dropout": 0.1
            },
            "training": {
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
                "epochs": 10,
                "batch_size": 8,
                "max_seq_length": 512,
                "log_interval": 10,
                "save_interval": 1,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 100,
                "checkpoint_dir": "./models/checkpoints",
                "log_dir": "./models/logs"
            },
            "data": {
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "max_length": 512,
                "tokenizer_vocab_size": 32000
            },
            "evaluation": {
                "batch_size": 16,
                "generation_max_length": 100,
                "generation_temperature": 0.8,
                "generation_top_k": 40,
                "generation_top_p": 0.9
            }
        }
    
    def save_config(self, config_path: str):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")
    
    def get(self, key: str, default: Any = None):
        """Get configuration value by key (supports nested keys with dots)"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports nested keys with dots)"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with a dictionary of values"""
        for key, value in updates.items():
            self.set(key, value)
    
    def get_model_config(self):
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_training_config(self):
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_data_config(self):
        """Get data configuration"""
        return self.config.get('data', {})
    
    def get_evaluation_config(self):
        """Get evaluation configuration"""
        return self.config.get('evaluation', {})
    
    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print(json.dumps(self.config, indent=2))
    
    def validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        # Validate model config
        model_config = self.get_model_config()
        if model_config.get('d_model', 0) <= 0:
            errors.append("model.d_model must be positive")
        if model_config.get('n_heads', 0) <= 0:
            errors.append("model.n_heads must be positive")
        if model_config.get('d_model', 1) % model_config.get('n_heads', 1) != 0:
            errors.append("model.d_model must be divisible by model.n_heads")
        
        # Validate training config
        training_config = self.get_training_config()
        if training_config.get('learning_rate', 0) <= 0:
            errors.append("training.learning_rate must be positive")
        if training_config.get('batch_size', 0) <= 0:
            errors.append("training.batch_size must be positive")
        if training_config.get('epochs', 0) <= 0:
            errors.append("training.epochs must be positive")
        
        # Validate data config
        data_config = self.get_data_config()
        splits_sum = (
            data_config.get('train_split', 0) + 
            data_config.get('val_split', 0) + 
            data_config.get('test_split', 0)
        )
        if abs(splits_sum - 1.0) > 0.01:
            errors.append("data splits (train_split + val_split + test_split) must sum to 1.0")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        print("Configuration validation passed ✓")
    
    def create_experiment_config(self, experiment_name: str, overrides: Dict[str, Any] = None):
        """Create a new experiment configuration"""
        experiment_config = self.config.copy()
        
        if overrides:
            for key, value in overrides.items():
                keys = key.split('.')
                config = experiment_config
                for k in keys[:-1]:
                    if k not in config:
                        config[k] = {}
                    config = config[k]
                config[keys[-1]] = value
        
        # Update paths for experiment
        experiment_config['training']['checkpoint_dir'] = f"./models/experiments/{experiment_name}/checkpoints"
        experiment_config['training']['log_dir'] = f"./models/experiments/{experiment_name}/logs"
        
        return Config.from_dict(experiment_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create Config instance from dictionary"""
        config = cls()
        config.config = config_dict
        return config
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return self.config.copy()


def load_config(config_path: str = None):
    """Convenience function to load configuration"""
    return Config(config_path)


def create_training_config(model_size='default'):
    """Create training configuration for different model sizes"""
    configs = {
        'small': 'small_config.json',
        'default': 'default_config.json'
    }
    
    config_file = configs.get(model_size, 'default_config.json')
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'configs',
        config_file
    )
    
    return Config(config_path)


if __name__ == "__main__":
    # Test configuration management
    config = Config()
    config.print_config()
    
    print("\nTesting configuration access:")
    print(f"Model d_model: {config.get('model.d_model')}")
    print(f"Training batch_size: {config.get('training.batch_size')}")
    
    print("\nTesting configuration updates:")
    config.set('training.batch_size', 16)
    print(f"Updated batch_size: {config.get('training.batch_size')}")
    
    print("\nValidating configuration:")
    config.validate_config()
    
    print("\nConfiguration management test completed successfully!")